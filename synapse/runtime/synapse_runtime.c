#include "synapse_runtime.h"

/* ========================================================================= */
/* Bellek Havuzu (Arena Allocator) - Zero Allocation Overhead               */
/* ========================================================================= */

syn_arena_t* syn_arena_create(size_t capacity) {
    syn_arena_t* arena = (syn_arena_t*)malloc(sizeof(syn_arena_t));
    if (!arena) return NULL;
    arena->buffer = (char*)malloc(capacity);
    arena->capacity = capacity;
    arena->offset = 0;
    return arena;
}

void* syn_arena_alloc(syn_arena_t* arena, size_t bytes) {
    if (!arena || !arena->buffer) return NULL;
    if (bytes > SIZE_MAX - 7) return NULL; /* Arithmetic overflow protection on alignment */
    size_t aligned = (bytes + 7) & ~((size_t)7); /* 8-byte boundary */
    if (aligned > arena->capacity || arena->offset > arena->capacity - aligned) {
        return NULL;
    }
    void* ptr = (void*)(arena->buffer + arena->offset);
    arena->offset += aligned;
    return ptr;
}

void syn_arena_reset(syn_arena_t* arena) {
    if (arena) arena->offset = 0;
}

void syn_arena_free(syn_arena_t* arena) {
    if (!arena) return;
    if (arena->buffer) free(arena->buffer);
    free(arena);
}

syn_arena_scope_t syn_arena_scope_enter(syn_arena_t* arena) {
    if (!arena) {
        syn_arena_scope_t empty_scope = { NULL, 0 };
        return empty_scope;
    }
    syn_arena_scope_t scope;
    scope.arena = arena;
    scope.saved_offset = arena->offset;
    return scope;
}

void syn_arena_scope_leave(syn_arena_scope_t scope) {
    if (scope.arena) {
        scope.arena->offset = scope.saved_offset;
    }
}

#if defined(_MSC_VER)
    #define SYN_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
    #define SYN_THREAD_LOCAL _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
    #define SYN_THREAD_LOCAL __thread
#else
    #define SYN_THREAD_LOCAL
#endif

static SYN_THREAD_LOCAL syn_arena_t* g_syn_active_arena = NULL;

void syn_arena_set_active(syn_arena_t* arena) {
    g_syn_active_arena = arena;
}

syn_arena_t* syn_arena_get_active(void) {
    return g_syn_active_arena;
}

syn_tensor_t* syn_tensor_create_arena(syn_arena_t* arena, const double* initial_data, const int* shape, int ndim, int requires_grad) {
    if (!arena) return syn_tensor_create(initial_data, shape, ndim, requires_grad);
    if (ndim <= 0 || ndim > 32 || !shape) return NULL;
    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return NULL;
        if (shape[i] > 0 && total_elements > SIZE_MAX / (size_t)shape[i]) {
            return NULL; /* Arithmetic overflow in dimensions */
        }
        total_elements *= (size_t)shape[i];
    }
    if (total_elements > SIZE_MAX / sizeof(double)) {
        return NULL; /* Overflow in byte size calculation */
    }

    syn_tensor_t* t = (syn_tensor_t*)syn_arena_alloc(arena, sizeof(syn_tensor_t));
    if (!t) return NULL;
    t->ndim = ndim;
    t->size = (int)total_elements;
    t->shape = (int*)syn_arena_alloc(arena, sizeof(int) * ndim);
    if (!t->shape) return NULL;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }
    t->data = (double*)syn_arena_alloc(arena, sizeof(double) * total_elements);
    if (!t->data) return NULL;
    if (initial_data) {
        memcpy(t->data, initial_data, sizeof(double) * total_elements);
    } else {
        memset(t->data, 0, sizeof(double) * total_elements);
    }
    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->left = NULL;
    t->right = NULL;
    t->op = 0;
    t->scalar_val = 0.0;
    t->visited = 0;
    return t;
}

syn_tensor_t* syn_tensor_zeros_arena(syn_arena_t* arena, const int* shape, int ndim, int requires_grad) {
    return syn_tensor_create_arena(arena, NULL, shape, ndim, requires_grad);
}

static syn_tensor_t* _alloc_tensor(const int* shape, int ndim, int requires_grad) {
    if (g_syn_active_arena) {
        return syn_tensor_zeros_arena(g_syn_active_arena, shape, ndim, requires_grad);
    }
    if (ndim <= 0 || ndim > 32 || !shape) return NULL;
    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return NULL;
        if (shape[i] > 0 && total_elements > SIZE_MAX / (size_t)shape[i]) {
            return NULL; /* Arithmetic overflow in dimensions */
        }
        total_elements *= (size_t)shape[i];
    }
    if (total_elements > SIZE_MAX / sizeof(double)) {
        return NULL; /* Overflow in byte size calculation */
    }

    syn_tensor_t* t = (syn_tensor_t*)malloc(sizeof(syn_tensor_t));
    if (!t) return NULL;
    t->ndim = ndim;
    t->shape = (int*)malloc(sizeof(int) * ndim);
    if (!t->shape) {
        free(t);
        return NULL;
    }
    t->size = (int)total_elements;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }
    t->data = (double*)calloc(total_elements, sizeof(double));
    if (!t->data) {
        free(t->shape);
        free(t);
        return NULL;
    }
    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->left = NULL;
    t->right = NULL;
    t->op = 0;
    t->scalar_val = 0.0;
    t->visited = 0;
    return t;
}

syn_tensor_t* syn_tensor_create(const double* initial_data, const int* shape, int ndim, int requires_grad) {
    syn_tensor_t* t = _alloc_tensor(shape, ndim, requires_grad);
    if (initial_data && t && t->data) {
        memcpy(t->data, initial_data, sizeof(double) * t->size);
    }
    return t;
}

syn_tensor_t* syn_tensor_scalar(double val, int requires_grad) {
    int shape[1] = {1};
    syn_tensor_t* t = _alloc_tensor(shape, 1, requires_grad);
    if (t && t->data) {
        t->data[0] = val;
    }
    return t;
}

syn_tensor_t* syn_tensor_zeros(const int* shape, int ndim, int requires_grad) {
    return _alloc_tensor(shape, ndim, requires_grad);
}

syn_tensor_t* syn_tensor_ones(const int* shape, int ndim, int requires_grad) {
    syn_tensor_t* t = _alloc_tensor(shape, ndim, requires_grad);
    if (t && t->data) {
        for (int i = 0; i < t->size; i++) {
            t->data[i] = 1.0;
        }
    }
    return t;
}

void syn_tensor_free(syn_tensor_t* t) {
    if (!t || !t->data) return;
    if (g_syn_active_arena && g_syn_active_arena->buffer) {
        char* p = (char*)t;
        if (p >= g_syn_active_arena->buffer && p < g_syn_active_arena->buffer + g_syn_active_arena->capacity) {
            return;
        }
    }
    free(t->data);
    t->data = NULL;
    if (t->shape) {
        free(t->shape);
        t->shape = NULL;
    }
    if (t->grad) {
        syn_tensor_free(t->grad);
        t->grad = NULL;
    }
    free(t);
}

double syn_tensor_item(syn_tensor_t* t) {
    if (!t || t->size == 0) return 0.0;
    return t->data[0];
}

void syn_tensor_print(const char* label, syn_tensor_t* t) {
    if (label && strlen(label) > 0) {
        printf("%s: ", label);
    }
    if (!t) {
        printf("null\n");
        return;
    }
    if (t->ndim == 1) {
        printf("tensor([");
        for (int i = 0; i < t->size; i++) {
            printf("%.4f%s", t->data[i], (i == t->size - 1) ? "" : ", ");
        }
        printf("])\n");
    } else if (t->ndim == 2) {
        printf("tensor([\n");
        int rows = t->shape[0];
        int cols = t->shape[1];
        for (int r = 0; r < rows; r++) {
            printf("  [");
            for (int c = 0; c < cols; c++) {
                printf("%.4f%s", t->data[r * cols + c], (c == cols - 1) ? "" : ", ");
            }
            printf("]%s\n", (r == rows - 1) ? "" : ",");
        }
        printf("])\n");
    } else {
        printf("tensor(size=%d)\n", t->size);
    }
}

static syn_tensor_t* syn_tensor_clone_heap(syn_tensor_t* t) {
    if (!t) return NULL;
    syn_tensor_t* clone = (syn_tensor_t*)malloc(sizeof(syn_tensor_t));
    if (!clone) return NULL;
    clone->ndim = t->ndim;
    clone->size = t->size;
    clone->requires_grad = t->requires_grad;
    clone->grad = NULL;
    clone->left = NULL;
    clone->right = NULL;
    clone->op = 0;
    clone->scalar_val = t->scalar_val;
    clone->visited = 0;

    int shape_count = (t->ndim > 0) ? t->ndim : 1;
    clone->shape = (int*)malloc(sizeof(int) * shape_count);
    if (!clone->shape) {
        free(clone);
        return NULL;
    }
    for (int i = 0; i < t->ndim; i++) {
        clone->shape[i] = t->shape[i];
    }

    int data_count = (t->size > 0) ? t->size : 1;
    clone->data = (double*)malloc(sizeof(double) * data_count);
    if (!clone->data) {
        free(clone->shape);
        free(clone);
        return NULL;
    }
    if (t->data && t->size > 0) {
        memcpy(clone->data, t->data, sizeof(double) * t->size);
    } else if (clone->data) {
        clone->data[0] = 0.0;
    }
    return clone;
}

syn_tensor_t* syn_tensor_escape(syn_tensor_t* t, syn_arena_t* arena) {
    if (!t || !arena || !arena->buffer) return t;
    char* p = (char*)t;
    if (p >= arena->buffer && p < arena->buffer + arena->capacity) {
        /* Tensör arena kapsamında tahsis edilmiş; kapsam dışına sızdığı için heap'e kopyalanır */
        return syn_tensor_clone_heap(t);
    }
    return t;
}

syn_tensor_t* syn_tensor_escape_active(syn_tensor_t* t) {
    return syn_tensor_escape(t, g_syn_active_arena);
}

/* ========================================================================= */
/* Aritmetik & Matris Operasyonları                                          */
/* ========================================================================= */

syn_tensor_t* syn_add(syn_tensor_t* a, syn_tensor_t* b) {
    int req_grad = (a->requires_grad || b->requires_grad);
    int a_scalar = (a->size == 1);
    int b_scalar = (b->size == 1);
    syn_tensor_t* out = _alloc_tensor((a_scalar && !b_scalar) ? b->shape : a->shape, (a_scalar && !b_scalar) ? b->ndim : a->ndim, req_grad);
    for (int i = 0; i < out->size; i++) {
        double a_v = a_scalar ? a->data[0] : a->data[i];
        double b_v = b_scalar ? b->data[0] : b->data[i];
        out->data[i] = a_v + b_v;
    }
    out->left = a;
    out->right = b;
    out->op = '+';
    return out;
}

syn_tensor_t* syn_sub(syn_tensor_t* a, syn_tensor_t* b) {
    int req_grad = (a->requires_grad || b->requires_grad);
    int a_scalar = (a->size == 1);
    int b_scalar = (b->size == 1);
    syn_tensor_t* out = _alloc_tensor((a_scalar && !b_scalar) ? b->shape : a->shape, (a_scalar && !b_scalar) ? b->ndim : a->ndim, req_grad);
    for (int i = 0; i < out->size; i++) {
        double a_v = a_scalar ? a->data[0] : a->data[i];
        double b_v = b_scalar ? b->data[0] : b->data[i];
        out->data[i] = a_v - b_v;
    }
    out->left = a;
    out->right = b;
    out->op = '-';
    return out;
}

syn_tensor_t* syn_mul(syn_tensor_t* a, syn_tensor_t* b) {
    int req_grad = (a->requires_grad || b->requires_grad);
    int a_scalar = (a->size == 1);
    int b_scalar = (b->size == 1);
    syn_tensor_t* out = _alloc_tensor((a_scalar && !b_scalar) ? b->shape : a->shape, (a_scalar && !b_scalar) ? b->ndim : a->ndim, req_grad);
    for (int i = 0; i < out->size; i++) {
        double a_v = a_scalar ? a->data[0] : a->data[i];
        double b_v = b_scalar ? b->data[0] : b->data[i];
        out->data[i] = a_v * b_v;
    }
    out->left = a;
    out->right = b;
    out->op = '*';
    return out;
}

syn_tensor_t* syn_div(syn_tensor_t* a, syn_tensor_t* b) {
    int req_grad = (a->requires_grad || b->requires_grad);
    int a_scalar = (a->size == 1);
    int b_scalar = (b->size == 1);
    syn_tensor_t* out = _alloc_tensor((a_scalar && !b_scalar) ? b->shape : a->shape, (a_scalar && !b_scalar) ? b->ndim : a->ndim, req_grad);
    for (int i = 0; i < out->size; i++) {
        double a_v = a_scalar ? a->data[0] : a->data[i];
        double b_v = b_scalar ? b->data[0] : b->data[i];
        out->data[i] = a_v / (b_v == 0.0 ? 1e-12 : b_v);
    }
    out->left = a;
    out->right = b;
    out->op = '/';
    return out;
}

syn_tensor_t* syn_add_scalar(syn_tensor_t* a, double scalar) {
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    for (int b_idx = 0; b_idx < out->size; b_idx += 512) {
        int end = (b_idx + 512 < out->size) ? b_idx + 512 : out->size;
        #pragma omp simd
        for (int i = b_idx; i < end; i++) {
            out->data[i] = a->data[i] + scalar;
        }
    }
#else
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + scalar;
    }
#endif
    out->left = a;
    out->op = '+';
    out->scalar_val = scalar;
    return out;
}

syn_tensor_t* syn_sub_scalar(syn_tensor_t* a, double scalar) {
    return syn_add_scalar(a, -scalar);
}

syn_tensor_t* syn_mul_scalar(syn_tensor_t* a, double scalar) {
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    for (int b_idx = 0; b_idx < out->size; b_idx += 512) {
        int end = (b_idx + 512 < out->size) ? b_idx + 512 : out->size;
        #pragma omp simd
        for (int i = b_idx; i < end; i++) {
            out->data[i] = a->data[i] * scalar;
        }
    }
#else
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * scalar;
    }
#endif
    out->left = a;
    out->op = 'm'; /* mul scalar */
    out->scalar_val = scalar;
    return out;
}

syn_tensor_t* syn_pow_scalar(syn_tensor_t* a, double power) {
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = pow(a->data[i], power);
    }
    out->left = a;
    out->op = 'P';
    out->scalar_val = power;
    return out;
}

syn_tensor_t* syn_matmul(syn_tensor_t* a, syn_tensor_t* b) {
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Synapse Runtime Error: Incompatible shapes for matmul\n");
        exit(1);
    }
    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];
    int out_shape[2] = {m, n};

    int req_grad = (a->requires_grad || b->requires_grad);
    syn_tensor_t* out = _alloc_tensor(out_shape, 2, req_grad);

    /* Küçük matrisler için direkt, orta/büyükler için Cache-blocked & Transpose optimizasyonu */
    if (k * n >= 256) {
        double* b_t = (double*)malloc(sizeof(double) * k * n);
        if (b_t) {
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int r = 0; r < k; r++) {
                for (int c = 0; c < n; c++) {
                    b_t[c * k + r] = b->data[r * n + c];
                }
            }
            const int TILE = 32;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int i0 = 0; i0 < m; i0 += TILE) {
                int imax = (i0 + TILE < m) ? i0 + TILE : m;
                for (int j0 = 0; j0 < n; j0 += TILE) {
                    int jmax = (j0 + TILE < n) ? j0 + TILE : n;
                    for (int i = i0; i < imax; i++) {
                        const double* a_row = &a->data[i * k];
                        double* out_row = &out->data[i * n];
                        for (int j = j0; j < jmax; j++) {
                            const double* b_row = &b_t[j * k];
                            double sum = 0.0;
                            int p = 0;
                            /* 4x loop unrolling for SIMD vectorization */
#if defined(_OPENMP)
                            #pragma omp simd reduction(+:sum)
#endif
                            for (; p + 3 < k; p += 4) {
                                sum += a_row[p] * b_row[p]
                                     + a_row[p + 1] * b_row[p + 1]
                                     + a_row[p + 2] * b_row[p + 2]
                                     + a_row[p + 3] * b_row[p + 3];
                            }
                            for (; p < k; p++) {
                                sum += a_row[p] * b_row[p];
                            }
                            out_row[j] += sum;
                        }
                    }
                }
            }
            free(b_t);
            out->left = a;
            out->right = b;
            out->op = '@';
            return out;
        }
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
#if defined(_OPENMP)
            #pragma omp simd reduction(+:sum)
#endif
            for (int p = 0; p < k; p++) {
                sum += a->data[i * k + p] * b->data[p * n + j];
            }
            out->data[i * n + j] = sum;
        }
    }
    out->left = a;
    out->right = b;
    out->op = '@';
    return out;
}

syn_tensor_t* syn_transpose(syn_tensor_t* a) {
    if (a->ndim != 2) return a;
    int r = a->shape[0];
    int c = a->shape[1];
    int t_shape[2] = {c, r};

    syn_tensor_t* out = _alloc_tensor(t_shape, 2, a->requires_grad);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            out->data[j * r + i] = a->data[i * c + j];
        }
    }
    out->left = a;
    out->op = 'T';
    return out;
}

syn_tensor_t* syn_relu(syn_tensor_t* a) {
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
    }
    out->left = a;
    out->op = 'R';
    return out;
}

syn_tensor_t* syn_sigmoid(syn_tensor_t* a) {
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = 1.0 / (1.0 + exp(-a->data[i]));
    }
    out->left = a;
    out->op = 's';
    return out;
}

syn_tensor_t* syn_tanh(syn_tensor_t* a) {
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = tanh(a->data[i]);
    }
    out->left = a;
    out->op = 't';
    return out;
}

syn_tensor_t* syn_gelu(syn_tensor_t* a) {
    const double sqrt_2_over_pi = 0.7978845608028654;
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
    for (int i = 0; i < out->size; i++) {
        double x = a->data[i];
        double inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        out->data[i] = 0.5 * x * (1.0 + tanh(inner));
    }
    out->left = a;
    out->op = 'g';
    return out;
}

syn_tensor_t* syn_softmax(syn_tensor_t* a) {
    syn_tensor_t* out = _alloc_tensor(a->shape, a->ndim, a->requires_grad);
    if (a->ndim == 1) {
        double max_v = -1e30;
        for (int i = 0; i < a->size; i++) if (a->data[i] > max_v) max_v = a->data[i];
        double sum = 0.0;
        for (int i = 0; i < a->size; i++) {
            out->data[i] = exp(a->data[i] - max_v);
            sum += out->data[i];
        }
        for (int i = 0; i < a->size; i++) out->data[i] /= (sum > 0 ? sum : 1.0);
    } else if (a->ndim == 2) {
        int rows = a->shape[0];
        int cols = a->shape[1];
        for (int r = 0; r < rows; r++) {
            double max_v = -1e30;
            for (int c = 0; c < cols; c++) {
                double v = a->data[r * cols + c];
                if (v > max_v) max_v = v;
            }
            double sum = 0.0;
            for (int c = 0; c < cols; c++) {
                double exp_v = exp(a->data[r * cols + c] - max_v);
                out->data[r * cols + c] = exp_v;
                sum += exp_v;
            }
            for (int c = 0; c < cols; c++) {
                out->data[r * cols + c] /= (sum > 0 ? sum : 1.0);
            }
        }
    } else {
        memcpy(out->data, a->data, sizeof(double) * a->size);
    }
    out->left = a;
    out->op = 'M';
    return out;
}

syn_tensor_t* syn_sum(syn_tensor_t* a) {
    int s_shape[1] = {1};
    syn_tensor_t* out = _alloc_tensor(s_shape, 1, a->requires_grad);
    double sum = 0.0;
#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int b_idx = 0; b_idx < a->size; b_idx += 512) {
        int end = (b_idx + 512 < a->size) ? b_idx + 512 : a->size;
        double local_sum = 0.0;
        #pragma omp simd reduction(+:local_sum)
        for (int i = b_idx; i < end; i++) {
            local_sum += a->data[i];
        }
        sum += local_sum;
    }
#else
    for (int i = 0; i < a->size; i++) {
        sum += a->data[i];
    }
#endif
    out->data[0] = sum;
    out->left = a;
    out->op = 'S';
    return out;
}

syn_tensor_t* syn_mean(syn_tensor_t* a) {
    int s_shape[1] = {1};
    syn_tensor_t* out = _alloc_tensor(s_shape, 1, a->requires_grad);
    double sum = 0.0;
#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int b_idx = 0; b_idx < a->size; b_idx += 512) {
        int end = (b_idx + 512 < a->size) ? b_idx + 512 : a->size;
        double local_sum = 0.0;
        #pragma omp simd reduction(+:local_sum)
        for (int i = b_idx; i < end; i++) {
            local_sum += a->data[i];
        }
        sum += local_sum;
    }
#else
    for (int i = 0; i < a->size; i++) {
        sum += a->data[i];
    }
#endif
    out->data[0] = sum / (a->size > 0 ? (double)a->size : 1.0);
    out->left = a;
    out->op = 'e';
    return out;
}

/* ========================================================================= */
/* Autograd: Topolojik Sıralama & Reverse-Mode Geri Yayılım                  */
/* ========================================================================= */

static void _ensure_grad(syn_tensor_t* t) {
    if (!t->grad) {
        t->grad = syn_tensor_zeros(t->shape, t->ndim, 0);
    }
}

static void _build_topo(syn_tensor_t* v, syn_tensor_t*** topo_list, int* topo_count, int* topo_capacity) {
    if (!v || v->visited) return;
    v->visited = 1;
    if (v->left) _build_topo(v->left, topo_list, topo_count, topo_capacity);
    if (v->right) _build_topo(v->right, topo_list, topo_count, topo_capacity);
    if (*topo_count >= *topo_capacity) {
        int new_cap = (*topo_capacity) * 2;
        if (new_cap < 1024) new_cap = 1024;
        syn_tensor_t** new_list = (syn_tensor_t**)realloc(*topo_list, sizeof(syn_tensor_t*) * new_cap);
        if (new_list) {
            *topo_list = new_list;
            *topo_capacity = new_cap;
        }
    }
    if (*topo_count < *topo_capacity) {
        (*topo_list)[(*topo_count)++] = v;
    }
}

void syn_backward(syn_tensor_t* root) {
    if (!root) return;
    _ensure_grad(root);
    for (int i = 0; i < root->size; i++) {
        root->grad->data[i] = 1.0;
    }

    int topo_capacity = 1024;
    int topo_count = 0;
    syn_tensor_t** topo_list = (syn_tensor_t**)malloc(sizeof(syn_tensor_t*) * topo_capacity);
    if (!topo_list) return;
    _build_topo(root, &topo_list, &topo_count, &topo_capacity);

    /* Düğümleri sondan başa doğru işlet */
    for (int k = topo_count - 1; k >= 0; k--) {
        syn_tensor_t* node = topo_list[k];
        node->visited = 0; /* reset */

        if (!node->grad || node->op == 0) continue;

        if (node->op == '+') {
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                for (int i = 0; i < node->left->size; i++) node->left->grad->data[i] += node->grad->data[i];
            }
            if (node->right && node->right->requires_grad) {
                _ensure_grad(node->right);
                for (int i = 0; i < node->right->size; i++) node->right->grad->data[i] += node->grad->data[i];
            }
        } else if (node->op == '-') {
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                for (int i = 0; i < node->left->size; i++) node->left->grad->data[i] += node->grad->data[i];
            }
            if (node->right && node->right->requires_grad) {
                _ensure_grad(node->right);
                for (int i = 0; i < node->right->size; i++) node->right->grad->data[i] -= node->grad->data[i];
            }
        } else if (node->op == '*') {
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                for (int i = 0; i < node->left->size; i++) node->left->grad->data[i] += node->grad->data[i] * node->right->data[i];
            }
            if (node->right && node->right->requires_grad) {
                _ensure_grad(node->right);
                for (int i = 0; i < node->right->size; i++) node->right->grad->data[i] += node->grad->data[i] * node->left->data[i];
            }
        } else if (node->op == '/') {
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                for (int i = 0; i < node->left->size; i++) {
                    double b_val = (node->right->data[i] == 0.0 ? 1e-12 : node->right->data[i]);
                    node->left->grad->data[i] += node->grad->data[i] / b_val;
                }
            }
            if (node->right && node->right->requires_grad) {
                _ensure_grad(node->right);
                for (int i = 0; i < node->right->size; i++) {
                    double b_val = (node->right->data[i] == 0.0 ? 1e-12 : node->right->data[i]);
                    node->right->grad->data[i] -= node->grad->data[i] * node->left->data[i] / (b_val * b_val);
                }
            }
        } else if (node->op == 'm') { /* mul scalar */
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                for (int i = 0; i < node->left->size; i++) node->left->grad->data[i] += node->grad->data[i] * node->scalar_val;
            }
        } else if (node->op == 'P') { /* pow scalar */
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                double p = node->scalar_val;
                for (int i = 0; i < node->left->size; i++) {
                    node->left->grad->data[i] += node->grad->data[i] * p * pow(node->left->data[i], p - 1.0);
                }
            }
        } else if (node->op == 'R') { /* relu */
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                for (int i = 0; i < node->left->size; i++) {
                    if (node->left->data[i] > 0.0) {
                        node->left->grad->data[i] += node->grad->data[i];
                    }
                }
            }
        } else if (node->op == 'S') { /* sum */
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                double g = node->grad->data[0];
                for (int i = 0; i < node->left->size; i++) {
                    node->left->grad->data[i] += g;
                }
            }
        } else if (node->op == 'e') { /* mean */
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                double g = node->grad->data[0] / (node->left->size > 0 ? (double)node->left->size : 1.0);
                for (int i = 0; i < node->left->size; i++) {
                    node->left->grad->data[i] += g;
                }
            }
        } else if (node->op == '@') { /* matmul */
            /* dA = dC @ B.T, dB = A.T @ dC */
            if (node->left && node->left->requires_grad) {
                _ensure_grad(node->left);
                syn_tensor_t* bt = syn_transpose(node->right);
                syn_tensor_t* d_left = syn_matmul(node->grad, bt);
                for (int i = 0; i < node->left->size; i++) node->left->grad->data[i] += d_left->data[i];
                syn_tensor_free(bt);
                syn_tensor_free(d_left);
            }
            if (node->right && node->right->requires_grad) {
                _ensure_grad(node->right);
                syn_tensor_t* at = syn_transpose(node->left);
                syn_tensor_t* d_right = syn_matmul(at, node->grad);
                for (int i = 0; i < node->right->size; i++) node->right->grad->data[i] += d_right->data[i];
                syn_tensor_free(at);
                syn_tensor_free(d_right);
            }
        }
    }
    free(topo_list);
}

void syn_tensor_zero_grad(syn_tensor_t* t) {
    if (!t || !t->grad) return;
    for (int i = 0; i < t->grad->size; i++) {
        t->grad->data[i] = 0.0;
    }
}

void syn_graph_free(syn_tensor_t* root) {
    if (!root) return;
    int topo_capacity = 1024;
    int topo_count = 0;
    syn_tensor_t** topo_list = (syn_tensor_t**)malloc(sizeof(syn_tensor_t*) * topo_capacity);
    if (!topo_list) return;
    _build_topo(root, &topo_list, &topo_count, &topo_capacity);

    for (int k = 0; k < topo_count; k++) {
        syn_tensor_t* node = topo_list[k];
        node->visited = 0;
        if (node != root && node->op != 0) {
            syn_tensor_free(node);
        } else if (node->grad) {
            syn_tensor_free(node->grad);
            node->grad = NULL;
        }
    }
    free(topo_list);
}

syn_tensor_t* syn_tensor_clone(syn_tensor_t* t) {
    if (!t) return NULL;
    syn_tensor_t* clone = _alloc_tensor(t->shape, t->ndim, t->requires_grad);
    if (t->data) {
        memcpy(clone->data, t->data, sizeof(double) * t->size);
    }
    return clone;
}

void syn_print_double(const char* label, double val) {
    if (label && strlen(label) > 0) {
        printf("%s: %.6f\n", label, val);
    } else {
        printf("%.6f\n", val);
    }
}

void syn_print_int(const char* label, int val) {
    if (label && strlen(label) > 0) {
        printf("%s: %d\n", label, val);
    } else {
        printf("%d\n", val);
    }
}

void syn_print_str(const char* val) {
    if (val) {
        printf("%s\n", val);
    }
}

/* ========================================================================= */
/* HPC & OpenMP Donanım Bilgisi                                              */
/* ========================================================================= */

int syn_openmp_info(void) {
#if defined(_OPENMP)
    int threads = omp_get_max_threads();
    return (threads > 0) ? threads : 1;
#else
    return 1;
#endif
}

/* ========================================================================= */
/* Eşzamanlılık, Görevler & CSP Kanalları (Concurrency & CSP Channels)       */
/* ========================================================================= */

#if defined(__EMSCRIPTEN__)

/* WebAssembly Tek İş Parçacıklı Cooperative FIFO Mikro-Görev Kuyruğu */
static syn_task_t* g_wasm_task_head = NULL;
static syn_task_t* g_wasm_task_tail = NULL;
static int g_wasm_pending_count = 0;

void syn_wasm_enqueue_task(syn_task_fn fn, void* arg) {
    if (!fn) return;
    syn_task_t* task = (syn_task_t*)malloc(sizeof(syn_task_t));
    if (!task) return;
    task->fn = fn;
    task->arg = arg;
    task->next = NULL;

    if (!g_wasm_task_tail) {
        g_wasm_task_head = task;
        g_wasm_task_tail = task;
    } else {
        g_wasm_task_tail->next = task;
        g_wasm_task_tail = task;
    }
    g_wasm_pending_count++;
}

int syn_wasm_pending_tasks(void) {
    return g_wasm_pending_count;
}

void syn_wasm_clear_tasks(void) {
    syn_task_t* curr = g_wasm_task_head;
    while (curr) {
        syn_task_t* nxt = curr->next;
        free(curr);
        curr = nxt;
    }
    g_wasm_task_head = NULL;
    g_wasm_task_tail = NULL;
    g_wasm_pending_count = 0;
}

int syn_wasm_run_microtasks(int max_tasks) {
    int executed = 0;
    while (g_wasm_task_head != NULL && (max_tasks <= 0 || executed < max_tasks)) {
        syn_task_t* task = g_wasm_task_head;
        g_wasm_task_head = task->next;
        if (!g_wasm_task_head) {
            g_wasm_task_tail = NULL;
        }
        g_wasm_pending_count--;

        if (task->fn) {
            task->fn(task->arg);
        }
        free(task);
        executed++;
    }
    return executed;
}

typedef struct syn_wasm_task_wrapper {
    syn_task_fn fn;
    void* arg;
    syn_task_handle_t* handle;
} syn_wasm_task_wrapper_t;

static void _syn_wasm_wrapper_fn(void* ctx) {
    syn_wasm_task_wrapper_t* wrapper = (syn_wasm_task_wrapper_t*)ctx;
    if (wrapper) {
        if (wrapper->fn) {
            wrapper->fn(wrapper->arg);
        }
        if (wrapper->handle) {
            wrapper->handle->done = 1;
        }
        free(wrapper);
    }
}

syn_task_handle_t* syn_spawn(syn_task_fn fn, void* arg) {
    syn_task_handle_t* handle = (syn_task_handle_t*)calloc(1, sizeof(syn_task_handle_t));
    if (!handle) return NULL;
    static int s_wasm_task_seq = 1;
    handle->id = s_wasm_task_seq++;
    handle->done = 0;

    syn_wasm_task_wrapper_t* wrapper = (syn_wasm_task_wrapper_t*)malloc(sizeof(syn_wasm_task_wrapper_t));
    if (!wrapper) {
        free(handle);
        return NULL;
    }
    wrapper->fn = fn;
    wrapper->arg = arg;
    wrapper->handle = handle;

    syn_wasm_enqueue_task(_syn_wasm_wrapper_fn, wrapper);
    return handle;
}

void syn_task_wait(syn_task_handle_t* handle) {
    if (!handle) return;
    while (!handle->done && g_wasm_task_head != NULL) {
        syn_wasm_run_microtasks(1);
    }
}

void syn_task_yield(void) {
    if (g_wasm_task_head != NULL) {
        syn_wasm_run_microtasks(1);
    }
}

void syn_task_handle_free(syn_task_handle_t* handle) {
    if (handle) {
        free(handle);
    }
}

syn_channel_t* syn_channel_create(int capacity) {
    syn_channel_t* ch = (syn_channel_t*)calloc(1, sizeof(syn_channel_t));
    if (!ch) return NULL;
    ch->capacity = (capacity <= 0) ? 1 : capacity;
    ch->buffer = (void**)calloc(ch->capacity, sizeof(void*));
    ch->head = 0;
    ch->tail = 0;
    ch->count = 0;
    ch->closed = 0;
    ch->is_wasm = 1;
    return ch;
}

int syn_channel_send(syn_channel_t* ch, void* data) {
    if (!ch || ch->closed) return 0;

    int loop_guard = 1000;
    while (ch->count >= ch->capacity && g_wasm_pending_count > 0 && loop_guard-- > 0) {
        syn_wasm_run_microtasks(1);
        if (ch->closed) return 0;
    }

    if (ch->count >= ch->capacity) {
        int new_cap = ch->capacity * 2;
        void** new_buf = (void**)calloc(new_cap, sizeof(void*));
        if (!new_buf) return 0;
        for (int i = 0; i < ch->count; i++) {
            new_buf[i] = ch->buffer[(ch->head + i) % ch->capacity];
        }
        free(ch->buffer);
        ch->buffer = new_buf;
        ch->head = 0;
        ch->tail = ch->count;
        ch->capacity = new_cap;
    }

    ch->buffer[ch->tail] = data;
    ch->tail = (ch->tail + 1) % ch->capacity;
    ch->count++;
    return 1;
}

void* syn_channel_recv(syn_channel_t* ch) {
    if (!ch) return NULL;

    while (ch->count == 0 && !ch->closed && g_wasm_pending_count > 0) {
        syn_wasm_run_microtasks(1);
    }

    if (ch->count == 0) {
        return NULL;
    }

    void* data = ch->buffer[ch->head];
    ch->buffer[ch->head] = NULL;
    ch->head = (ch->head + 1) % ch->capacity;
    ch->count--;
    return data;
}

int syn_channel_try_recv(syn_channel_t* ch, void** out_data) {
    if (!ch || ch->count == 0) {
        if (out_data) *out_data = NULL;
        return 0;
    }
    void* data = ch->buffer[ch->head];
    ch->buffer[ch->head] = NULL;
    ch->head = (ch->head + 1) % ch->capacity;
    ch->count--;
    if (out_data) *out_data = data;
    return 1;
}

void syn_channel_close(syn_channel_t* ch) {
    if (ch) {
        ch->closed = 1;
    }
}

int syn_channel_is_closed(syn_channel_t* ch) {
    return ch ? ch->closed : 1;
}

int syn_channel_count(syn_channel_t* ch) {
    return ch ? ch->count : 0;
}

void syn_channel_free(syn_channel_t* ch) {
    if (!ch) return;
    if (ch->buffer) free(ch->buffer);
    free(ch);
}

#else

/* Standart POSIX / Windows Çoklu İş Parçacığı & CSP Kanalları */
#if defined(_WIN32)
#include <windows.h>
#include <process.h>

typedef struct syn_win_chan_sync {
    CRITICAL_SECTION cs;
    CONDITION_VARIABLE cv_send;
    CONDITION_VARIABLE cv_recv;
} syn_win_chan_sync_t;

syn_channel_t* syn_channel_create(int capacity) {
    syn_channel_t* ch = (syn_channel_t*)calloc(1, sizeof(syn_channel_t));
    if (!ch) return NULL;
    ch->capacity = (capacity <= 0) ? 1 : capacity;
    ch->buffer = (void**)calloc(ch->capacity, sizeof(void*));
    ch->head = 0;
    ch->tail = 0;
    ch->count = 0;
    ch->closed = 0;

    syn_win_chan_sync_t* sync = (syn_win_chan_sync_t*)malloc(sizeof(syn_win_chan_sync_t));
    if (sync) {
        InitializeCriticalSection(&sync->cs);
        InitializeConditionVariable(&sync->cv_send);
        InitializeConditionVariable(&sync->cv_recv);
    }
    ch->mutex = (void*)sync;
    return ch;
}

int syn_channel_send(syn_channel_t* ch, void* data) {
    if (!ch || ch->closed) return 0;
    syn_win_chan_sync_t* sync = (syn_win_chan_sync_t*)ch->mutex;
    if (!sync) return 0;

    EnterCriticalSection(&sync->cs);
    while (ch->count >= ch->capacity && !ch->closed) {
        SleepConditionVariableCS(&sync->cv_send, &sync->cs, INFINITE);
    }
    if (ch->closed) {
        LeaveCriticalSection(&sync->cs);
        return 0;
    }

    ch->buffer[ch->tail] = data;
    ch->tail = (ch->tail + 1) % ch->capacity;
    ch->count++;

    WakeConditionVariable(&sync->cv_recv);
    LeaveCriticalSection(&sync->cs);
    return 1;
}

void* syn_channel_recv(syn_channel_t* ch) {
    if (!ch) return NULL;
    syn_win_chan_sync_t* sync = (syn_win_chan_sync_t*)ch->mutex;
    if (!sync) return NULL;

    EnterCriticalSection(&sync->cs);
    while (ch->count == 0 && !ch->closed) {
        SleepConditionVariableCS(&sync->cv_recv, &sync->cs, INFINITE);
    }
    if (ch->count == 0 && ch->closed) {
        LeaveCriticalSection(&sync->cs);
        return NULL;
    }

    void* data = ch->buffer[ch->head];
    ch->buffer[ch->head] = NULL;
    ch->head = (ch->head + 1) % ch->capacity;
    ch->count--;

    WakeConditionVariable(&sync->cv_send);
    LeaveCriticalSection(&sync->cs);
    return data;
}

int syn_channel_try_recv(syn_channel_t* ch, void** out_data) {
    if (!ch) {
        if (out_data) *out_data = NULL;
        return 0;
    }
    syn_win_chan_sync_t* sync = (syn_win_chan_sync_t*)ch->mutex;
    if (!sync) return 0;

    EnterCriticalSection(&sync->cs);
    if (ch->count == 0) {
        LeaveCriticalSection(&sync->cs);
        if (out_data) *out_data = NULL;
        return 0;
    }
    void* data = ch->buffer[ch->head];
    ch->buffer[ch->head] = NULL;
    ch->head = (ch->head + 1) % ch->capacity;
    ch->count--;

    WakeConditionVariable(&sync->cv_send);
    LeaveCriticalSection(&sync->cs);
    if (out_data) *out_data = data;
    return 1;
}

void syn_channel_close(syn_channel_t* ch) {
    if (!ch) return;
    syn_win_chan_sync_t* sync = (syn_win_chan_sync_t*)ch->mutex;
    if (sync) {
        EnterCriticalSection(&sync->cs);
        ch->closed = 1;
        WakeAllConditionVariable(&sync->cv_recv);
        WakeAllConditionVariable(&sync->cv_send);
        LeaveCriticalSection(&sync->cs);
    } else {
        ch->closed = 1;
    }
}

int syn_channel_is_closed(syn_channel_t* ch) {
    return ch ? ch->closed : 1;
}

int syn_channel_count(syn_channel_t* ch) {
    if (!ch) return 0;
    syn_win_chan_sync_t* sync = (syn_win_chan_sync_t*)ch->mutex;
    int c = 0;
    if (sync) {
        EnterCriticalSection(&sync->cs);
        c = ch->count;
        LeaveCriticalSection(&sync->cs);
    } else {
        c = ch->count;
    }
    return c;
}

void syn_channel_free(syn_channel_t* ch) {
    if (!ch) return;
    syn_win_chan_sync_t* sync = (syn_win_chan_sync_t*)ch->mutex;
    if (sync) {
        DeleteCriticalSection(&sync->cs);
        free(sync);
    }
    if (ch->buffer) free(ch->buffer);
    free(ch);
}

typedef struct syn_win_task_wrapper {
    syn_task_fn fn;
    void* arg;
    syn_task_handle_t* handle;
} syn_win_task_wrapper_t;

static unsigned __stdcall _syn_win_thread_proc(void* param) {
    syn_win_task_wrapper_t* wrapper = (syn_win_task_wrapper_t*)param;
    if (wrapper) {
        if (wrapper->fn) {
            wrapper->fn(wrapper->arg);
        }
        if (wrapper->handle) {
            wrapper->handle->done = 1;
        }
        free(wrapper);
    }
    return 0;
}

syn_task_handle_t* syn_spawn(syn_task_fn fn, void* arg) {
    syn_task_handle_t* handle = (syn_task_handle_t*)calloc(1, sizeof(syn_task_handle_t));
    if (!handle) return NULL;
    static int s_win_task_seq = 1;
    handle->id = s_win_task_seq++;
    handle->done = 0;

    syn_win_task_wrapper_t* wrapper = (syn_win_task_wrapper_t*)malloc(sizeof(syn_win_task_wrapper_t));
    if (!wrapper) {
        free(handle);
        return NULL;
    }
    wrapper->fn = fn;
    wrapper->arg = arg;
    wrapper->handle = handle;

    uintptr_t th = _beginthreadex(NULL, 0, _syn_win_thread_proc, wrapper, 0, NULL);
    handle->thread_handle = (void*)th;
    return handle;
}

void syn_task_wait(syn_task_handle_t* handle) {
    if (!handle) return;
    if (handle->thread_handle) {
        WaitForSingleObject((HANDLE)handle->thread_handle, INFINITE);
        CloseHandle((HANDLE)handle->thread_handle);
        handle->thread_handle = NULL;
    }
    handle->done = 1;
}

void syn_task_yield(void) {
    SwitchToThread();
}

void syn_task_handle_free(syn_task_handle_t* handle) {
    if (!handle) return;
    if (handle->thread_handle) {
        CloseHandle((HANDLE)handle->thread_handle);
    }
    free(handle);
}

#else
/* POSIX pthreads */
#include <pthread.h>
#include <sched.h>

typedef struct syn_posix_chan_sync {
    pthread_mutex_t mutex;
    pthread_cond_t cv_send;
    pthread_cond_t cv_recv;
} syn_posix_chan_sync_t;

syn_channel_t* syn_channel_create(int capacity) {
    syn_channel_t* ch = (syn_channel_t*)calloc(1, sizeof(syn_channel_t));
    if (!ch) return NULL;
    ch->capacity = (capacity <= 0) ? 1 : capacity;
    ch->buffer = (void**)calloc(ch->capacity, sizeof(void*));
    ch->head = 0;
    ch->tail = 0;
    ch->count = 0;
    ch->closed = 0;

    syn_posix_chan_sync_t* sync = (syn_posix_chan_sync_t*)malloc(sizeof(syn_posix_chan_sync_t));
    if (sync) {
        pthread_mutex_init(&sync->mutex, NULL);
        pthread_cond_init(&sync->cv_send, NULL);
        pthread_cond_init(&sync->cv_recv, NULL);
    }
    ch->mutex = (void*)sync;
    return ch;
}

int syn_channel_send(syn_channel_t* ch, void* data) {
    if (!ch || ch->closed) return 0;
    syn_posix_chan_sync_t* sync = (syn_posix_chan_sync_t*)ch->mutex;
    if (!sync) return 0;

    pthread_mutex_lock(&sync->mutex);
    while (ch->count >= ch->capacity && !ch->closed) {
        pthread_cond_wait(&sync->cv_send, &sync->mutex);
    }
    if (ch->closed) {
        pthread_mutex_unlock(&sync->mutex);
        return 0;
    }

    ch->buffer[ch->tail] = data;
    ch->tail = (ch->tail + 1) % ch->capacity;
    ch->count++;

    pthread_cond_signal(&sync->cv_recv);
    pthread_mutex_unlock(&sync->mutex);
    return 1;
}

void* syn_channel_recv(syn_channel_t* ch) {
    if (!ch) return NULL;
    syn_posix_chan_sync_t* sync = (syn_posix_chan_sync_t*)ch->mutex;
    if (!sync) return NULL;

    pthread_mutex_lock(&sync->mutex);
    while (ch->count == 0 && !ch->closed) {
        pthread_cond_wait(&sync->cv_recv, &sync->mutex);
    }
    if (ch->count == 0 && ch->closed) {
        pthread_mutex_unlock(&sync->mutex);
        return NULL;
    }

    void* data = ch->buffer[ch->head];
    ch->buffer[ch->head] = NULL;
    ch->head = (ch->head + 1) % ch->capacity;
    ch->count--;

    pthread_cond_signal(&sync->cv_send);
    pthread_mutex_unlock(&sync->mutex);
    return data;
}

int syn_channel_try_recv(syn_channel_t* ch, void** out_data) {
    if (!ch) {
        if (out_data) *out_data = NULL;
        return 0;
    }
    syn_posix_chan_sync_t* sync = (syn_posix_chan_sync_t*)ch->mutex;
    if (!sync) return 0;

    pthread_mutex_lock(&sync->mutex);
    if (ch->count == 0) {
        pthread_mutex_unlock(&sync->mutex);
        if (out_data) *out_data = NULL;
        return 0;
    }
    void* data = ch->buffer[ch->head];
    ch->buffer[ch->head] = NULL;
    ch->head = (ch->head + 1) % ch->capacity;
    ch->count--;

    pthread_cond_signal(&sync->cv_send);
    pthread_mutex_unlock(&sync->mutex);
    if (out_data) *out_data = data;
    return 1;
}

void syn_channel_close(syn_channel_t* ch) {
    if (!ch) return;
    syn_posix_chan_sync_t* sync = (syn_posix_chan_sync_t*)ch->mutex;
    if (sync) {
        pthread_mutex_lock(&sync->mutex);
        ch->closed = 1;
        pthread_cond_broadcast(&sync->cv_recv);
        pthread_cond_broadcast(&sync->cv_send);
        pthread_mutex_unlock(&sync->mutex);
    } else {
        ch->closed = 1;
    }
}

int syn_channel_is_closed(syn_channel_t* ch) {
    return ch ? ch->closed : 1;
}

int syn_channel_count(syn_channel_t* ch) {
    if (!ch) return 0;
    syn_posix_chan_sync_t* sync = (syn_posix_chan_sync_t*)ch->mutex;
    int c = 0;
    if (sync) {
        pthread_mutex_lock(&sync->mutex);
        c = ch->count;
        pthread_mutex_unlock(&sync->mutex);
    } else {
        c = ch->count;
    }
    return c;
}

void syn_channel_free(syn_channel_t* ch) {
    if (!ch) return;
    syn_posix_chan_sync_t* sync = (syn_posix_chan_sync_t*)ch->mutex;
    if (sync) {
        pthread_mutex_destroy(&sync->mutex);
        pthread_cond_destroy(&sync->cv_send);
        pthread_cond_destroy(&sync->cv_recv);
        free(sync);
    }
    if (ch->buffer) free(ch->buffer);
    free(ch);
}

typedef struct syn_posix_task_wrapper {
    syn_task_fn fn;
    void* arg;
    syn_task_handle_t* handle;
} syn_posix_task_wrapper_t;

static void* _syn_posix_thread_proc(void* param) {
    syn_posix_task_wrapper_t* wrapper = (syn_posix_task_wrapper_t*)param;
    if (wrapper) {
        if (wrapper->fn) {
            wrapper->fn(wrapper->arg);
        }
        if (wrapper->handle) {
            wrapper->handle->done = 1;
        }
        free(wrapper);
    }
    return NULL;
}

syn_task_handle_t* syn_spawn(syn_task_fn fn, void* arg) {
    syn_task_handle_t* handle = (syn_task_handle_t*)calloc(1, sizeof(syn_task_handle_t));
    if (!handle) return NULL;
    static int s_posix_task_seq = 1;
    handle->id = s_posix_task_seq++;
    handle->done = 0;

    syn_posix_task_wrapper_t* wrapper = (syn_posix_task_wrapper_t*)malloc(sizeof(syn_posix_task_wrapper_t));
    if (!wrapper) {
        free(handle);
        return NULL;
    }
    wrapper->fn = fn;
    wrapper->arg = arg;
    wrapper->handle = handle;

    pthread_t th;
    if (pthread_create(&th, NULL, _syn_posix_thread_proc, wrapper) != 0) {
        free(wrapper);
        free(handle);
        return NULL;
    }
    handle->thread_id = (unsigned long)th;
    return handle;
}

void syn_task_wait(syn_task_handle_t* handle) {
    if (!handle) return;
    if (handle->thread_id) {
        pthread_join((pthread_t)handle->thread_id, NULL);
        handle->thread_id = 0;
    }
    handle->done = 1;
}

void syn_task_yield(void) {
    sched_yield();
}

void syn_task_handle_free(syn_task_handle_t* handle) {
    if (!handle) return;
    free(handle);
}

#endif /* _WIN32 / POSIX */
#endif /* __EMSCRIPTEN__ */
