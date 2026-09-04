#ifndef SYNAPSE_RUNTIME_H
#define SYNAPSE_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#define SYN_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
#define SYN_THREAD_LOCAL _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
#define SYN_THREAD_LOCAL __thread
#else
#define SYN_THREAD_LOCAL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syn_tensor {
    double* data;
    int* shape;
    int ndim;
    int size;
    int requires_grad;
    struct syn_tensor* grad;

    /* Autograd Hesaplama Grafı */
    struct syn_tensor* left;
    struct syn_tensor* right;
    char op; /* '+', '-', '*', '/', '@', 'R' (relu), 'S' (sum), 'P' (pow), 'T' (transpose) */
    double scalar_val;
    int visited;
} syn_tensor_t;

/* ========================================================================= */
/* Bellek Havuzu (Arena Allocator) - Zero Allocation Overhead               */
/* ========================================================================= */
typedef struct syn_arena {
    char* buffer;
    size_t capacity;
    size_t offset;
} syn_arena_t;

typedef struct syn_arena_scope {
    syn_arena_t* arena;
    size_t saved_offset;
} syn_arena_scope_t;

syn_arena_t* syn_arena_create(size_t capacity);
void* syn_arena_alloc(syn_arena_t* arena, size_t bytes);
void syn_arena_reset(syn_arena_t* arena);
void syn_arena_free(syn_arena_t* arena);
syn_arena_scope_t syn_arena_scope_enter(syn_arena_t* arena);
void syn_arena_scope_leave(syn_arena_scope_t scope);
syn_tensor_t* syn_tensor_create_arena(syn_arena_t* arena, const double* initial_data, const int* shape, int ndim, int requires_grad);
syn_tensor_t* syn_tensor_zeros_arena(syn_arena_t* arena, const int* shape, int ndim, int requires_grad);
void syn_arena_set_active(syn_arena_t* arena);
syn_arena_t* syn_arena_get_active(void);

/* Tensör Oluşturma & Bellek */
syn_tensor_t* syn_tensor_create(const double* initial_data, const int* shape, int ndim, int requires_grad);
syn_tensor_t* syn_tensor_scalar(double val, int requires_grad);
syn_tensor_t* syn_tensor_zeros(const int* shape, int ndim, int requires_grad);
syn_tensor_t* syn_tensor_ones(const int* shape, int ndim, int requires_grad);
void syn_tensor_free(syn_tensor_t* t);

/* Aritmetik & Matris İşlemleri */
syn_tensor_t* syn_add(syn_tensor_t* a, syn_tensor_t* b);
syn_tensor_t* syn_sub(syn_tensor_t* a, syn_tensor_t* b);
syn_tensor_t* syn_mul(syn_tensor_t* a, syn_tensor_t* b);
syn_tensor_t* syn_div(syn_tensor_t* a, syn_tensor_t* b);
syn_tensor_t* syn_add_scalar(syn_tensor_t* a, double scalar);
syn_tensor_t* syn_sub_scalar(syn_tensor_t* a, double scalar);
syn_tensor_t* syn_mul_scalar(syn_tensor_t* a, double scalar);
syn_tensor_t* syn_pow_scalar(syn_tensor_t* a, double power);
syn_tensor_t* syn_matmul(syn_tensor_t* a, syn_tensor_t* b);
syn_tensor_t* syn_transpose(syn_tensor_t* a);

/* Aktivasyonlar & Redüksiyonlar */
syn_tensor_t* syn_relu(syn_tensor_t* a);
syn_tensor_t* syn_sigmoid(syn_tensor_t* a);
syn_tensor_t* syn_tanh(syn_tensor_t* a);
syn_tensor_t* syn_gelu(syn_tensor_t* a);
syn_tensor_t* syn_softmax(syn_tensor_t* a);
syn_tensor_t* syn_sum(syn_tensor_t* a);
syn_tensor_t* syn_mean(syn_tensor_t* a);

/* Autograd */
void syn_backward(syn_tensor_t* root);
void syn_tensor_zero_grad(syn_tensor_t* t);
void syn_graph_free(syn_tensor_t* root);

/* Yardımcılar & I/O */
double syn_tensor_item(syn_tensor_t* t);
void syn_tensor_print(const char* label, syn_tensor_t* t);
syn_tensor_t* syn_tensor_clone(syn_tensor_t* t);
syn_tensor_t* syn_tensor_escape(syn_tensor_t* t, syn_arena_t* arena);
syn_tensor_t* syn_tensor_escape_active(syn_tensor_t* t);
void syn_print_double(const char* label, double val);
void syn_print_int(const char* label, int val);
void syn_print_str(const char* val);

/* HPC & OpenMP Donanım Hızlandırma */
int syn_openmp_info(void);

/* ========================================================================= */
/* Eşzamanlılık, Görevler & CSP Kanalları (Concurrency & CSP Channels)       */
/* ========================================================================= */

typedef void (*syn_task_fn)(void* arg);

typedef struct syn_task {
    syn_task_fn fn;
    void* arg;
    struct syn_task* next;
} syn_task_t;

typedef struct syn_channel {
    void** buffer;
    int capacity;
    int head;
    int tail;
    int count;
    int closed;
#if !defined(__EMSCRIPTEN__)
    void* mutex;
    void* cond_send;
    void* cond_recv;
#else
    int is_wasm;
#endif
} syn_channel_t;

typedef struct syn_task_handle {
    int id;
    int done;
    void* result;
#if !defined(__EMSCRIPTEN__)
    #if defined(_WIN32)
    void* thread_handle;
    #else
    unsigned long thread_id;
    #endif
#endif
} syn_task_handle_t;

/* CSP Kanal API */
syn_channel_t* syn_channel_create(int capacity);
int syn_channel_send(syn_channel_t* ch, void* data);
void* syn_channel_recv(syn_channel_t* ch);
int syn_channel_try_recv(syn_channel_t* ch, void** out_data);
void syn_channel_close(syn_channel_t* ch);
void syn_channel_free(syn_channel_t* ch);
int syn_channel_is_closed(syn_channel_t* ch);
int syn_channel_count(syn_channel_t* ch);

/* Task / Spawn API */
syn_task_handle_t* syn_spawn(syn_task_fn fn, void* arg);
void syn_task_wait(syn_task_handle_t* handle);
void syn_task_yield(void);
void syn_task_handle_free(syn_task_handle_t* handle);

#if defined(__EMSCRIPTEN__)
/* WASM Tek İş Parçacıklı Cooperative Mikro-Görev Kuyruğu (Microtask Event Loop) */
void syn_wasm_enqueue_task(syn_task_fn fn, void* arg);
int syn_wasm_run_microtasks(int max_tasks);
int syn_wasm_pending_tasks(void);
void syn_wasm_clear_tasks(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SYNAPSE_RUNTIME_H */
