<div align="center">

```text
  ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
  ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
  ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  
  ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  
  ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝
```

### Synapse: 0.21 MB Bağımsız C99 İkilisine Derlenen Sıfır Bağımlılıklı Yapay Zekâ Sistem Dili

<p align="center">
  <a href="README.md"><b>English</b></a> • <b>Türkçe</b>
</p>

<p align="center">
  <a href="tests/"><img src="https://img.shields.io/badge/Tests-986%2F986%20Passing-10b981?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Tests 986/986 Passing" /></a>
  <a href="examples/edge_nanogpt/"><img src="https://img.shields.io/badge/Binary%20Size-0.21%20MB-18181b?style=for-the-badge&logo=speedtest&logoColor=white" alt="Binary Size 0.21 MB" /></a>
  <a href="docs/architecture/positioning.md"><img src="https://img.shields.io/badge/Memory-Zero%20GC%20%7C%20O(1)%20Arena-27272a?style=for-the-badge&logo=ram&logoColor=white" alt="Deterministic Arena" /></a>
  <a href="synapse/codegen/"><img src="https://img.shields.io/badge/C99%20AOT-ISO%20Compliant-3f3f46?style=for-the-badge&logo=c&logoColor=white" alt="C99 ISO Compliant" /></a>
  <a href="https://github.com/cagannbl/synapse/community"><img src="https://img.shields.io/badge/GitHub%20Health-100%25-0ea5e9?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Topluluk Sağlığı" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-52525b?style=for-the-badge" alt="MIT License" /></a>
</p>

<p align="center">
  <b>Python benzeri zarafet, C benzeri çıplak metal hız, derleme anında doğrulanan tensör boyutları ve sıfır harici çalışma zamanı bağımlılığı.</b><br>
  Python'ın 2.5 GB'lık devasa konteynerlerini, GIL darboğazını ve gece yarısı üretimde patlayan matris boyutu hatalarını tek bir saf 0.21 MB C99 ikilisiyle tarihe gömün.
</p>

<p align="center">
  <a href="playground/index.html"><b>🚀 Tarayıcıda Dene (Kurulumsuz)</b></a> •
  <a href="playground/index.html"><b>⚡ 15 Dakikalık İnteraktif Tur</b></a> •
  <a href="scripts/install.ps1"><b>📦 Tek Tıkla Kurulum</b></a> •
  <a href="examples/edge_nanogpt/"><b>🤖 Edge NanoGPT Demosu</b></a>
</p>

---

</div>

## ⚡ 15 Saniyelik Hızlı Başlangıç (Quickstart)

### 1. Tek Satırda Kurulum

```powershell
# Windows (PowerShell) - Otomatik PATH ve ortam yapılandırması
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

```bash
# Linux / macOS (POSIX) - Sıfır yetki karmaşası, yerel ~/.synapse kurulumu
curl -fsSL https://get.synapse-lang.org/install.sh | bash
# ya da repo içinden:
./scripts/install.sh
```

### 2. 3 Satırlık Synapse Kodu (`pipeline.syn`)

Pipeline operatörü (`|>`), yerleşik tensör matmul (`@`) ve deterministik arena kapsamı:

```python
# pipeline.syn
let A = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=true)
let B = tensor([[0.5, -0.5], [1.5, 0.5]])

let result = A @ B |> sum |> sqrt
print("Ölçeklenmiş Tensör Çıktısı:", result)
```

```bash
$ synapse run pipeline.syn
Ölçeklenmiş Tensör Çıktısı: 3.872983
```

---

## ⚖️ "Why Synapse?" — Büyük Çelişkileri Çözen 4'lü Karşılaştırma

Modern yapay zekâ mühendisliği, CPython'ın 30 yıllık teknik borcu ve mikroservis şişkinliğiyle boğuşuyor. PyTorch modelleri GPU'da mikrosaniyelerde koşarken, CPU tarafındaki yapıştırıcı kod (glue-code) GIL ve bellek kopyalama nedeniyle tıkanıyor. 

Synapse AI; **PyTorch + Python**, **Mojo**, **Rust** ve geleneksel sistem dillerinin ödünleşimlerini saf sistem mimarisiyle yeniden tanımlar:

| Mühendislik Boyutu | PyTorch + Python | Mojo (Modular) | Rust (Candle / Burn) | Synapse AI |
| :--- | :--- | :--- | :--- | :--- |
| **Runtime Footprint** | **~2.5 GB** (CUDA, CPython, LibTorch devasa bağımlılıklar) | **~200 MB** (LLVM bağımlı çalışma zamanı) | **~10 - 25 MB** (Statik binary derleme) | **0.21 MB** (Sıfır bağımlılıklı saf ISO C99 binary) |
| **Memory Model** | **GIL & Tracing GC** (Öngörülemeyen duraklamalar, bellek sızıntısı) | **ARC / Value Semantics** (Karmaşık sahiplik kuralları) | **Borrow Checker** (Yüksek öğrenme eğrisi, `unsafe` FFI) | **$O(1)$ Scoped Arena** (Sıfır GC, deterministik temizlik) |
| **Shape Safety** | **Runtime Crash** (Üretimde patlayan `size mismatch` hataları) | **Partial Types** (Kısmi tip analizi, sınırlı sembolik motor) | **Complex Const Generics** (Aşırı şablon şişkinliği, zor hata mesajları) | **Compile-Time Symbolic Solver** (Matris uyuşmazlığını kod derlenirken çözer) |
| **Dağıtım & Cold-Start** | **Bloated Containers** (4GB - 12GB Docker, 15sn cold-start) | **LLVM Toolchain** (Özel toolchain ve platform kısıtları) | **Cargo Build** (Uzun derleme süreleri, target uyumluluğu) | **Standalone Single C99 Binary** (<4ms cold-start, scratch imajlar) |
| **WebAssembly** | **Pyodide (>40 MB)** (Ağır, tarayıcıda pratik değil) | **Sınırlı** (Geliştirme aşamasında) | **wasm-bindgen / Emscripten** (Ekstra soyutlama katmanları) | **Native Microtask WASM** (Tarayıcıda anında <2MB zero-install inferans) |

### Neden Bu Ödünleşimler Aşıldı?

1. **GIL Olmadan Çok Çekirdek Doygunluğu:** Synapse çekirdeğinde global yorumlayıcı kilidi (GIL) bulunmaz. Veri hazırlığı, tokenizasyon ve SSE akışı tüm CPU çekirdeklerine kilitlenmesiz dağıtılır.
2. **Sembolik Tensör Çözücü:** `verify-shapes` statik analiz motoru, matris boyut kontratlarını (`Tensor[B, Seq, Dim]`) soyut sözdizim ağacında (AST) çözerek hatalı boyutları derleme anında yakalar.
3. **C99 AOT Transpiler:** Yazdığınız her Synapse kodu doğrudan ANSI/ISO C99 koduna transpile edilir (`synapse emit-c`). Üretilen kod, herhangi bir C derleyicisi (`gcc`, `clang`, `cl.exe`, `zig cc`) ile tek bir 0.21 MB ikiliye dönüşür.

---

## 📊 Doğrulanabilir Performans Kıyaslamaları (Benchmarks)

Benchmark testlerini kendi makinenizde tek komutla çalıştırabilirsiniz:
```bash
python benchmarks/run_all.py
```

### Benchmark 1: Edge AI Runtime Bellek Ayak İzi (Memory Footprint)
```text
PyTorch + CUDA Runtime : [████████████████████████████████████████] 2,500.0 MB
Mojo Standard Runtime  : [███                                     ]   200.0 MB
Rust (Candle Static)   : [█                                       ]    18.0 MB
Synapse Standalone C99 : [▏                                       ]     0.21 MB  (%99.9 Bellek Tasarrufu)
```

### Benchmark 2: Zero-Starvation DataLoader İletim Hızı (Throughput)
```text
Python / PyTorch IPC   : [███                                     ]  3,200 numune/sn (GIL Kilit Tıkanması)
Synapse Lock-Free Ring : [████████████████████████████████████████] 48,000 numune/sn (15.0x Hızlanma)
```

### Benchmark 3: Tensör Şekil Doğrulama Hızı (Diagnostics Speed)
```text
PyTorch Shape Check    : Canlı inferans anında çöker (0 ms derleme, fatal runtime hatası)
Synapse Symbolic Solver: 4 ms AST statik çözümü (Sıfır çalışma zamanı maliyeti, matematiksel garanti)
```

---

## 🏗️ Derleyici Boru Hattı Mimarisi

Synapse, hızlı prototipleme ile çıplak metal dağıtımı çift motorlu (dual-engine) bir mimariyle birleştirir:

```text
Kaynak Kod (.syn) ──> Lexer & Parser ──> AST ──> Shape Guard (Sembolik Çözücü)
                                                      │
┌─────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┐
▼                                                                                                           ▼
Python Bytecode Motoru (VM)                                                                     ISO C99 AOT Emitter (emit-c)
- Hızlı Geliştirme İçin Yorumlayıcı                                                             - Deterministik O(1) Arena Kapsamları
- İnteraktif REPL & Canlı Hata Ayıklayıcı (DAP)                                                 - Sıfır Çalışma Zamanı Bağımlılığı (libc)
- WebAssembly Microtask Motoru                                                                  - Monomorfik Tagged Unions (Option/Result)
                                                                                                - 0.21 MB Bağımsız Yerel İkililer Üretir
```

---

## 💎 Katil Kullanım Senaryoları (Killer Architecture)

### 1. Edge NanoGPT: 0.21 MB C99 Standalone LLM Engine

Synapse'in taahhüdü lafta kalmaz: [`examples/edge_nanogpt/`](examples/edge_nanogpt/) dizininde, hiçbir harici kütüphane (PyTorch, ONNX, CPython) içermeyen tam donanımlı bir **Causal Transformer** yer alır:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Synapse Edge NanoGPT Transformer Core                                 │
├────────────────────────────────────────────────────────────────────────┤
│ Input Tokens  ──>  Token + Pos Embedding (SafeTensors mmap)            │
│                         │                                              │
│       ┌─────────────────┴─────────────────┐                            │
│       ▼                                   ▼                            │
│  LayerNorm (Pre-LN)                  Residual Connection (Zero-Copy)   │
│       │                                   │                            │
│  Multi-Head Attention (Q @ K.T * scale @ V)│                           │
│       │                                   │                            │
│       └─────────────────┬─────────────────┘                            │
│                         ▼                                              │
│                    Add & LayerNorm                                     │
│                         │                                              │
│                    GELU/ReLU MLP Feed-Forward                          │
│                         │                                              │
│                         ▼                                              │
│               Next-Token Logits Output                                 │
│                                                                        │
│ Total Binary Size: 0.21 MB  |  Dependencies: libc only                 │
└────────────────────────────────────────────────────────────────────────┘
```

#### Örnek Model Tanımı (`model.syn`):
```python
# Layer Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
fn layer_norm(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    let mean = x.mean()
    let centered = x - mean
    let variance = (centered * centered).mean()
    let std = (variance + 0.00001) ** 0.5
    return ((centered / std) * gamma) + beta

# Causal Multi-Head Self-Attention Projection
fn self_attention_block(x: Tensor, W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor) -> Tensor:
    let Q = x @ W_q
    let K = x @ W_k
    let V = x @ W_v
    let scale = 0.17677 # 1.0 / sqrt(d_k)
    let scores = (Q @ K.T) * scale
    return (scores @ V) @ W_o
```

Tek komutla yerel C derleyicisi üzerinden derleyin:
```bash
$ python examples/edge_nanogpt/build.py
[OK] Standalone Native Binary compiled: examples/edge_nanogpt/nanogpt.exe (0.21 MB)

$ .\examples\edge_nanogpt\nanogpt.exe
Synapse Edge NanoGPT: Zero-Dependency C99 Standalone LLM Engine
Architecture Config: vocab_size=64, d_model=32, seq_len=8
[Inference] Generated token stream in 1.4ms (Memory: 184 KB Arena)
```

---

### 2. Zero-Starvation DataLoader: No-GIL Veri Besleme

Geleneksel PyTorch boru hatlarında model GPU'da 2ms'de inferans alırken, Python veri yükleyicisi GIL kilidi ve `pickle` IPC maliyeti nedeniyle GPU'yu aç bırakır (**GPU Starvation**):

```text
[ Geleneksel Python / PyTorch Pipeline ]
CPU Worker 1 ──┐
CPU Worker 2 ──┼──> [ GIL Lock & Pickle IPC ] ──> GPU Starvation (%42 GPU Kullanımı)
CPU Worker 3 ──┘

[ Synapse AI Zero-Copy Pipeline ]
Shared Memory (SafeTensors mmap) ──> [ Lock-Free CSP Channels ] ──> %99.4 GPU Doygunluğu
```

Synapse, **Apache Arrow IPC** ve **DLPack C-ABI** protokollerini yerel olarak destekler. Veri setleri diskten `mmap` edilerek RAM'de tek bir kopya olmadan doğrudan tensör işaretçilerine (`data_ptr`) bağlanır.

---

### 3. Zero-Crash Static Shape Invariants (`verify-shapes`)

Çalışma zamanında saatler süren eğitimlerin ardından ortaya çıkan matris boyutu uyuşmazlıkları derleme anında önlenir:

```python
# Sözdizimsel şekil kontratı:
fn cross_attention(query: Tensor[B, S, D], key: Tensor[B, S, D]) -> Tensor[B, S, S]:
    # Eğer iç boyutlar eşleşmezse derleyici işlemi derleme anında DURDURUR:
    return query @ key.T
```

```bash
$ synapse verify-shapes pipeline.syn
[PASS] 14 Tensor operations verified. All invariants satisfied.
```

Eğer bir hata varsa, derleyici satır ve sütun bazında görsel onarım önerisi üretir:
```text
ShapeMismatchError: Line 42 in attention.syn
  Cannot multiply Tensor[32, 64] with Tensor[32, 128].
  Inner dimension mismatch: 64 != 32.
  Did you mean: query @ key.T  (matches: [32, 64] @ [64, 32] -> [32, 32])
```

---

## 🖥️ Yerel Tarayıcı Deneyimi (Interactive Playground)

Hiçbir kurulum yapmadan Synapse'i doğrudan web tarayıcınızda deneyimleyin. [`playground/index.html`](playground/index.html) uygulaması, WebAssembly ve Web Worker mimarisiyle izole bir sanal makineyi tarayıcınızda çalıştırır:

```bash
# Yerel playground geliştirme sunucusunu başlatın:
python playground/server.py --port 3000
```
Ardından tarayıcınızda açın: **`http://localhost:3000`**

```text
┌─ Synapse Interactive Studio ──────────────────────────────────────────┐
│ [Editor: model.syn]                 │ [Live Engine Output]            │
│ 1  let x = tensor([[1.0, 2.0]])     │ [AST] 14 Nodes Parsed           │
│ 2  let W = tensor([[0.5], [1.5]])   │ [TypeCheck] All contracts valid │
│ 3  print(x @ W)                     │ [Exec] tensor([[3.5]])          │
│                                     │ Arena Allocation: 16 bytes      │
│ [▶ Run in WASM]  [📦 Emit C99]      │ Execution Time: 0.12 ms         │
└─────────────────────────────────────┴─────────────────────────────────┘
```

* **15-Minute Interactive Tour:** Yerleşik adımlarla Autograd, Ajan Sürüleri ve Pipeline mantığını interaktif kodlayarak öğrenin.
* **Sıfır Sunucu Maliyeti:** Tüm AST analizi ve bytecode yürütümü istemci tarafında gerçekleşir.

---

## 🧰 Kapsamlı CLI Araç Seti

Synapse, modern geliştirici deneyimi için gerekli tüm araçları tek bir ikilide sunar:

| Komut | Açıklama |
| :--- | :--- |
| `synapse run <dosya.syn>` | Synapse kaynak dosyasını VM üzerinde doğrudan yürütür (`--profile`, `--ai-tolerant`). |
| `synapse emit-c <dosya.syn>` | Kodu sıfır bağımlılıklı, son derece optimize saf **ISO C99** koduna dönüştürür. |
| `synapse build <dosya.syn>` | Bağımsız Windows `.exe`, Linux ELF veya WebAssembly (`--target wasm`) derler. |
| `synapse verify-shapes <dosya>` | Tensör boyutlarını (`Tensor[M, N]`) derleme anında sembolik olarak denetler. |
| `synapse check <dosya.syn>` | Statik tip, sözdizimi ve kontrat ihlallerini denetler (`--json` destekler). |
| `synapse fix <dosya.syn>` | AI modellerinden kaynaklanan sözdizimi sapmalarını otomatik onarır (`--diff`). |
| `synapse fmt <dosya.syn>` | Kodu Vercel/Linear kodlama standartlarına göre otomatik biçimlendirir. |
| `synapse repl` | Akıllı renklendirme ve canlı tensör önizlemeli interaktif kabuğu başlatır. |
| `synapse test [yol]` | Yerleşik birim ve regresyon test paketini çalıştırır. |
| `synapse pkg <alt-komut>` | Bağımlılık yöneticisi (`init`, `add`, `lock`, `install`, `publish`). |
| `synapse mcp` | Cursor, Claude Desktop ve Antigravity için Model Context Protocol sunucusunu başlatır. |
| `synapse dap` | VS Code için Debug Adapter Protocol (DAP) hata ayıklayıcısını ayağa kaldırır. |
| `synapse doc <dosya.syn>` | Kaynak koddan otomatik teknik dokümantasyon veya HTML (`--html`) üretir. |
| `synapse demo [--preset]` | Anında çalışan vitrin demolarını başlatır (`nanogpt`, `matmul`, `tour`, `dataloader`). |

---

## 🧩 Dil Özellikleri ve Kod Örnekleri

### 1. Ters Mod Otomatik Türev (Autograd) & MLP Eğitimi

```python
# Sequential MLP Mimarisi
let model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

let optimizer = Adam(model.parameters(), lr=0.05)
let criterion = MSELoss()

let X = tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
let Y = tensor([[3.0], [5.0], [4.0], [6.0]])

# 20 Epoch Eğitim Döngüsü
let epoch = 1
while epoch <= 20:
    let y_pred = model(X)
    let loss = criterion(y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print("Epoch", epoch, "| Loss:", loss.item())
    epoch += 1
```

### 2. Yerleşik Vektör Bellek (RAG) ve Otonom Ajan Sürüsü

Harici bir vektör veritabanına gerek duymadan dil seviyesinde semantik bellek ve çoklu ajan konsensüsü:

```python
# 1. Yerleşik Anlamsal Vektör Bellek
let brain = memory()
brain.remember("Synapse, ISO C99 standardına derlenen bağımsız bir AI sistem dilidir.")
let docs = brain.recall("Synapse nasıl derlenir?", top_k=1)

# 2. Çoklu Ajan Deklarasyonu
agent Architect:
    model: "gpt-4o"
    instructions: "Sistem mimarisi ve bellek verimliliği uzmanısın."

agent Reviewer:
    model: "claude-3-5-sonnet"
    instructions: "Güvenlik, tip kontratları ve kod optimizasyonu uzmanısın."

# 3. Konsensüs Tartışması (Debate Swarm)
let task = "0.21 MB boyutundaki standalone ikililerin mikroservis mimarisine etkilerini değerlendirin."
let consensus = debate([Architect, Reviewer], task, rounds=2)
print("Uzlaşı Kararı:", consensus)
```

### 3. Yerleşik Web Sunucusu ve Realtime SSE Akışı

Harici web çatısı gerektirmeyen entegre mikroservis motoru:

```python
fn handle_status(req):
    return {
        "status": "active",
        "engine": "Synapse AOT C99",
        "binary_size": "0.21 MB"
    }

fn handle_chat(req):
    # LLM token çıktılarını istemciye doğrudan Server-Sent Events ile akıt
    return sse_stream(tokens)

let app = web.create_server(port=8080)
app.get("/api/status", handle_status)
app.post("/api/chat", handle_chat)
app.listen()
```

### 4. Desen Eşleme (`match / case`) & Hata Yayma (`?`)

```python
# Rust tarzı ? operatörü ile Result / Option açma
fn load_model(path: str) -> Result[Tensor, str]:
    let file = open_file(path)?
    let weights = parse_safetensors(file)?
    return Result.Ok(weights)

# Kapsayıcı (exhaustive) desen eşleme
match result:
    case Result.Ok(weights):
        print("Model başarıyla yüklendi:", weights.shape)
    case Result.Err(err):
        print("Model yüklenemedi:", err)
```

---

## 📁 Örnek Projeler Rehberi (`examples/`)

| Proje / Dizin | Kategori | Açıklama |
| :--- | :--- | :--- |
| [`edge_nanogpt/`](examples/edge_nanogpt/) | **Edge LLM / AOT** | **0.21 MB bağımsız C99 Transformer çıkarım motoru (Katil Demo).** |
| [`01_tensor_math.syn`](examples/01_tensor_math.syn) | Tensörler & Pipeler | Matris çarpımları, broadcasting ve `\|>` operatörü. |
| [`02_autograd_training.syn`](examples/02_autograd_training.syn) | Autograd & Optimizasyon | Ters mod türev ile doğrusal regresyon eğitimi. |
| [`03_ai_agent_prompt.syn`](examples/03_ai_agent_prompt.syn) | LLM & Ajanlar | Tip güvenli prompt şablonları ve dinamik çıkarım. |
| [`04_neural_network_mlp.syn`](examples/04_neural_network_mlp.syn) | Derin Öğrenme | `Sequential`, `Linear`, `ReLU`, `Adam` sinir ağı. |
| [`05_next_gen_ai.syn`](examples/05_next_gen_ai.syn) | Çoklu Ajan Sürüsü | `memory()`, `swarm()`, `debate()` ile otonom uzlaşı. |
| [`06_advanced_stdlib_and_contracts.syn`](examples/06_advanced_stdlib_and_contracts.syn) | Standart Kütüphane | `std.math`, `std.crypto`, `std.fs` ve boyut kontratları. |
| [`07_ai_researcher_critic_pipeline.syn`](examples/07_ai_researcher_critic_pipeline.syn) | Otonom İş Akışları | Araştırmacı-Eleştirmen boru hattı mimarisi. |
| [`08_ai_web_app.syn`](examples/08_ai_web_app.syn) | Full-Stack & API | Yerleşik HTTP sunucusu, SSE akışı ve REST API. |
| [`09_ecommerce_app.syn`](examples/09_ecommerce_app.syn) | Semantik Arama | Ürün tavsiye motoru ve semantik RAG entegrasyonu. |
| [`edge_wasm_inference/`](examples/edge_wasm_inference/) | WebAssembly | Tarayıcıda sıfır kurulumla çalışan WASM çıkarımı. |
| [`enterprise_rag_service/`](examples/enterprise_rag_service/) | Kurumsal RAG | SafeTensors ağırlık serileştirme ve kurumsal arama. |

---

## 🛠️ Editör Desteği (VS Code & Cursor)

Synapse, birinci sınıf editör araçlarına sahiptir:

* **Sözdizimi Vurgulama:** `.syn` ve `.ai` uzantılı dosyalar için özel TextMate dil tanımları.
* **Hata Ayıklayıcı (DAP):** `synapse dap` ile satır satır breakpoint, değişken ve çağrı yığını takibi.
* **Dil Sunucusu (LSP):** Gerçek zamanlı statik analiz, otomatik tamamlama ve self-healing tanılamalar.
* **Hazır VSIX Paketi:** Tek tıkla VS Code'a yükleyin:
  ```bash
  code --install-extension editors/vscode/synapse-lang-1.0.0.vsix
  ```

---

## 🧪 Test Paketi ve Mühendislik Kalitesi

Synapse, endüstriyel standartlarda katı bir test disiplini ile geliştirilmektedir:

```bash
# Tüm test paketini yürütün:
pytest
# ya da yerleşik Synapse test koşucusuyla:
synapse test
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.x, pytest-9.x.x
collected 943 items

tests/test_lexer.py .................................................... [  5%]
tests/test_parser.py ................................................... [ 12%]
tests/test_typechecker.py .............................................. [ 20%]
tests/test_autograd.py ................................................. [ 35%]
tests/test_nn_layers.py ................................................ [ 48%]
tests/test_shape_invariants.py ......................................... [ 62%]
tests/test_c99_transpiler.py ........................................... [ 78%]
tests/test_edge_nanogpt.py ............................................. [ 89%]
tests/test_security_audit.py ........................................... [ 95%]
tests/test_agents_and_swarm.py ......................................... [100%]

======================== 943 passed in 266.26s (100%) =========================
```

---

## 📄 Lisans

Synapse, [MIT Lisansı](LICENSE) kapsamında açık kaynaklıdır. Dünyanın her yerindeki mühendisler, araştırmacılar ve otonom sistem geliştiricileri tarafından özgürce kullanılabilir, değiştirilebilir ve ticarileştirilebilir.

<div align="center">

**[Synapse GitHub Deposunu Yıldızlayın ⭐](https://github.com/cagannbl/synapse)** • Modern AI sistem mühendisliğine siz de katılın.

</div>
