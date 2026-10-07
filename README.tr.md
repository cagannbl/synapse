<div align="center">

# Synapse

**Tensör kodu için Python benzeri bir dil: şekil (shape) hatalarını program çalışmadan önce yakalar ve küçük, yerel C99 binary'lerine derlenir.**

<p>
  <a href="README.md"><b>English</b></a> • <b>Türkçe</b>
</p>

<p>
  <a href="https://github.com/cagannbl/synapse/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/cagannbl/synapse/tests.yml?branch=main&label=tests" alt="Testler" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT Lisansı" /></a>
</p>

</div>

```python
fn project(x: Tensor[N, D], w: Tensor[K, M]) -> Tensor[N, M]:
    return x @ w    # D ve K farklı boyutlar: kod çalışmadan reddedilir
```

```text
$ synapse verify-shapes project.syn
[Shape Guard] FAILED: Found 1 shape violation in 'project.syn':

CompileTimeShapeMismatchError: Cannot multiply tensor of shape ('N', 'D') with tensor of shape ('K', 'M'). Inner dimensions must match: D != K.
  --> Line 2, Column 14
```

> **Durum:** Synapse genç ve tek geliştiricili bir proje. Çekirdek dil, yorumlayıcı, shape checker ve C arka ucu CI'da Linux ve macOS üzerinde test ediliyor; diğer kısımlar deneysel (bkz. [Modül kararlılığı](#modül-kararlılığı)). Geriye dönük uyumsuz değişiklikler olabilir.

---

## Hızlı Başlangıç

```bash
git clone https://github.com/cagannbl/synapse.git
cd synapse
pip install -e .            # Python 3.10+, tek bağımlılık numpy
```

Ya da Python ortamınıza dokunmadan `~/.synapse` altına kurun:

```bash
curl -fsSL https://raw.githubusercontent.com/cagannbl/synapse/main/scripts/install.sh | bash
```

Boru hattı operatörü (`|>`) ve matris çarpımı (`@`) ile ilk program:

```python
# pipeline.syn
let A = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=true)
let B = tensor([[0.5, -0.5], [1.5, 0.5]])

let result = A @ B |> sum |> sqrt
print("Ölçeklenmiş Tensör Çıktısı:", result)
```

```bash
$ synapse run pipeline.syn
Ölçeklenmiş Tensör Çıktısı: tensor(3.4641016151377544, requires_grad=True)
```

Tarayıcıda denemek isterseniz: `python playground/server.py` komutu `http://127.0.0.1:3000` adresinde, kodunuzu gerçek yorumlayıcıyla çalıştıran yerel bir playground açar.

---

## Synapse bugün neler yapıyor

| | |
| :--- | :--- |
| **Statik şekil kontrolü** | `synapse verify-shapes`, matris çarpımı ve broadcasting işlemlerini imzalarda bildirilen şekillere (`Tensor[B, S, D]`) göre kontrol eder; sembolik boyutlar ve dönüş tipleri dahil. Aynı kontrol `synapse run` öncesinde de çalışır. |
| **İki motor** | Geliştirme için bytecode VM (Python ile yazılmış) ve yerel binary'ler için C99 arka ucu (`synapse emit-c`, `synapse build`). C arka ucunun desteklemediği kod VM'de çalışır. |
| **Tensörler ve autograd** | NumPy üzerinde N boyutlu tensörler, ters mod otomatik türev, `Linear` / `Sequential` / `Adam` / `MSELoss` vb. |
| **Dil özellikleri** | Koşullu (`if`) `match` / `case`, `?` operatörlü `Result` / `Option`, `let`, tipli fonksiyonlar, struct'lar, iş parçacıkları (`spawn`) ve kanallar. |
| **Araçlar** | REPL, formatlayıcı, linter, LSP, VS Code eklentisi, DAP hata ayıklayıcı, MCP sunucusu. |

### Yerel binary'ler

[`examples/edge_nanogpt/`](examples/edge_nanogpt/), Synapse ile yazılıp C99'a derlenen tek bir pre-LN transformer bloğudur (attention + MLP). 2×2 boyutlu oyuncak ağırlıklar kullanır: derleme hattını gösterir, kullanılabilir bir dil modeli değildir.

```bash
python examples/edge_nanogpt/build.py   # gcc, clang, cl.exe veya zig cc gerekir
./examples/edge_nanogpt/nanogpt
```

Linux x86_64 üzerinde clang ile binary **≈48 KB** boyutundadır ve yalnızca libc ile libm'e bağlıdır. CI her push'ta onu Linux ve macOS'ta derleyip çalıştırır ve ölçülen boyutu raporlar.

---

## Benchmark'lar

```bash
python benchmarks/run_all.py     # makine tarafından okunabilir çıktı için --json
```

Her sayı çalıştırdığınız anda ölçülür; hiçbir değer sabit yazılmamıştır. Linux x86_64, Python 3.13 üzerinde bir çalıştırma:

| Benchmark | Sonuç |
| :--- | :--- |
| Edge NanoGPT yerel binary | 47.6 KB (clang), çalışıyor |
| DataLoader: NumPy dilimleme (kopyasız, üst sınır) | ~990 M örnek/sn |
| DataLoader: `SynapseFastDataLoader` | ~6.1 M örnek/sn |
| DataLoader: `multiprocessing` + pickle kuyruğu | ~1.9 M örnek/sn |
| Uyumsuz bir matris çarpımının şekil kontrolü | 0.22 ms'de yakalandı, `A @ B.T` öneriliyor |

Synapse loader, süreç + pickle hattından yaklaşık 3 kat hızlı; ama düz NumPy dilimlemeden çok daha yavaş. Sonuçlar makineden makineye büyük farklılık gösterir.

---

## Dil turu

Aşağıdaki kod parçalarının hepsi test paketi tarafından çalıştırılır; bu yüzden dille her zaman uyumludur.

### Şekil sözleşmeleri

```python
# İmzada adı geçen boyutlar fonksiyon gövdesinde birbiriyle uyuşmalıdır.
fn attention_scores(q: Tensor[B, S, D], k: Tensor[B, S, D]) -> Tensor[B, S, S]:
    return q @ k.T
```

```bash
$ synapse verify-shapes attention.syn
[Shape Guard] PASS: All tensor shape invariants and dimension contracts verified in 'attention.syn'
```

### Autograd ve eğitim

```python
let model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

let optimizer = Adam(model.parameters(), lr=0.05)
let criterion = MSELoss()

let X = tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
let Y = tensor([[3.0], [5.0], [4.0], [6.0]])

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

### Desen eşleme ve `?`

```python
fn parse_dim(text: str) -> Result[int, str]:
    if text == "":
        return Err("boş boyut")
    return Ok(int(text))

fn parse_shape(rows: str, cols: str) -> Result[list, str]:
    let r = parse_dim(rows)?   # ayrıştırma başarısızsa Err'i hemen döndürür
    let c = parse_dim(cols)?
    return Ok([r, c])

for args in [["3", "4"], ["3", ""]]:
    match parse_shape(args[0], args[1]):
        case Ok(shape) if shape[0] == shape[1]:
            print("Kare şekil:", shape)
        case Ok(shape):
            print("Şekil:", shape)
        case Err(err):
            print("Geçersiz şekil:", err)
```

### İş parçacıkları ve kanallar

```python
let ch = channel(capacity=4)

fn producer(count):
    for i in range(count):
        ch.send(i * i)
    ch.close()

spawn(producer, 5)
for value in ch:       # kanal kapanıp boşalana kadar bekler
    print(value)
```

### Prompt'lar, ajanlar ve bellek (deneysel)

`OPENAI_API_KEY` veya `ANTHROPIC_API_KEY` tanımlı değilse yerleşik bir sahte (mock) model yanıt verir; bu yüzden bu örnekler çevrimdışı çalışır.

```python
prompt Summarize(text: str):
    system: "Teknik metinleri tek cümlede özetlersin."
    user: "Özetle: " + text

print(Summarize("Synapse tensör şekillerini çalışmadan önce kontrol eder ve C99'a derlenir."))

let brain = memory()
brain.remember("Synapse ISO C99'a derlenir.")
print(brain.recall("Synapse nasıl derlenir?", top_k=1))
```

### HTTP sunucusu (deneysel)

```python
fn handle_status(req):
    return {
        "status": "active",
        "engine": "Synapse"
    }

fn handle_chat(req):
    # Token'ları istemciye Server-Sent Events ile akıt
    let tokens = ["Hello", " from", " Synapse"]
    return sse_response(tokens)

serve(port=8080, routes={
    "GET /api/status": handle_status,
    "POST /api/chat": handle_chat
})
```

Daha fazla program [`examples/`](examples/) klasöründe; her biri test paketi tarafından çalıştırılır.

---

## Modül kararlılığı

| Durum | Modüller |
|---|---|
| **Çekirdek** | `synapse/lexer`, `synapse/parser`, `synapse/analyzer`, `synapse/vm`, `synapse/codegen` (C99 arka ucu), `synapse/runtime`, `synapse/core` (tensörler ve autograd), `synapse/nn`, `synapse/optim`, `synapse/cli.py`, `synapse/lsp` |
| **Deneysel** | `synapse/ai` (ajanlar, swarm, LLM sağlayıcıları), `synapse/web`, `synapse/orm`, `packages/`, `synapse/pkg`, `synapse/mcp_server.py`, WASM / CUDA birlikte çalışabilirliği |

---

## Nasıl çalışır

```text
source.syn ─▶ lexer ─▶ parser ─▶ AST ─▶ shape checker
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            bytecode derleyici + VM                    C99 üretici (emit-c)
            (Python; run/repl kullanır)           ─▶ gcc / clang / cl / zig cc
                                                   ─▶ yerel binary (libc + libm)
```

Derleyici, TorchInductor, Triton'ın ön ucu ve Cython gibi Python ile yazılmıştır. Derlenen binary'lerde Python yoktur: libc ve libm'e bağlanan düz C99 kodudur ve tensör belleği için arena allocator kullanır.

---

## CLI

| Komut | Ne yapar |
| :--- | :--- |
| `synapse run <dosya>` | Şekilleri kontrol eder, sonra çalıştırır (mümkünse C arka ucu, değilse VM; `--vm` VM'i zorlar). |
| `synapse verify-shapes <dosya>` | Yalnızca statik şekil kontrolü (`--json` desteklenir). |
| `synapse emit-c <dosya>` / `synapse build <dosya>` | C99 kaynak kodu üretir / yerel çalıştırılabilir dosya derler. |
| `synapse check`, `lint`, `fmt`, `fix` | Tip kontrolü, lint, formatlama, LLM'lerin sık yaptığı hatalar için otomatik düzeltme. |
| `synapse repl` | Etkileşimli REPL. |
| `synapse lsp`, `dap`, `mcp` | Dil sunucusu, hata ayıklama adaptörü, Model Context Protocol sunucusu. |
| `synapse demo` | Kısa vitrin demoları (`--preset nanogpt`, `matmul`, `tour`, `dataloader`). |

Tam liste için `synapse --help`.

---

## Geliştirme

```bash
pip install -e ".[test]"
pytest
```

Test paketi yaklaşık bir dakikada çalışır; dili, iki motoru, shape checker'ı, C çalışma zamanını ve bu README'deki, örneklerdeki ve playground'daki her kod parçasını kapsar. Bkz. [CONTRIBUTING.md](CONTRIBUTING.md).

Editör desteği: [`editors/vscode/`](editors/vscode/) altında TextMate grameri ve VS Code eklentisi:

```bash
python editors/vscode/build_vsix.py
code --install-extension editors/vscode/synapse-lang-1.0.0.vsix
```

## Lisans

[MIT](LICENSE)
