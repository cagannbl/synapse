# Synapse Language Syntax Guide (v1.0)
*An AI-Optimized, Python-like Programming Language*

## 1. Temel Sözdizimi & Değişkenler

Synapse, Python'un temiz ve okunabilir girintiye dayalı yapısını korurken modern tip güvenliği ve bildirim kolaylığı sunar:

```python
# Değişken bildirimi
let learning_rate = 0.001
let epochs = 100
let model_name: str = "SynapseTransformer"
const PI = 3.1415926535

# Listeler ve Sözlükler
let batch_sizes = [16, 32, 64, 128]
let config = {
    "lr": learning_rate,
    "optimizer": "adamw",
    "use_gpu": true
}
```

---

## 2. Birinci Sınıf Tensörler (First-Class Tensors)

Harici NumPy veya PyTorch yükleme zorunluluğu olmadan tensörler dilin çekirdeğindedir:

```python
# Tensör oluşturma
let A = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=true)
let B = tensor([[5.0, 6.0], [7.0, 8.0]])

# Matris Çarpımı (@ operatörü)
let C = A @ B

# Element-wise işlemler ve Transpoz
let D = (A + B) * 2.5
let At = A.T

print("Matrix Product shape:", C.shape)
print("Result C:", C)
```

---

## 3. Otomatik Türev (Autograd Engine)

Hesaplama grafı dil tarafından otomatik izlenir:

```python
fn loss_fn(w, x, y_true):
    let y_pred = x @ w
    let diff = y_pred - y_true
    return (diff * diff).sum()

let w = tensor([[0.5], [1.5]], requires_grad=true)
let x = tensor([[1.0, 2.0], [3.0, 4.0]])
let y = tensor([[3.5], [9.5]])

let loss = loss_fn(w, x, y)
loss.backward()

print("w gradyanı:", w.grad)
```

Ayrıca fonksiyonel türev alma fonksiyonu `grad(f)`:
```python
fn square(x):
    return x * x

let d_square = grad(square)
let slope = d_square(4.0) # 8.0
```

---

## 4. Fonksiyonel Pipeline Operatörü (`|>`)

Veri akışını ve yapay zeka tensör dönüşümlerini akıcı kılmak için:

```python
fn normalize(t):
    return t / t.max()

fn relu(t):
    return t.relu()

let data = tensor([-1.0, 2.0, -3.0, 4.0])
let output = data |> relu |> normalize
```

---

## 5. Yerleşik Yapay Zeka Prompt Blokları (`prompt`)

Büyük dil modellerini (LLM) harici istemci kütüphaneleriyle sarmalamak yerine doğrudan dil semantiğiyle tanımlayın:

```python
prompt analyze_sentiment(review_text: str) -> str:
    system: "You are an accurate sentiment classifier. Output POSITIVE or NEGATIVE only."
    user: f"Analyze this text: {review_text}"
    temperature: 0.1

let feedback = "This new language is lightning fast and intuitive!"
let sentiment = analyze_sentiment(feedback)
print("Sentiment:", sentiment)
```

---

## 6. Otonom Ajan & Araç Primitifleri (`agent` & `tool`)

Ajanik sistemler doğrudan dilin yapıtaşlarıdır:

```python
tool search_database(query: str) -> str:
    # Veritabanı arama simülasyonu
    return f"Results for: {query}"

agent CodeAssistant:
    model: "gpt-4o"
    tools: [search_database]
    instructions: "You assist engineers with repository queries and architecture."

let response = CodeAssistant.run("Find memory leaks in auth service")
```

---

## 7. Python Birlikte Çalışabilirlik (Python Interop)

Python ekosistemindeki tüm kütüphaneler sıfır sürtünmeyle kullanılabilir:

```python
import py.math as math
import py.time as time

let angle = math.pi / 4.0
print("Cosine:", math.cos(angle))
```
