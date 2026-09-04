# Synapse Quick Reference & Cheat Sheet (Python vs. Synapse)

Yapay zekanın ve geliştiricinin anında bakıp kod yazabileceği hızlı başvuru kartı.

---

## ⚡ Hızlı Sözlük (Python $\to$ Synapse)

| Özellik | Standart Python (PyTorch/NumPy) | Synapse AI Language |
|---|---|---|
| **Değişken** | `x = 10` | `let x = 10` veya `const PI = 3.14` |
| **Fonksiyon** | `def topla(a, b): return a + b` | `fn topla(a, b): return a + b` |
| **Tensör Tanımı** | `import torch; t = torch.tensor([1, 2])` | `let t = tensor([1, 2])` |
| **Matris Çarpımı** | `import numpy as np; C = np.matmul(A, B)` | `let C = A @ B` |
| **Transpoz** | `A.T` | `A.T` |
| **Türev Alma** | `loss.backward(); g = w.grad` | `loss.backward(); let g = w.grad.item()` |
| **Boru Hattı (Pipe)** | `h = norm(relu(x))` *(İç içe parantez)* | `let h = x \|> relu \|> norm` |
| **Yapay Zeka Promptu** | `client.chat.completions.create(...)` | `prompt ozet(t): system: "..." user: t` |
| **Python Kütüphanesi** | `import math` | `import py.math as math` |

---

## 🛠️ En Sık Kullanılan Kod Şablonları

### 1. Tensör ve Matris Çarpımı
```python
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])
let C = A @ B
let At = A.T
print("Çarpım:", C)
```

### 2. Çok Katmanlı Sinir Ağı (MLP) Eğitimi (nn & optim)
```python
let model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])
let optimizer = Adam(model.parameters(), lr=0.05)
let criterion = MSELoss()

let epoch = 1
while epoch <= 20:
    let y_pred = model(X)
    let loss = criterion(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1

let tahmin = model(yeni_veri)
```

### 3. Tekil Parametre ile Gradyan İnişi

### 3. Fonksiyonel Boru Hattı (`|>`)
```python
fn double(x):
    return x * 2.0

let veri = tensor([-2.0, 0.0, 3.0])
let cikti = veri |> relu |> double
print("Sonuç:", cikti)
```

### 4. Yapay Zeka Prompt Çağrısı
```python
prompt kod_acikla(kod):
    system: "Verilen kodu 1 cümlede açıkla."
    user: kod
    temperature: 0.2

let aciklama = kod_acikla("let A = tensor([[1, 2]])")
print("Açıklama:", aciklama)
```
