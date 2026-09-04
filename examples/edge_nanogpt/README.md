# Synapse Edge NanoGPT: Zero-Dependency C99 Standalone LLM Engine

Synapse AI'ın **"Katil Kullanım Alanı" (Killer Use-Case)**: Harici Python, PyTorch veya ONNX çalışma zamanına ihtiyaç duymadan, **saf Synapse** ile yazılmış ve tek bir **0.21 MB'lık C99 bağımsız çalıştırılabilir dosyasına (standalone binary)** derlenen bir Causal Transformer / NanoGPT çıkarım motoru.

---

## 🎯 Temel Yetenekler & Mimari

- **Zero-Dependency C99 Binary:** PyTorch (`~2.5 GB`), ONNX Runtime (`~150 MB`) veya Python (`~50 MB`) olmadan tek başına çalışan ultra kompakt binary (**0.21 MB**).
- **Causal Multi-Head Self-Attention:** $Q \times K^T \rightarrow \text{Softmax} \times V$ ölçekli iç çarpım dikkat mekanizması.
- **Pre-LN Transformer Blokları:** Sayısal kararlılık için katman öncesi Layer Normalization (`mean`, `variance`, $\sqrt{\sigma^2 + \epsilon}$, $\gamma$, $\beta$).
- **GELU/ReLU MLP Feed-Forward:** İki katmanlı doğrusal olmayan projeksiyon ağı.
- **Hugging Face SafeTensors Uyumluluğu:** 8-bayt hizalı ikili SafeTensors ağırlıklarını `mmap` ile sıfır kopyalı belleğe eşleme.
- **Deterministik Bellek:** Arena bellek yöneticisi ile sıfır sızıntı ve $O(1)$ sürede bellek geri kazanımı.

---

## 📂 Dizin Yapısı

```
examples/edge_nanogpt/
├── model.syn               # Transformer blokları, self-attention ve forward pass
├── tokenizer.py            # UTF-8 karakter/byte seviyesinde tokenizer
├── weights.py              # SafeTensors ikili ağırlık oluşturucu ve yükleyici
├── build.py                # Native C99 compiler ile nanogpt.exe derleyicisi
└── README.md               # Dokümantasyon
```

---

## 🚀 Derleme ve Çalıştırma

### 1. C99 İkilisine Derleme
```bash
python examples/edge_nanogpt/build.py
```
Çıktı: `examples/edge_nanogpt/nanogpt.exe` (Windows) veya `nanogpt` (Linux/macOS) — Boyut: **~0.21 MB**.

### 2. Bağımsız Çalıştırma
```bash
# Windows
.\examples\edge_nanogpt\nanogpt.exe

# Linux / macOS
./examples/edge_nanogpt/nanogpt
```
