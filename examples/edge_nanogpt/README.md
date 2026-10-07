# Edge NanoGPT: bir transformer bloğunu yerel C99 binary'sine derlemek

Bu örnek, Synapse ile yazılmış **tek bir pre-LN transformer bloğunu** (self-attention + MLP + LayerNorm + residual bağlantılar) C99'a derleyip yerel bir çalıştırılabilir dosya üretir.

> Ağırlıklar 2×2 boyutlu oyuncak değerlerdir (`seq_len=2`, `d_model=2`). Örnek, derleme hattını uçtan uca göstermek içindir; kullanılabilir bir dil modeli değildir.

## Dosyalar

```
examples/edge_nanogpt/
├── model.syn               # LayerNorm, attention, MLP ve forward pass
├── build.py                # model.syn'i NativeCompiler ile yerel binary'ye derler
├── tokenizer.py            # Karakter/byte seviyesinde tokenizer (Python)
├── weights.py              # SafeTensors ağırlık üretici/yükleyici (Python)
└── nanogpt_generated.c     # Üretilen C99 kodunun örneği
```

## Derleme ve çalıştırma

Bir C derleyicisi gerekir (gcc, clang, cl.exe veya zig cc).

```bash
python examples/edge_nanogpt/build.py
./examples/edge_nanogpt/nanogpt          # Windows: .\examples\edge_nanogpt\nanogpt.exe
```

Ya da derleyip çalıştırmayı tek adımda yapın:

```bash
synapse demo --preset nanogpt
```

## Boyut

Linux x86_64 üzerinde clang ile binary yaklaşık **48 KB**'tır ve yalnızca libc ile libm'e dinamik olarak bağlıdır. CI her push'ta binary'yi Linux ve macOS'ta derleyip çalıştırır ve ölçülen boyutu iş özetine yazar. Kendi makinenizde ölçmek için:

```bash
python benchmarks/run_all.py
```
