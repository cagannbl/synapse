# Construction Plan: Synapse (AI-Optimized Programming Language)

## Objective
Python benzeri ergonomik sözdizimine sahip, tensörleri, otomatik türevi (autograd), tensör hesaplama graflarını ve LLM/Ajan semantiklerini birinci sınıf yerleşik vatandaş olarak destekleyen yapay zeka optimize programlama dili ("Synapse") geliştirmek.

---

## Step 1: Formal Grammar & Language Specification
**Context:** Dilin sözdizimini, anahtar sözcüklerini, tensör semantiğini ve prompt bloklarını formel olarak tanımlamak.
- **Tasks:**
  - `specs/grammar.ebnf` dosyasında EBNF gramerini yaz (indentation kuralları, ifadeler, bildirimler, tensör operatörleri).
  - `specs/syntax_guide.md` ile örnek kodları ve dil rehberini hazırla (değişkenler, fonksiyonlar, matris işlemleri, `grad` fonksiyonu, `prompt` blokları).
- **Verification Commands:**
  - `python -c "print('Grammar docs validated')"`
- **Exit Criteria:** EBNF grameri ve sentaks rehberi eksiksiz, tutarlı ve örneklerle belgelenmiş olmalı.

---

## Step 2: Lexer & Indentation Tokenizer
**Context:** Python benzeri girintiye duyarlı blokları (`INDENT`, `DEDENT`, `NEWLINE`) ve özel AI operatörlerini (`@`, `|>`, `~>`) tokenize eden sözcüksel analizci.
- **Tasks:**
  - `synapse/lexer/token.py`: Token tiplerini (Keywords, Operators, Literals, Indents) tanımla.
  - `synapse/lexer/lexer.py`: Regex ve indentation stack tabanlı tokenizer motorunu yaz.
  - `tests/test_lexer.py`: Girintiler, çok satırlı ifadeler ve tensör sembolleri için kapsamlı testler ekle.
- **Verification Commands:**
  - `pytest tests/test_lexer.py`
- **Exit Criteria:** Tüm token türleri ve karmaşık Pythonik girinti senaryoları %100 doğrulukla testten geçmeli.

---

## Step 3: AST Nodes & Recursive Descent Parser
**Context:** Token akışını hiyerarşik ve tip güvenli Soyut Sözdizim Ağacına (AST) dönüştüren parser.
- **Tasks:**
  - `synapse/parser/ast_nodes.py`: `Program`, `FunctionDef`, `TensorLiteral`, `BinaryOp`, `PromptDef`, `CallExpr` vb. AST veri sınıflarını oluştur.
  - `synapse/parser/parser.py`: Operatör önceliği (Pratt parsing / Precedence climbing) ve recursive descent parser'ı yaz.
  - Hata yakalama: Satır, sütun ve net hata mesajları üreten sözdizim hata motoru.
  - `tests/test_parser.py`: Matematiksel ifadeler, fonksiyonlar ve tensör tanımları için AST testleri yaz.
- **Verification Commands:**
  - `pytest tests/test_parser.py`
- **Exit Criteria:** Geçerli Synapse kodları eksiksiz AST üretmeli, hatalı sentaks durumunda açıklayıcı hata fırlatmalı.

---

## Step 4: Core Tensor Engine & Autograd System
**Context:** Harici bağımlılık gerektirmeden çalışan, N-boyutlu dinamik tensör motoru ve reverse-mode otomatik türev (Computational Graph) altyapısı.
- **Tasks:**
  - `synapse/core/tensor.py`: Shape, strides, broadcast kuralları, matris çarpımı (`matmul`), toplama, çıkarma, scalar işlemler.
  - `synapse/core/autograd.py`: Computational Graph düğümleri, `requires_grad=True`, zincir kuralı (chain rule) ve `backward()` hesaplaması.
  - `tests/test_tensor.py` ve `tests/test_autograd.py`: Skaler ve matris gradyanlarının analitik türevlerle birebir eşleştiğini doğrula.
- **Verification Commands:**
  - `pytest tests/test_tensor.py tests/test_autograd.py`
- **Exit Criteria:** Tensör matris çarpımı ve karmaşık gradyan zincirleri doğru gradyan değerlerini hesaplamalı.

---

## Step 5: Bytecode Compiler & Virtual Machine (VM)
**Context:** AST'yi optimize edilebilir bytecode talimatlarına derleyen ve yürüten Synapse Sanal Makinesi.
- **Tasks:**
  - `synapse/vm/opcodes.py`: VM buyruk kümesini tanımla (`LOAD_CONST`, `STORE_NAME`, `BINARY_ADD`, `MATMUL`, `CALL_FUNCTION`, `JUMP_IF_FALSE`, vb.).
  - `synapse/vm/compiler.py`: AST'den Bytecode talimat dizisine derleyici.
  - `synapse/vm/virtual_machine.py`: Call stack, frame stack ve operand stack içeren sanal makine çekirdeği.
  - `tests/test_vm.py`: Döngüler, fonksiyon çağrıları ve tensör işlemlerini VM üzerinde çalıştırarak test et.
- **Verification Commands:**
  - `pytest tests/test_vm.py`
- **Exit Criteria:** Synapse kodu bytecode'a derlenip VM üzerinde hatasız yürütülmeli.

---

## Step 6: AI-Native Prompt & Agent Runtime
**Context:** Dil düzeyinde LLM çağrısı, structured JSON çıktı ve tool-calling desteği.
- **Tasks:**
  - `synapse/ai/providers.py`: Mock, OpenAI, Anthropic ve Ollama sağlayıcı arayüzleri.
  - `synapse/ai/prompt_engine.py`: Prompt şablonları, parametre enjeksiyonu ve JSON şema doğrulayıcı.
  - `synapse/ai/agent_runtime.py`: Araç çağırma (tool execution) döngüsü.
  - `tests/test_ai_primitives.py`: Yapılandırılmış çıktı ve araç çağırma testleri.
- **Verification Commands:**
  - `pytest tests/test_ai_primitives.py`
- **Exit Criteria:** `prompt` ve `agent` anahtar sözcükleri derlenip çalıştığında doğru yanıt ve fonksiyon tetiklemeleri yapabilmeli.

---

## Step 7: Python Interoperability (Bridge)
**Context:** Synapse içerisinden Python ekosistemindeki zengin kütüphaneleri (PyTorch, HuggingFace, NumPy) çağırma.
- **Tasks:**
  - `synapse/interop/python_bridge.py`: `import py.<modül>` direktifi ile Python modüllerini bağlayan FFI katmanı.
  - Synapse Tensor ile NumPy/PyTorch tensörleri arasında sıfır kopyalama/dönüştürme köprüsü.
  - `tests/test_interop.py`: Standart Python modüllerini (ör. `math`, `json`, `os`) çağıran test senaryoları.
- **Verification Commands:**
  - `pytest tests/test_interop.py`
- **Exit Criteria:** Synapse programları Python modüllerine erişebilmeli ve sonuçları işleyebilmeli.

---

## Step 8: CLI Tooling, REPL & Ready-to-Run Showcases
**Context:** Geliştiricinin dili kolayca kullanabilmesi için komut satırı aracı, etkileşimli kabuk ve eksiksiz örnekler.
- **Tasks:**
  - `synapse/cli.py`: `synapse run <file>`, `synapse repl`, `synapse ast <file>`, `synapse dis <file>` komutları.
  - `examples/01_tensor_math.syn`: Tensör aritmetiği ve matris çarpımı örneği.
  - `examples/02_autograd_training.syn`: Basit lineer regresyon modeli eğitimi (sıfırdan gradyan inişi - gradient descent).
  - `examples/03_ai_agent_prompt.syn`: Yapay zeka prompt ve structured veri çıkarma örneği.
- **Verification Commands:**
  - `python -m synapse.cli run examples/01_tensor_math.syn`
  - `python -m synapse.cli run examples/02_autograd_training.syn`
- **Exit Criteria:** Kullanıcı tek bir komutla Synapse dosyalarını çalıştırabilmeli veya REPL ile anlık denemeler yapabilmeli.
