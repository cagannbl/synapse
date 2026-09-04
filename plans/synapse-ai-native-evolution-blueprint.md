# Blueprint: Synapse AI-Native Geliştirici & Model Optimizasyonu

## Objective
Yapay zeka modellerinin (LLM'ler, Claude, GPT, DeepSeek, yerel 7B/8B modeller) Synapse (`.syn`) dilini sıfır sentaks hatasıyla, minimum token maliyetiyle yazabilmesi, hataları kendi kendine onarabilmesi (self-healing) ve dilin AI primitiflerini (tensör, autograd, swarm, memory) en üst düzeyde kullanabilmesi için derleyici, araç seti ve ekosistem seviyesinde yapılacak geliştirmelerin adım adım inşa planı.

---

## 1. Problem & Çözüm Mimarisi

### AI Modellerinin Yeni Dillerdeki Temel Darboğazları:
1. **Python Drift (Eğitim Yanlılığı):** LLM'ler trilyonlarca Python token'ı ile eğitildiği için Synapse yazarken gayriihtiyari `def`, `import numpy` veya `let`'siz değişken atamalarına kayarlar.
2. **Kriptik Hata Mesajları:** Standart derleyiciler sadece `SyntaxError: unexpected token` der; LLM hatayı anlamakta ve onarmakta zorlanır.
3. **Tensör Boyut Uyuşmazlıkları:** LLM'ler matris çarpımlarında (`[A, B] @ [C, D]`) boyutları akılda tutmakta zorlanır ve runtime'da crash olur.
4. **Entegrasyon Eksikliği:** Modelin yazdığı kodu derleyiciye anında doğrulatabileceği bir araç (MCP) olmaması.

---

## 2. Adım Adım İnşa Planı (Step-by-Step Blueprint)

### Step 1: AI Auto-Fix & Esnek Derleme Modu (Zero-Friction Ingestion)
**Context:** LLM'lerin Python alışkanlığıyla yaptığı ufak sentaks kaçaklarını derleme anında akıllıca normalize eden veya otomatik onaran `--ai-mode` bayrağı.
- **Tasks:**
  - `synapse/lexer/lexer.py` ve `synapse/parser/parser.py` içine esnek derleme bayrağı ekle:
    - `def foo():` görüldüğünde hata vermek yerine otomatik `fn foo():` olarak normalize et ve uyarı (warning) günlüğüne yaz.
    - `x = 10` (let olmadan yeni değişken ataması) görüldüğünde otomatik `let x = 10` olarak yorumla.
    - `import numpy as np` görüldüğünde "NumPy gereksiz, Synapse tensörleri yerleşiktir" uyarısıyla yerleşik tensörlere dönüştür.
  - `synapse/cli.py`: `synapse fmt --fix-ai-drift <file>` komutu ile hatalı yazılmış AI kodlarını anında geçerli Synapse sözdizimine dönüştüren CLI formatlayıcı.
- **Verification Commands:**
  - `pytest tests/test_ai_tolerant_parser.py`
- **Exit Criteria:** Pythonik `def` ve `let` eksiklikleri içeren bir kod parçası bu mod ile derlenip VM'de hatasız çalışmalı.

---

### Step 2: Yerleşik Synapse MCP Sunucusu (Model Context Protocol)
**Context:** Claude Desktop, Cursor, Antigravity ve diğer AI ajanlarının Synapse kodunu arka planda test edip derleyebileceği sıfır kurulumlu yerel MCP sunucusu.
- **Tasks:**
  - `synapse/mcp_server.py`: Standart JSON-RPC / FastMCP tabanlı MCP sunucusu geliştir.
  - Sağlanacak AI Araçları (Tools):
    1. `validate_synapse_code(code: str) -> DiagnosticReport`: Kodun AST ve tip geçerliliğini test edip JSON rapor döner.
    2. `execute_synapse_sandbox(code: str, timeout_sec: int) -> ExecutionResult`: Kodu izole VM'de çalıştırıp stdout/stderr ve değişken durumunu döner.
    3. `inspect_tensor_shapes(code: str) -> ShapeReport`: Kod içerisindeki matris ve tensör boyut zincirini döner.
    4. `transpile_to_c(code: str) -> str`: Kodu C eşdeğerine çevirir.
  - `synapse/cli.py`: `python -m synapse.cli mcp` komutuyla stdio üzerinden başlatılabilirlik sağla.
- **Verification Commands:**
  - `pytest tests/test_mcp_server.py`
- **Exit Criteria:** MCP istemcisi üzerinden gönderilen kod parçası, satır satır JSON tanılama ve onarım önerisiyle dönmeli.

---

### Step 3: Derleme Zamanı Tensör Boyut Çıkarımı (Compile-Time Shape Contracts)
**Context:** LLM'lerin en sık yaptığı `Matmul Dimension Mismatch` hatalarını runtime'a kalmadan AST analizi sırasında yakalayan sembolik şekil çıkarıcı.
- **Tasks:**
  - `synapse/parser/ast_nodes.py`: `TensorType(shape=(M, N), dtype=float)` tip belirteçlerini genişlet:
    - Örn: `fn matmul_layer(x: Tensor[B, In]) -> Tensor[B, Out]:`
  - `synapse/core/shape_checker.py`: Sembolik şekil denetleyicisi (Static Shape Inference Engine).
    - `A: [2, 3]` ve `B: [4, 5]` ise `A @ B` işleminde derleme anında: `"Shape Mismatch: Cannot multiply (2, 3) with (4, 5). Expected inner dimension 3, got 4."` uyarısı üret.
- **Verification Commands:**
  - `pytest tests/test_shape_checker.py`
- **Exit Criteria:** Geçersiz matris çarpımları sanal makineye geçmeden derleme zamanında AI-dostu net hata mesajıyla yakalanmalı.

---

### Step 4: Ultra-Yoğun AI Sistem Kartı & Token-Optimize Sözdizim Rehberi (Context-Saver)
**Context:** LLM'lerin bağlam penceresini (context window) şişirmeden, sadece ~300 token içinde Synapse dil kurallarını modele enjekte eden sıkıştırılmış format.
- **Tasks:**
  - `specs/synapse_llm_compact.json`: JSON formatında token tasarruflu kural haritası (Keywords, BNF, canonical few-shots).
  - `specs/system_prompt_dense.txt`: 250 tokenlık sistem promptu:
    - `def -> fn`, `let/const required`, `@ for matmul`, `|> for pipe`, `native tensor/autograd (NO numpy/torch)`, `agent/prompt keywords`.
  - CLI entegrasyonu: `python -m synapse.cli prompt-hint` komutu doğrudan kopyalanabilir sistem promptunu terminale basmalı.
- **Verification Commands:**
  - `python -m synapse.cli prompt-hint`
- **Exit Criteria:** Küçük boyutlu yerel modeller (Llama-3-8B, Qwen-2.5-7B) bu prompt ile sıfır sentaks hatasıyla kod üretebilmeli.

---

### Step 5: Sentetik Veri Seti Boru Hattının Ölçeklenmesi & LoRA / Fine-Tuning Dışa Aktarımı
**Context:** `tools/generate_dataset.py` dosyasını binlerce doğrulanmış örneğe genişletip popüler fine-tuning araçlarına (Unsloth, Axolotl, Llama-Factory) hazır veri seti üretmek.
- **Tasks:**
  - `tools/generate_dataset.py` içine yeni prosedürel şablonlar ekle:
    - CNN / Konvolüsyonel katmanlar & aktivasyonlar
    - `swarm` ve `debate` çoklu ajan iş akışları
    - `memory()` ile RAG ve yerel vektör arama kalıpları
    - `import py.<modül>` ile Python FFI kütüphane bağlama
  - Çoklu format desteği:
    - `--format alpaca` (instruction, input, output)
    - `--format sharegpt` (conversations role/content)
    - `--format chatml` (<|im_start|> formatı)
  - Otomatik filtreleme: Kod VM'de çalıştırılır, çıktısı doğrulanır, yalnızca %100 sıfır hata veren örnekler veri setine kaydedilir.
- **Verification Commands:**
  - `python tools/generate_dataset.py --count 100 --format sharegpt --output data/synapse_sharegpt_100.jsonl`
- **Exit Criteria:** 100 örneğin tamamı hatasız üretilmeli ve format standartlarına uymalı.

---

### Step 6: Synapse LSP (Language Server Protocol) & Akıllı Satır İçi Geri Bildirim
**Context:** VS Code / Cursor içerisinde kod yazan yapay zeka ve geliştiriciye gerçek zamanlı hata altı çizme (red squiggles) ve AI tamamlama sağlamak.
- **Tasks:**
  - `synapse/lsp/`: `pygls` (Python Generic Language Server) tabanlı hafif LSP sunucusu.
  - Özellikler:
    - `textDocument/didChange` ile anlık sözdizim ve tip doğrulaması.
    - `textDocument/hover` ile tensör şekillerini ve fonksiyon imzalarını gösterme.
    - `textDocument/codeAction` ile "Replace 'def' with 'fn'" veya "Add 'let' declaration" tek tıkla otomatik düzeltme (Quick Fix).
  - `vscode-synapse` eklentisine dil sunucusu istemcisini bağla.
- **Verification Commands:**
  - `pytest tests/test_lsp_server.py`
- **Exit Criteria:** VS Code içerisinde geçersiz sentaks girildiğinde anında kırmızı hata çizgisi ve önerilen onarım görünmeli.
