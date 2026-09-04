# Synapse Enterprise RAG Service Architecture

An industrial-grade, zero-external-dependency Semantic Search and Autonomous LLM Agent service implemented in the **Synapse AI-Native Programming Language**.

This architecture unifies **Multi-Dialect Relational Databases** (PostgreSQL/SQLite), **Hugging Face SafeTensors zero-copy weight loading**, **Native Vector Semantic Memory**, and **Microsecond TokenStream Server-Sent Events (SSE)** into a unified service.

---

## 1. Architectural Blueprint & Data Flow

```mermaid
flowchart TD
    subgraph ClientLayer["Client & Consumers"]
        ClientApp["Web / Enterprise Portal"]
        SSEListener["EventSource (SSE Stream)"]
    end

    subgraph SynapseServer["Synapse Native HTTP Engine (:8081)"]
        Router{"Route Dispatcher"}
        HealthEp["/api/health"]
        DocsEp["/api/documents"]
        SearchEp["/api/search"]
        StreamEp["/api/stream"]
    end

    subgraph RAGCore["Synapse RAG & Inference Core"]
        Agent["EnterpriseRAGAgent (Autonomous LLM)"]
        VectorMem[("Native Vector Memory (TF-IDF + Cosine)")]
        SafeTensorsEngine["SafeTensors Zero-Copy Buffer"]
        TokenStreamEngine["TokenStream Chunk Engine"]
    end

    subgraph StorageLayer["Multi-Dialect Storage Layer"]
        DB[("Database (PostgreSQL / SQLite)")]
        CatalogTable[("knowledge_catalog")]
        AuditTable[("access_audit_logs")]
        ModelFile[("embedding_weights.safetensors")]
    end

    ClientApp -->|HTTP GET/POST| Router
    SSEListener -->|GET text/event-stream| StreamEp

    Router --> HealthEp
    Router --> DocsEp
    Router --> SearchEp
    Router --> StreamEp

    DocsEp -->|Parameterized SQL| DB
    SearchEp -->|Semantic Recall| VectorMem
    SearchEp -->|Audit Entry| DB
    StreamEp -->|Generate Deltas| TokenStreamEngine
    TokenStreamEngine -->|Stream Chunks| SSEListener

    VectorMem --> Agent
    SafeTensorsEngine -->|mmap weights| ModelFile
    DB --> CatalogTable
    DB --> AuditTable
```

---

## 2. Key Architectural Components

### A. Multi-Dialect Relational Database (`Database`)
- **Dialect Auto-Detection:** Seamlessly targets embedded `sqlite:///:memory:` for local dev/testing, or production PostgreSQL clusters (`postgresql://user:pass@host:5432/rag_db`).
- **ACID Security:** All queries use parameterized bindings (`?` for SQLite, `$1, $2` for Postgres), eliminating SQL injection vulnerabilities.
- **Relational Metadata & Audit Trail:** Manages document access tiers (`Confidential`, `Restricted`, `Internal`) and keeps immutable access audit logs for compliance auditing.

### B. Hugging Face SafeTensors Weight Management (`save_safetensors` / `load_safetensors` / `safe_open`)
- **Zero-Copy Memory Mapping:** SafeTensors files are memory-mapped directly into memory without Python pickle security risks or deserialization memory spikes.
- **Header Metadata Integrity:** Preserves model architecture identifiers, embedding dimensions, and float32/float16 precision tags.

### C. Native Semantic Vector Memory (`memory()`)
- **Zero Third-Party Dependencies:** No ChromaDB, Pinecone, or Docker containers required. Embedded directly in the Synapse VM.
- **Cosine Similarity Retrieval:** Provides microsecond semantic similarity matching over chunked documents with custom metadata payload filtering.

### D. TokenStream SSE Akışı (Server-Sent Events)
- **Sub-Microsecond Latency:** Emits safe textual deltas with lazy buffer evaluation.
- **Cross-Chunk Stop Sequence Detection:** Automatically detects stop sequences (such as `<|im_end|>`, `[DONE]`) split across arbitrary chunk boundaries and cleanly terminates emission without leaking sentinel tokens.

---

## 3. API Endpoints Specification

| Method | Endpoint | Description | Content-Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health, dialect name, and model info | `application/json` |
| `GET` | `/api/documents` | List active documents from relational database | `application/json` |
| `GET` | `/api/search?q={query}` | Hybrid search (vector recall + relational audit log) | `application/json` |
| `GET` | `/api/stream?q={query}` | Real-time TokenStream SSE event stream | `text/event-stream` |

---

## 4. How to Run & Verify

### Running with Synapse CLI
```bash
# Run standalone service and self-test suite
synapse run examples/enterprise_rag_service/main.syn

# Or execute with AI-tolerant syntax repair
synapse run examples/enterprise_rag_service/main.syn --ai-tolerant
```

### Linting & Diagnostic Inspection
```bash
# Validate AST and static contracts
synapse check examples/enterprise_rag_service/main.syn
synapse lint examples/enterprise_rag_service/
```

### Testing with cURL / HTTP
```bash
# Health check
curl http://localhost:8081/api/health

# Query documents
curl http://localhost:8081/api/documents

# Semantic search
curl "http://localhost:8081/api/search?q=Zero-Trust+Security"

# Stream SSE tokens
curl -N "http://localhost:8081/api/stream?q=InfiniBand+GPU"
```
