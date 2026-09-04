import re
import math
from typing import Any, Optional
from collections import Counter


class VectorMemory:
    """
    Yerleşik, sıfır bağımlılıklı anlamsal vektör bellek motoru (In-Memory Semantic Vector Store).
    Harici veritabanı (Chroma, Pinecone vb.) kurmadan doğrudan Synapse dilleinde
    anlamsal hafıza ve RAG arama imkanı sunar.
    """
    def __init__(self):
        self._documents: list[dict[str, Any]] = []
        self._doc_vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        # Kelimeleri ve karakter n-gramlarını ayıkla
        words = re.findall(r"\w+", text.lower())
        tokens = list(words)
        # Karakter 3-gramları (yazım hatalarına ve eklemelere karşı dayanıklılık)
        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tokens.append(f"#{w[i:i+3]}")
        return tokens

    def _compute_vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec: dict[str, float] = {}
        for token, count in tf.items():
            idf_val = self._idf.get(token, 1.5)
            vec[token] = (count / total) * idf_val

        # L2 Normalizasyonu
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def _update_idf(self):
        n_docs = len(self._documents) or 1
        doc_freq = Counter()
        for doc in self._documents:
            seen_tokens = set(self._tokenize(doc["text"]))
            for t in seen_tokens:
                doc_freq[t] += 1

        self._idf = {
            token: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }

        # Mevcut döküman vektörlerini güncelle
        self._doc_vectors = [
            self._compute_vector(self._tokenize(d["text"]))
            for d in self._documents
        ]

    def remember(self, text: str, metadata: Optional[dict[str, Any]] = None) -> int:
        """Yeni bir bilgiyi vektör hafızaya kaydeder."""
        doc_id = len(self._documents)
        record = {
            "id": doc_id,
            "text": str(text),
            "metadata": metadata or {}
        }
        self._documents.append(record)
        self._update_idf()
        return doc_id

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[dict[str, Any]]:
        """Sorguya en yakın anlamsal dökümanları cosine similarity ile sıralayıp döndürür."""
        if not self._documents:
            return []

        q_tokens = self._tokenize(str(query))
        q_vec = self._compute_vector(q_tokens)

        scored = []
        for i, doc_vec in enumerate(self._doc_vectors):
            # Cosine similarity (normalize edilmiş vektörlerin iç çarpımı)
            score = sum(doc_vec.get(token, 0.0) * val for token, val in q_vec.items())
            if score >= min_score:
                entry = dict(self._documents[i])
                entry["score"] = round(float(score), 4)
                scored.append(entry)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._documents)

    def clear(self):
        self._documents.clear()
        self._doc_vectors.clear()
        self._idf.clear()

    def __len__(self) -> int:
        return len(self._documents)

    def __repr__(self) -> str:
        return f"<VectorMemory items={len(self._documents)}>"


def memory() -> VectorMemory:
    """Synapse dilinde 'let brain = memory()' fabrikası."""
    return VectorMemory()
