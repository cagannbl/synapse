"""
Synapse Edge NanoGPT Lightweight UTF-8 Character / Byte Tokenizer
==================================================================
Converts text prompts into numeric token tensors and decodes model output IDs.
"""

from typing import List


class EdgeNanoTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Encodes UTF-8 string to a list of byte integers [0..255]."""
        return list(text.encode("utf-8"))

    def decode(self, token_ids: List[int]) -> str:
        """Decodes byte integer list back to a string."""
        safe_bytes = bytes([t % 256 for t in token_ids])
        return safe_bytes.decode("utf-8", errors="replace")


def get_default_tokenizer() -> EdgeNanoTokenizer:
    return EdgeNanoTokenizer(vocab_size=256)
