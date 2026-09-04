from abc import ABC, abstractmethod
from typing import Any, Optional
import os
import json
import urllib.request
import urllib.error


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        tools: Optional[list[dict[str, Any]]] = None,
        json_schema: Optional[dict[str, Any]] = None
    ) -> str:
        pass


class MockProvider(BaseLLMProvider):
    """Birim testler ve çevrimdışı geliştirme için akıllı Mock Sağlayıcı."""
    def __init__(self, default_response: str = "Mock AI Response", responses: Optional[list[str]] = None):
        self.default_response = default_response
        self.responses: list[str] = list(responses) if responses else []
        self.last_prompt: Optional[str] = None
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        tools: Optional[list[dict[str, Any]]] = None,
        json_schema: Optional[dict[str, Any]] = None
    ) -> str:
        self.call_count += 1
        self.last_prompt = user_prompt

        if self.responses:
            return self.responses.pop(0)

        # Sentiment analizi kalıbı
        if "sentiment" in system_prompt.lower() or "sentiment" in user_prompt.lower():
            if any(w in user_prompt.lower() for w in ("good", "great", "fast", "love", "awesome", "intuitive")):
                return "POSITIVE"
            elif any(w in user_prompt.lower() for w in ("bad", "slow", "terrible", "hate", "bug")):
                return "NEGATIVE"
            return "NEUTRAL"

        if self.default_response != "Mock AI Response":
            return self.default_response

        # Şema JSON beklentisi
        if json_schema or "json" in system_prompt.lower():
            return json.dumps({"status": "success", "summary": f"Processed: {user_prompt[:30]}"})

        return f"[MockLLM] (System: {system_prompt}) Processed: {user_prompt}"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Sağlayıcısı (gpt-4o, gpt-4o-mini)."""
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        tools: Optional[list[dict[str, Any]]] = None,
        json_schema: Optional[dict[str, Any]] = None
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Sağlayıcısı (claude-3-5-sonnet, claude-3-5-haiku)."""
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        tools: Optional[list[dict[str, Any]]] = None,
        json_schema: Optional[dict[str, Any]] = None
    ) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]


class OllamaProvider(BaseLLMProvider):
    """Yerel Ollama Sağlayıcısı (Ücretsiz, Yerel, İnternetsiz LLM)."""
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host = host.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        tools: Optional[list[dict[str, Any]]] = None,
        json_schema: Optional[dict[str, Any]] = None
    ) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "options": {"temperature": temperature}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"]


class AutoProvider(BaseLLMProvider):
    """
    Akıllı Otomatik Sağlayıcı:
    1. OPENAI_API_KEY varsa -> OpenAI
    2. ANTHROPIC_API_KEY varsa -> Anthropic
    3. Yerel Ollama çalışıyorsa -> Ollama
    4. Hiçbiri yoksa -> Akıllı MockProvider (Asla çökmez!)
    """
    def __init__(self):
        self._provider: Optional[BaseLLMProvider] = None

    def resolve_provider(self) -> BaseLLMProvider:
        if self._provider is not None:
            return self._provider

        if os.getenv("OPENAI_API_KEY"):
            self._provider = OpenAIProvider()
        elif os.getenv("ANTHROPIC_API_KEY"):
            self._provider = AnthropicProvider()
        else:
            ollama = OllamaProvider()
            if ollama.is_available():
                self._provider = ollama
            else:
                self._provider = MockProvider()

        return self._provider

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        tools: Optional[list[dict[str, Any]]] = None,
        json_schema: Optional[dict[str, Any]] = None
    ) -> str:
        provider = self.resolve_provider()
        return provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            tools=tools,
            json_schema=json_schema
        )


_current_provider: BaseLLMProvider = AutoProvider()


def set_llm_provider(provider: BaseLLMProvider):
    global _current_provider
    _current_provider = provider


def reset_llm_provider():
    global _current_provider
    _current_provider = AutoProvider()


def get_llm_provider() -> BaseLLMProvider:
    global _current_provider
    return _current_provider


def get_active_provider_name() -> str:
    prov = get_llm_provider()
    if isinstance(prov, AutoProvider):
        active = prov.resolve_provider()
        return f"Auto ({active.__class__.__name__})"
    return prov.__class__.__name__
