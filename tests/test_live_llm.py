import os
import pytest
from synapse.ai.providers import (
    AutoProvider, OpenAIProvider, AnthropicProvider, OllamaProvider, MockProvider,
    set_llm_provider, reset_llm_provider, get_active_provider_name
)


@pytest.fixture(autouse=True)
def clean_env():
    # Test öncesi ortam değişkenlerini koru
    old_openai = os.environ.get("OPENAI_API_KEY")
    old_anthropic = os.environ.get("ANTHROPIC_API_KEY")
    yield
    reset_llm_provider()
    if old_openai:
        os.environ["OPENAI_API_KEY"] = old_openai
    else:
        os.environ.pop("OPENAI_API_KEY", None)
    if old_anthropic:
        os.environ["ANTHROPIC_API_KEY"] = old_anthropic
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_autoprovider_fallback_to_mock():
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)

    auto = AutoProvider()
    provider = auto.resolve_provider()
    # Hiçbir anahtar yoksa MockProvider seçilmeli
    assert isinstance(provider, (MockProvider, OllamaProvider))


def test_autoprovider_detects_openai():
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-123"
    auto = AutoProvider()
    provider = auto.resolve_provider()
    assert isinstance(provider, OpenAIProvider)
    assert provider.api_key == "sk-test-fake-key-123"


def test_autoprovider_detects_anthropic():
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-fake-key-123"
    auto = AutoProvider()
    provider = auto.resolve_provider()
    assert isinstance(provider, AnthropicProvider)
    assert provider.api_key == "sk-ant-test-fake-key-123"


def test_ollama_provider_init():
    ollama = OllamaProvider(host="http://localhost:11434", model="mistral")
    assert ollama.host == "http://localhost:11434"
    assert ollama.model == "mistral"
