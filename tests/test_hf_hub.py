"""Tests for Synapse AI Hugging Face Hub Client (synapse.ai.hf_hub).

Verifies URL resolution, token authentication headers, cache management,
streaming downloads, SHA-256 integrity verification, config retrieval,
transient HTTP retry backoff, sharded SafeTensors resolution, path traversal
security, and SafeTensors integration with Synapse Tensors.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Optional
import unittest.mock as mock
import urllib.error

import numpy as np
import pytest

from synapse.ai.hf_hub import (
    HFHubClient,
    compute_sha256,
    get_default_cache_dir,
)
from synapse.core.tensor import Tensor
from synapse.nn.safetensors import SafeTensorsDict, load_file, save_file


class MockHTTPResponse:
    """Mock for urllib.request.urlopen response context manager."""

    def __init__(
        self,
        data: bytes,
        headers: Optional[dict[str, str]] = None,
        status: int = 200,
    ) -> None:
        self._stream = io.BytesIO(data)
        self.status = status
        default_headers = {
            "Content-Length": str(len(data)),
            "Content-Type": "application/octet-stream",
        }
        if headers:
            default_headers.update(headers)
        self.headers = default_headers

    def read(self, amt: Optional[int] = None) -> bytes:
        if amt is None:
            return self._stream.read()
        return self._stream.read(amt)

    def __enter__(self) -> "MockHTTPResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


@pytest.fixture(autouse=True)
def clean_hf_env():
    """Isolate tests from ambient environment variables."""
    old_hf_token = os.environ.get("HF_TOKEN")
    old_hugging_face_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    old_hf_endpoint = os.environ.get("HF_ENDPOINT")
    old_synapse_cache = os.environ.get("SYNAPSE_CACHE_DIR")

    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ.pop("HF_ENDPOINT", None)
    os.environ.pop("SYNAPSE_CACHE_DIR", None)

    yield

    if old_hf_token is not None:
        os.environ["HF_TOKEN"] = old_hf_token
    else:
        os.environ.pop("HF_TOKEN", None)

    if old_hugging_face_token is not None:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = old_hugging_face_token
    else:
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    if old_hf_endpoint is not None:
        os.environ["HF_ENDPOINT"] = old_hf_endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)

    if old_synapse_cache is not None:
        os.environ["SYNAPSE_CACHE_DIR"] = old_synapse_cache
    else:
        os.environ.pop("SYNAPSE_CACHE_DIR", None)


# =============================================================================
# 1. URL Generation Tests
# =============================================================================
def test_url_generation_standard_and_custom(tmp_path: Path):
    """Test standard URL formatting, revisions, and custom endpoints."""
    client = HFHubClient(cache_dir=tmp_path)

    # Standard model and revision
    url1 = client.get_download_url("openai-community/gpt2", "config.json")
    assert url1 == "https://huggingface.co/openai-community/gpt2/resolve/main/config.json"

    # Custom revision (commit hash / branch / tag)
    url2 = client.get_download_url("meta-llama/Llama-2-7b", "model.safetensors", revision="refs/pr/1")
    assert url2 == "https://huggingface.co/meta-llama/Llama-2-7b/resolve/refs/pr/1/model.safetensors"

    # Leading slashes in filename should be stripped
    url3 = client.get_download_url("org/model", "/subfolder/weights.safetensors", revision="v2.0")
    assert url3 == "https://huggingface.co/org/model/resolve/v2.0/subfolder/weights.safetensors"

    # Custom endpoint with trailing slash stripped
    client_custom = HFHubClient(cache_dir=tmp_path, endpoint="https://hf-mirror.com/")
    assert client_custom.endpoint == "https://hf-mirror.com"
    url4 = client_custom.get_download_url("tiiuae/falcon-7b", "config.json")
    assert url4 == "https://hf-mirror.com/tiiuae/falcon-7b/resolve/main/config.json"


def test_url_generation_normalizes_windows_backslashes(tmp_path: Path):
    """Verify that Windows backslashes in repo_id, revision, or filename are normalized to forward slashes."""
    client = HFHubClient(cache_dir=tmp_path)
    url = client.get_download_url("org\\model", "subfolder\\weights.safetensors", revision="feature\\branch")
    assert "\\" not in url
    assert url == "https://huggingface.co/org/model/resolve/feature/branch/subfolder/weights.safetensors"


# =============================================================================
# 2. Authorization Header Tests
# =============================================================================
def test_auth_header_with_constructor_token(tmp_path: Path):
    """Test Bearer token formatting when token is passed to constructor."""
    client = HFHubClient(cache_dir=tmp_path, token="hf_abc123testtoken")
    headers = client.get_headers()
    assert headers["Authorization"] == "Bearer hf_abc123testtoken"
    assert "Synapse-AI" in headers["User-Agent"]


def test_auth_header_from_hf_token_env(tmp_path: Path):
    """Test Bearer token formatting read from HF_TOKEN environment variable."""
    os.environ["HF_TOKEN"] = "hf_env_secret_token"
    client = HFHubClient(cache_dir=tmp_path)
    headers = client.get_headers()
    assert headers["Authorization"] == "Bearer hf_env_secret_token"


def test_auth_header_from_huggingface_hub_token_env(tmp_path: Path):
    """Test Bearer token formatting read from HUGGING_FACE_HUB_TOKEN fallback."""
    os.environ["HUGGING_FACE_HUB_TOKEN"] = "hf_hub_fallback_token"
    client = HFHubClient(cache_dir=tmp_path)
    headers = client.get_headers()
    assert headers["Authorization"] == "Bearer hf_hub_fallback_token"


def test_auth_header_absent_when_no_token(tmp_path: Path):
    """Test that Authorization header is omitted when no token is configured."""
    client = HFHubClient(cache_dir=tmp_path)
    headers = client.get_headers()
    assert "Authorization" not in headers
    assert "User-Agent" in headers


def test_token_false_suppresses_auth_header_even_with_env(tmp_path: Path):
    """Test that token=False explicitly disables Authorization header even if env var exists."""
    os.environ["HF_TOKEN"] = "hf_env_secret_token"
    client = HFHubClient(cache_dir=tmp_path, token=False)
    headers = client.get_headers()
    assert "Authorization" not in headers


# =============================================================================
# 3. Cache Directory and Path Traversal Tests
# =============================================================================
def test_cache_directory_explicit_and_env(tmp_path: Path):
    """Test cache dir resolution via argument, environment variable, and default."""
    custom_dir = tmp_path / "my_custom_cache"
    client = HFHubClient(cache_dir=custom_dir)
    assert client.cache_dir == custom_dir.resolve()

    env_dir = tmp_path / "env_cache"
    os.environ["SYNAPSE_CACHE_DIR"] = str(env_dir)
    client_env = HFHubClient()
    assert client_env.cache_dir == env_dir.resolve()


def test_default_cache_directory():
    """Verify get_default_cache_dir returns a valid resolved Path."""
    default_dir = get_default_cache_dir()
    assert isinstance(default_dir, Path)
    assert default_dir.is_absolute()


def test_path_traversal_prevention_raises_value_error(tmp_path: Path):
    """Verify that path traversal attempts in filename or repo_id are blocked with ValueError."""
    client = HFHubClient(cache_dir=tmp_path)

    # Attempt to escape cache directory via filename
    with pytest.raises(ValueError, match="Path traversal detected"):
        client.get_local_path("test-org/model", "../../escaped_file.txt")

    # Attempt to escape cache directory via repo_id
    with pytest.raises(ValueError, match="Path traversal detected"):
        client.get_local_path("../../../evil_repo", "weights.bin")

    # Attempt to clear cache with traversal
    with pytest.raises(ValueError, match="Path traversal detected"):
        client.clear_cache("../../../evil_repo")


# =============================================================================
# 4. Download and Streaming Tests with Mock HTTP
# =============================================================================
def test_download_file_mock_http_chunked(tmp_path: Path):
    """Verify chunked streaming download to disk via mock HTTP."""
    client = HFHubClient(cache_dir=tmp_path)
    content = b"Mock model configuration payload data \x00\x01\x02\x03"

    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(content)) as mock_urlopen:
        result_path = client.download_file("test-org/test-model", "config.json", chunk_size=8)

        mock_urlopen.assert_called_once()
        assert result_path.is_file()
        assert result_path.read_bytes() == content
        assert client.is_cached("test-org/test-model", "config.json")


def test_download_progress_callback_invoked(tmp_path: Path):
    """Verify progress_callback receives byte counts and total bytes."""
    client = HFHubClient(cache_dir=tmp_path)
    content = b"0123456789" * 100  # 1000 bytes
    progress_history = []

    def callback(downloaded: int, total: Optional[int]):
        progress_history.append((downloaded, total))

    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(content)):
        client.download_file(
            "test-org/test-model",
            "weights.bin",
            chunk_size=100,
            progress_callback=callback,
        )

    assert len(progress_history) > 1
    assert progress_history[0] == (0, len(content))
    assert progress_history[-1] == (len(content), len(content))


# =============================================================================
# 5. Cache Hit and Force Download Tests
# =============================================================================
def test_cache_hit_does_not_redownload(tmp_path: Path):
    """Verify that download_file skips network requests when file already exists."""
    client = HFHubClient(cache_dir=tmp_path)
    local_file = client.get_local_path("test-org/test-model", "config.json")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(b"existing cached content")

    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        result = client.download_file("test-org/test-model", "config.json")
        mock_urlopen.assert_not_called()
        assert result == local_file
        assert result.read_bytes() == b"existing cached content"


def test_force_download_bypasses_cache(tmp_path: Path):
    """Verify that force_download=True overrides existing cache and re-downloads."""
    client = HFHubClient(cache_dir=tmp_path)
    local_file = client.get_local_path("test-org/test-model", "config.json")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(b"old stale content")

    new_content = b"new fresh content"
    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(new_content)) as mock_urlopen:
        result = client.download_file("test-org/test-model", "config.json", force_download=True)
        mock_urlopen.assert_called_once()
        assert result == local_file
        assert result.read_bytes() == new_content


# =============================================================================
# 6. SHA-256 Integrity Verification Tests
# =============================================================================
def test_sha256_verification_success(tmp_path: Path):
    """Verify that matching SHA-256 succeeds cleanly."""
    client = HFHubClient(cache_dir=tmp_path)
    content = b"Verified binary weights buffer content"
    expected_sha = hashlib.sha256(content).hexdigest()

    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(content)):
        path = client.download_file("test-org/model", "weights.bin", expected_sha256=expected_sha)
        assert path.is_file()
        assert compute_sha256(path) == expected_sha


def test_sha256_verification_mismatch_raises_and_cleans_up(tmp_path: Path):
    """Verify that SHA-256 mismatch raises ValueError and cleans up temporary file."""
    client = HFHubClient(cache_dir=tmp_path)
    content = b"Corrupted content from untrusted mirror"
    wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(content)):
        with pytest.raises(ValueError, match="SHA-256 verification failed"):
            client.download_file("test-org/model", "weights.bin", expected_sha256=wrong_sha)

    dest_path = client.get_local_path("test-org/model", "weights.bin")
    assert not dest_path.exists()
    tmp_files = list(dest_path.parent.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_cache_hit_with_sha256_mismatch_triggers_redownload(tmp_path: Path):
    """Verify that if a cached file's hash doesn't match expected_sha256, it re-downloads."""
    client = HFHubClient(cache_dir=tmp_path)
    local_file = client.get_local_path("test-org/model", "weights.bin")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(b"corrupted cached file")

    fresh_content = b"fresh uncorrupted remote content"
    correct_sha = hashlib.sha256(fresh_content).hexdigest()

    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(fresh_content)) as mock_urlopen:
        result = client.download_file("test-org/model", "weights.bin", expected_sha256=correct_sha)
        mock_urlopen.assert_called_once()
        assert result.read_bytes() == fresh_content


# =============================================================================
# 7. Error Handling and Retry Tests (HTTP 404, 401/403, 429, 503, 500)
# =============================================================================
def test_http_404_raises_file_not_found(tmp_path: Path):
    """Verify HTTP 404 response raises FileNotFoundError without retry."""
    client = HFHubClient(cache_dir=tmp_path)
    http_err = urllib.error.HTTPError(
        url="https://huggingface.co/nonexistent/resolve/main/model.bin",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )

    with mock.patch("urllib.request.urlopen", side_effect=http_err) as mock_urlopen:
        with pytest.raises(FileNotFoundError, match="not found in repo"):
            client.download_file("nonexistent/repo", "model.bin")
        assert mock_urlopen.call_count == 1


def test_http_401_raises_permission_error(tmp_path: Path):
    """Verify HTTP 401 / 403 unauthorized raises PermissionError without retry."""
    client = HFHubClient(cache_dir=tmp_path)
    http_err = urllib.error.HTTPError(
        url="https://huggingface.co/gated/resolve/main/model.bin",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=None,
    )

    with mock.patch("urllib.request.urlopen", side_effect=http_err) as mock_urlopen:
        with pytest.raises(PermissionError, match="Access denied"):
            client.download_file("gated/repo", "model.bin")
        assert mock_urlopen.call_count == 1


def test_retry_on_transient_http_429_succeeds(tmp_path: Path):
    """Verify that HTTP 429 Rate Limit triggers retry and succeeds on 2nd attempt."""
    client = HFHubClient(cache_dir=tmp_path, max_retries=2, retry_delay=0.01)
    err_429 = urllib.error.HTTPError(
        url="https://huggingface.co/test/resolve/main/model.bin",
        code=429,
        msg="Too Many Requests",
        hdrs={"Retry-After": "0"},
        fp=None,
    )
    content = b"success after rate limit backoff"

    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[err_429, MockHTTPResponse(content)],
    ) as mock_urlopen:
        res = client.download_file("test/repo", "model.bin")
        assert mock_urlopen.call_count == 2
        assert res.read_bytes() == content


def test_retry_on_transient_http_503_succeeds(tmp_path: Path):
    """Verify that HTTP 503 Service Unavailable triggers retry and succeeds on 2nd attempt."""
    client = HFHubClient(cache_dir=tmp_path, max_retries=2, retry_delay=0.01)
    err_503 = urllib.error.HTTPError(
        url="https://huggingface.co/test/resolve/main/model.bin",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=None,
    )
    content = b"success after server recovery"

    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[err_503, MockHTTPResponse(content)],
    ) as mock_urlopen:
        res = client.download_file("test/repo", "model.bin")
        assert mock_urlopen.call_count == 2
        assert res.read_bytes() == content


def test_retry_exhaustion_on_persistent_500_raises_runtime_error(tmp_path: Path):
    """Verify that persistent HTTP 500 exhausts max_retries and raises RuntimeError."""
    client = HFHubClient(cache_dir=tmp_path, max_retries=2, retry_delay=0.01)
    err_500 = urllib.error.HTTPError(
        url="https://huggingface.co/test/resolve/main/model.bin",
        code=500,
        msg="Internal Server Error",
        hdrs={},
        fp=None,
    )

    with mock.patch("urllib.request.urlopen", side_effect=err_500) as mock_urlopen:
        with pytest.raises(RuntimeError, match="HTTP error 500"):
            client.download_file("test/repo", "model.bin")
        # 1 initial attempt + 2 retries = 3 calls
        assert mock_urlopen.call_count == 3


# =============================================================================
# 8. Model Config Parsing Tests
# =============================================================================
def test_get_model_config_parses_json(tmp_path: Path):
    """Verify get_model_config downloads and parses config.json."""
    client = HFHubClient(cache_dir=tmp_path)
    config_dict = {
        "model_type": "gpt2",
        "vocab_size": 50257,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "architectures": ["GPT2LMHeadModel"],
    }
    raw_json = json.dumps(config_dict).encode("utf-8")

    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(raw_json)):
        parsed = client.get_model_config("openai-community/gpt2")
        assert parsed["model_type"] == "gpt2"
        assert parsed["vocab_size"] == 50257
        assert parsed["architectures"] == ["GPT2LMHeadModel"]


def test_get_model_config_from_local_directory(tmp_path: Path):
    """Verify get_model_config can directly read from a local directory."""
    client = HFHubClient(cache_dir=tmp_path)
    local_model_dir = tmp_path / "local_gpt2"
    local_model_dir.mkdir()
    config_data = {"model_type": "bert", "hidden_size": 768}
    (local_model_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")

    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        parsed = client.get_model_config(str(local_model_dir))
        mock_urlopen.assert_not_called()
        assert parsed == config_data


def test_local_directory_missing_config_raises_file_not_found(tmp_path: Path):
    """Verify passing a local directory missing config.json raises FileNotFoundError."""
    client = HFHubClient(cache_dir=tmp_path)
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        with pytest.raises(FileNotFoundError, match="config.json not found in local directory"):
            client.get_model_config(str(empty_dir))
        mock_urlopen.assert_not_called()


# =============================================================================
# 9. SafeTensors Single-File and Sharded Tests
# =============================================================================
def test_load_safetensors_model_integration(tmp_path: Path):
    """Integration test: save SafeTensors file, load via HFHubClient, and verify Synapse Tensors."""
    client = HFHubClient(cache_dir=tmp_path)

    # 1. Create real SafeTensors weights
    raw_weights_dir = tmp_path / "source_model"
    raw_weights_dir.mkdir()
    safetensors_source_path = raw_weights_dir / "model.safetensors"

    tensors_to_save = {
        "encoder.weight": Tensor(np.array([[1.0, 2.5], [-3.0, 4.25]], dtype=np.float32)),
        "encoder.bias": Tensor(np.array([0.1, -0.2], dtype=np.float32)),
        "layer_norm.gamma": Tensor(np.array([1.0, 1.0], dtype=np.float32)),
    }
    metadata = {"format": "synapse", "version": "1.0"}
    save_file(tensors_to_save, safetensors_source_path, metadata=metadata)
    safetensors_bytes = safetensors_source_path.read_bytes()

    # 2. Mock download of model.safetensors from repo 'synapse-ai/tiny-encoder'
    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(safetensors_bytes)):
        loaded_weights = client.load_safetensors_model("synapse-ai/tiny-encoder")

        assert isinstance(loaded_weights, SafeTensorsDict)
        assert loaded_weights.metadata == metadata
        assert loaded_weights["__metadata__"] == metadata

        for name, original in tensors_to_save.items():
            assert name in loaded_weights
            t = loaded_weights[name]
            assert isinstance(t, Tensor)
            assert t.shape == original.shape
            assert t.dtype == original.dtype
            np.testing.assert_allclose(t.data, original.data)


def test_open_safetensors_context_manager(tmp_path: Path):
    """Verify open_safetensors returns a working safe_open context manager."""
    client = HFHubClient(cache_dir=tmp_path)

    model_dir = tmp_path / "stream_model"
    model_dir.mkdir()
    st_path = model_dir / "model.safetensors"

    tensors = {
        "fc.weight": Tensor(np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)),
    }
    save_file(tensors, st_path)
    st_bytes = st_path.read_bytes()

    with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse(st_bytes)):
        with client.open_safetensors("synapse-ai/test-open") as f:
            assert "fc.weight" in f.keys()
            tensor = f.get_tensor("fc.weight")
            assert isinstance(tensor, Tensor)
            np.testing.assert_allclose(tensor.data, [[2.0, 4.0], [6.0, 8.0]])


def test_sharded_safetensors_loading_from_local_dir(tmp_path: Path):
    """Verify loading a multi-shard SafeTensors model from a local directory."""
    client = HFHubClient(cache_dir=tmp_path)

    sharded_dir = tmp_path / "local_sharded_model"
    sharded_dir.mkdir()

    # Create shard 1
    shard1_tensors = {
        "model.layers.0.weight": Tensor(np.array([1.0, 2.0], dtype=np.float32)),
    }
    save_file(shard1_tensors, sharded_dir / "model-00001-of-00002.safetensors")

    # Create shard 2
    shard2_tensors = {
        "model.layers.1.weight": Tensor(np.array([3.0, 4.0], dtype=np.float32)),
    }
    save_file(shard2_tensors, sharded_dir / "model-00002-of-00002.safetensors")

    # Create index.json
    index_data = {
        "metadata": {"total_size": 16},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    }
    (sharded_dir / "model.safetensors.index.json").write_text(json.dumps(index_data), encoding="utf-8")

    # Load from local dir
    loaded = client.load_safetensors_model(str(sharded_dir))
    assert isinstance(loaded, SafeTensorsDict)
    assert "model.layers.0.weight" in loaded
    assert "model.layers.1.weight" in loaded
    np.testing.assert_allclose(loaded["model.layers.0.weight"].data, [1.0, 2.0])
    np.testing.assert_allclose(loaded["model.layers.1.weight"].data, [3.0, 4.0])


def test_sharded_safetensors_loading_from_mock_hub(tmp_path: Path):
    """Verify loading a sharded SafeTensors model when repo has model.safetensors.index.json."""
    client = HFHubClient(cache_dir=tmp_path)

    # Shard 1 bytes
    t1 = {"w1": Tensor(np.array([10.0, 20.0], dtype=np.float32))}
    p1 = tmp_path / "shard1.safetensors"
    save_file(t1, p1)
    b1 = p1.read_bytes()

    # Shard 2 bytes
    t2 = {"w2": Tensor(np.array([30.0, 40.0], dtype=np.float32))}
    p2 = tmp_path / "shard2.safetensors"
    save_file(t2, p2)
    b2 = p2.read_bytes()

    # Index bytes
    index_content = json.dumps({
        "metadata": {"total_shards": 2},
        "weight_map": {
            "w1": "model-00001-of-00002.safetensors",
            "w2": "model-00002-of-00002.safetensors",
        },
    }).encode("utf-8")

    # When requesting model.safetensors -> 404
    # When requesting model.safetensors.index.json -> index_content
    # When requesting shard1 -> b1
    # When requesting shard2 -> b2
    def mock_urlopen_router(req: urllib.request.Request, timeout: float = 30.0):
        url = req.full_url
        if "model.safetensors.index.json" in url:
            return MockHTTPResponse(index_content)
        elif "model.safetensors" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        elif "model-00001-of-00002.safetensors" in url:
            return MockHTTPResponse(b1)
        elif "model-00002-of-00002.safetensors" in url:
            return MockHTTPResponse(b2)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen_router):
        loaded = client.load_safetensors_model("my-org/sharded-llm")
        assert isinstance(loaded, SafeTensorsDict)
        assert "w1" in loaded
        assert "w2" in loaded
        np.testing.assert_allclose(loaded["w1"].data, [10.0, 20.0])
        np.testing.assert_allclose(loaded["w2"].data, [30.0, 40.0])


# =============================================================================
# 10. Cache Management and Inspection Tests
# =============================================================================
def test_clear_cache_specific_repo_and_all(tmp_path: Path):
    """Verify clear_cache deletes specific repo or entire cache directory."""
    client = HFHubClient(cache_dir=tmp_path)

    # Create mock files for two repos
    repo_a_file = client.get_local_path("orgA/model1", "weights.bin")
    repo_b_file = client.get_local_path("orgB/model2", "weights.bin")
    repo_a_file.parent.mkdir(parents=True, exist_ok=True)
    repo_b_file.parent.mkdir(parents=True, exist_ok=True)
    repo_a_file.write_bytes(b"model1")
    repo_b_file.write_bytes(b"model2")

    assert repo_a_file.exists()
    assert repo_b_file.exists()

    # Clear orgA only
    client.clear_cache(repo_id="orgA/model1")
    assert not repo_a_file.exists()
    assert repo_b_file.exists()

    # Clear everything
    client.clear_cache()
    assert not tmp_path.exists() or len(list(tmp_path.iterdir())) == 0


def test_list_cached_files_and_repos(tmp_path: Path):
    """Verify list_cached_repos and list_cached_files inspection methods."""
    client = HFHubClient(cache_dir=tmp_path)

    f1 = client.get_local_path("orgX/modelA", "config.json")
    f2 = client.get_local_path("orgX/modelA", "model.safetensors")
    f3 = client.get_local_path("orgY/modelB", "config.json")

    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f3.parent.mkdir(parents=True, exist_ok=True)

    f1.write_text("cfg1", encoding="utf-8")
    f2.write_text("weights", encoding="utf-8")
    f3.write_text("cfg2", encoding="utf-8")

    repos = client.list_cached_repos()
    assert "orgX/modelA" in repos
    assert "orgY/modelB" in repos

    files_all = client.list_cached_files()
    assert len(files_all) == 3

    files_orgX = client.list_cached_files("orgX/modelA")
    assert len(files_orgX) == 2


def test_is_cached_and_get_cached_path(tmp_path: Path):
    """Verify is_cached and get_cached_path helper methods with and without hash."""
    client = HFHubClient(cache_dir=tmp_path)

    f = client.get_local_path("org/test", "test.bin")
    f.parent.mkdir(parents=True, exist_ok=True)
    content = b"cache test payload"
    f.write_bytes(content)
    expected_sha = hashlib.sha256(content).hexdigest()

    assert client.is_cached("org/test", "test.bin")
    assert client.is_cached("org/test", "test.bin", expected_sha256=expected_sha)
    assert not client.is_cached("org/test", "test.bin", expected_sha256="wronghash")
    assert not client.is_cached("org/test", "nonexistent.bin")

    cached_p = client.get_cached_path("org/test", "test.bin")
    assert cached_p == f
    assert client.get_cached_path("org/test", "nonexistent.bin") is None
