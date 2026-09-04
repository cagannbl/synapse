"""Synapse AI Engine: Native Hugging Face Model Hub Client.

Provides zero-heavy-dependency model resolution, chunked streaming download,
local caching, SHA-256 integrity verification, transient retry with exponential backoff,
sharded SafeTensors index resolution, and seamless integration with Synapse SafeTensors weights.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Union
import urllib.error
import urllib.request
import uuid

from synapse.core.device import Device
from synapse.core.tensor import Tensor
from synapse.nn.safetensors import SafeTensorsDict, load_file, safe_open

logger = logging.getLogger("synapse.ai.hf_hub")


def _find_project_root() -> Optional[Path]:
    """Locates the project root by inspecting module location and current working directory."""
    # Check from module file location
    try:
        module_path = Path(__file__).resolve()
        for p in [module_path, *module_path.parents]:
            if (p / "pyproject.toml").exists() or (p / ".git").exists():
                return p
    except Exception:
        pass

    # Check from current working directory
    try:
        curr = Path.cwd().resolve()
        for p in [curr, *curr.parents]:
            if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "synapse").is_dir():
                return p
    except Exception:
        pass

    return None


def get_default_cache_dir() -> Path:
    """Determines the default cache directory:
    1. SYNAPSE_CACHE_DIR environment variable if specified.
    2. .synapse_models in project root (if root exists).
    3. ~/.synapse/models in user home directory.
    """
    if "SYNAPSE_CACHE_DIR" in os.environ:
        return Path(os.environ["SYNAPSE_CACHE_DIR"]).expanduser().resolve()

    root = _find_project_root()
    if root is not None:
        proj_cache = root / ".synapse_models"
        return proj_cache.resolve()

    return (Path.home() / ".synapse" / "models").resolve()


def compute_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calculates the lowercase hex SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class HFHubClient:
    """Hugging Face Model Hub Client for Synapse AI.

    Provides downloading, local caching, integrity verification,
    transient retry backoff, sharded weight resolution, configuration parsing,
    and SafeTensors loading into Synapse Tensors.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[Union[str, bool]] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        """Initializes the HFHubClient.

        Args:
            cache_dir: Custom path to local cache directory. Defaults to .synapse_models/ or ~/.synapse/models.
            token: Hugging Face user access token (or False to disable auth even if env vars exist).
                   Defaults to HF_TOKEN or HUGGING_FACE_HUB_TOKEN env vars.
            endpoint: Hugging Face base endpoint. Defaults to HF_ENDPOINT env var or https://huggingface.co.
            timeout: HTTP request timeout in seconds (default 30.0s).
            max_retries: Number of retry attempts on transient network or HTTP 429/5xx errors (default 3).
            retry_delay: Base delay in seconds for exponential backoff on retries (default 0.5s).
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            self.cache_dir = get_default_cache_dir().resolve()

        self.token = token
        raw_endpoint = endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        self.endpoint = raw_endpoint.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))

    def get_download_url(self, repo_id: str, filename: str, revision: str = "main") -> str:
        """Constructs the Hugging Face resolve URL for a given repository file.

        Normalizes backslashes to forward slashes for RFC 3986 compliance.
        Format: https://huggingface.co/{repo_id}/resolve/{revision}/{filename}
        """
        clean_repo = repo_id.strip("/\\").replace("\\", "/")
        clean_revision = revision.strip("/\\").replace("\\", "/")
        clean_filename = filename.lstrip("/\\").replace("\\", "/")
        return f"{self.endpoint}/{clean_repo}/resolve/{clean_revision}/{clean_filename}"

    def get_local_path(self, repo_id: str, filename: str, revision: str = "main") -> Path:
        """Calculates the local cache path for a given repository file with path traversal validation."""
        clean_repo = repo_id.strip("/\\")
        clean_revision = revision.strip("/\\")
        clean_filename = filename.lstrip("/\\")

        repo_dir = (self.cache_dir / clean_repo).resolve()
        try:
            repo_dir.relative_to(self.cache_dir)
        except ValueError:
            raise ValueError(f"Path traversal detected in repo_id '{repo_id}'")

        revision_dir = (repo_dir / clean_revision).resolve()
        try:
            revision_dir.relative_to(repo_dir)
        except ValueError:
            raise ValueError(f"Path traversal detected in revision '{revision}'")

        target_path = (revision_dir / clean_filename).resolve()
        try:
            target_path.relative_to(revision_dir)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: path '{target_path}' escapes revision directory '{revision_dir}'"
            )

        return target_path

    def is_cached(
        self,
        repo_id: str,
        filename: str,
        revision: str = "main",
        expected_sha256: Optional[str] = None,
    ) -> bool:
        """Checks whether the file already exists in the local cache and matches hash if given."""
        try:
            path = self.get_local_path(repo_id, filename, revision)
        except ValueError:
            return False

        if not path.is_file():
            return False

        if expected_sha256 is not None:
            return compute_sha256(path) == expected_sha256.strip().lower()

        return True

    def get_cached_path(
        self,
        repo_id: str,
        filename: str,
        revision: str = "main",
    ) -> Optional[Path]:
        """Returns the cached Path if the file is locally available, otherwise None."""
        if self.is_cached(repo_id, filename, revision):
            return self.get_local_path(repo_id, filename, revision)
        return None

    def get_headers(self) -> Dict[str, str]:
        """Constructs request headers, including Authorization if token is configured."""
        headers = {
            "User-Agent": "Synapse-AI/1.0 (HFHubClient; Windows x64)",
            "Accept": "*/*",
        }
        if self.token is False:
            token = None
        elif isinstance(self.token, str):
            token = self.token
        else:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
        return headers

    def download_file(
        self,
        repo_id: str,
        filename: str,
        revision: str = "main",
        force_download: bool = False,
        expected_sha256: Optional[str] = None,
        chunk_size: int = 64 * 1024,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> Path:
        """Downloads a file from Hugging Face Hub with chunked streaming and cache management.

        Args:
            repo_id: Hugging Face repository ID (e.g., 'openai-community/gpt2').
            filename: Target file name (e.g., 'config.json', 'model.safetensors').
            revision: Branch, tag, or commit hash (default 'main').
            force_download: If True, bypasses local cache and re-downloads.
            expected_sha256: Optional SHA-256 hash string for integrity verification.
            chunk_size: Streaming chunk size in bytes (default 64KB).
            progress_callback: Optional callback receiving (bytes_downloaded, total_bytes).

        Returns:
            Path object pointing to the cached file on disk.

        Raises:
            FileNotFoundError: If the remote file or repository does not exist (HTTP 404).
            PermissionError: If unauthorized or forbidden (HTTP 401/403).
            ValueError: If downloaded or cached file fails SHA-256 verification or has path traversal.
            RuntimeError: On network or HTTP transfer errors after retry exhaustion.
        """
        dest_path = self.get_local_path(repo_id, filename, revision)

        # Cache check
        if dest_path.is_file() and not force_download:
            if expected_sha256 is not None:
                cached_sha = compute_sha256(dest_path)
                if cached_sha == expected_sha256.strip().lower():
                    logger.debug(f"Cache hit with verified SHA-256 for {dest_path}")
                    return dest_path
                logger.warning(
                    f"Cache hit for {dest_path} failed SHA-256 verification "
                    f"(cached={cached_sha}, expected={expected_sha256}). Re-downloading."
                )
            else:
                logger.debug(f"Cache hit for {dest_path}")
                return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = self.get_download_url(repo_id, filename, revision)
        headers = self.get_headers()
        req = urllib.request.Request(url, headers=headers, method="GET")

        temp_filename = f".{dest_path.name}.{uuid.uuid4().hex}.tmp"
        temp_path = dest_path.parent / temp_filename

        attempts = 0
        total_attempts = 1 + self.max_retries

        while attempts < total_attempts:
            attempts += 1
            hasher = hashlib.sha256()
            bytes_downloaded = 0

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    content_length_hdr = resp.headers.get("Content-Length")
                    total_bytes: Optional[int] = None
                    if content_length_hdr and content_length_hdr.strip().isdigit():
                        total_bytes = int(content_length_hdr.strip())

                    if progress_callback:
                        progress_callback(0, total_bytes)

                    with open(temp_path, "wb") as f_out:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            hasher.update(chunk)
                            bytes_downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(bytes_downloaded, total_bytes)

                # Successfully downloaded stream
                break

            except urllib.error.HTTPError as e:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass

                if e.code == 404:
                    raise FileNotFoundError(
                        f"File '{filename}' not found in repo '{repo_id}' at revision '{revision}'. "
                        f"URL: {url} (HTTP 404)"
                    ) from e
                elif e.code in (401, 403):
                    raise PermissionError(
                        f"Access denied to '{url}' (HTTP {e.code}). "
                        f"Please verify permissions or supply a valid HF_TOKEN."
                    ) from e

                # Retryable HTTP status codes
                if e.code in (429, 500, 502, 503, 504) and attempts < total_attempts:
                    retry_after = e.headers.get("Retry-After") if hasattr(e, "headers") else None
                    if retry_after and retry_after.strip().isdigit():
                        delay = min(5.0, float(retry_after.strip()))
                    else:
                        delay = min(5.0, self.retry_delay * (2 ** (attempts - 1)))
                    logger.warning(
                        f"HTTP {e.code} downloading {url}. Retrying in {delay:.2f}s "
                        f"(attempt {attempts}/{total_attempts})..."
                    )
                    time.sleep(delay)
                    continue

                raise RuntimeError(
                    f"HTTP error {e.code} while downloading {url}: {e.reason}"
                ) from e

            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass

                if attempts < total_attempts:
                    delay = min(5.0, self.retry_delay * (2 ** (attempts - 1)))
                    logger.warning(
                        f"Network error '{e}' downloading {url}. Retrying in {delay:.2f}s "
                        f"(attempt {attempts}/{total_attempts})..."
                    )
                    time.sleep(delay)
                    continue

                raise RuntimeError(
                    f"Network error downloading {url} after {attempts} attempts: {e}"
                ) from e

            except Exception:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                raise

        actual_sha256 = hasher.hexdigest()
        if expected_sha256 is not None:
            clean_expected = expected_sha256.strip().lower()
            if actual_sha256 != clean_expected:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                raise ValueError(
                    f"SHA-256 verification failed for '{filename}' from '{repo_id}': "
                    f"expected {clean_expected}, computed {actual_sha256}"
                )

        os.replace(temp_path, dest_path)
        return dest_path

    def get_model_config(
        self,
        repo_id: str,
        revision: str = "main",
        force_download: bool = False,
        expected_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Downloads and parses config.json for the specified model repository.

        Args:
            repo_id: Hugging Face model repository ID (or local directory path).
            revision: Git revision/branch/tag (default 'main').
            force_download: If True, bypasses local cache and re-downloads.
            expected_sha256: Optional SHA-256 checksum for config.json.

        Returns:
            Dictionary containing model configuration JSON.
        """
        if os.path.isdir(repo_id):
            local_config = Path(repo_id) / "config.json"
            if local_config.is_file():
                with open(local_config, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise FileNotFoundError(f"config.json not found in local directory '{repo_id}'")

        config_path = self.download_file(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            force_download=force_download,
            expected_sha256=expected_sha256,
        )
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_sharded_from_index(
        self,
        index_path: Path,
        base_dir: Path,
        device: Union[Device, str],
        use_mmap: bool,
    ) -> SafeTensorsDict:
        """Loads all shards of a sharded SafeTensors model from a local directory."""
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        weight_map: Dict[str, str] = index_data.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"SafeTensors index file '{index_path}' does not contain 'weight_map'.")

        metadata = index_data.get("metadata", {})
        unique_shards = sorted(set(weight_map.values()))
        result_dict = SafeTensorsDict(metadata=metadata)

        for shard_name in unique_shards:
            shard_path = base_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Shard '{shard_name}' listed in '{index_path}' not found in '{base_dir}'."
                )
            shard_loaded = load_file(shard_path, device=device, use_mmap=use_mmap)
            for k, v in shard_loaded.items():
                if k != "__metadata__":
                    result_dict[k] = v
            if hasattr(shard_loaded, "metadata") and shard_loaded.metadata:
                result_dict.metadata.update(shard_loaded.metadata)

        return result_dict

    def _load_sharded_from_hub(
        self,
        index_path: Path,
        repo_id: str,
        revision: str,
        device: Union[Device, str],
        force_download: bool,
        use_mmap: bool,
    ) -> SafeTensorsDict:
        """Downloads and loads all shards of a sharded SafeTensors model from Hugging Face Hub."""
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        weight_map: Dict[str, str] = index_data.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"SafeTensors index file '{index_path}' does not contain 'weight_map'.")

        metadata = index_data.get("metadata", {})
        unique_shards = sorted(set(weight_map.values()))
        result_dict = SafeTensorsDict(metadata=metadata)

        for shard_name in unique_shards:
            shard_path = self.download_file(
                repo_id=repo_id,
                filename=shard_name,
                revision=revision,
                force_download=force_download,
            )
            shard_loaded = load_file(shard_path, device=device, use_mmap=use_mmap)
            for k, v in shard_loaded.items():
                if k != "__metadata__":
                    result_dict[k] = v
            if hasattr(shard_loaded, "metadata") and shard_loaded.metadata:
                result_dict.metadata.update(shard_loaded.metadata)

        return result_dict

    def load_safetensors_model(
        self,
        repo_id: str,
        filename: str = "model.safetensors",
        revision: str = "main",
        device: Union[Device, str] = "cpu",
        force_download: bool = False,
        expected_sha256: Optional[str] = None,
        use_mmap: bool = False,
    ) -> SafeTensorsDict:
        """Downloads or resolves local SafeTensors weights and loads them into Synapse Tensors.
        Seamlessly resolves single-file weights or multi-file sharded index models.

        Args:
            repo_id: Hugging Face repository ID or local path to file/directory.
            filename: Name of the SafeTensors file or index (default 'model.safetensors').
            revision: Git revision (default 'main').
            device: Target device for loaded Tensors (e.g. 'cpu', 'cuda').
            force_download: If True, bypasses cache and re-downloads.
            expected_sha256: Optional SHA-256 verification checksum.
            use_mmap: If True, uses memory mapping (default False for safe Windows file handles).

        Returns:
            SafeTensorsDict mapping tensor names to Synapse Tensor objects.
        """
        # Case 1: repo_id is an explicit local file
        if os.path.isfile(repo_id):
            return load_file(Path(repo_id), device=device, use_mmap=use_mmap)

        # Case 2: repo_id is an existing local directory
        if os.path.isdir(repo_id):
            local_dir = Path(repo_id)
            target_file = local_dir / filename
            if target_file.is_file():
                if filename.endswith(".index.json"):
                    return self._load_sharded_from_index(
                        index_path=target_file,
                        base_dir=local_dir,
                        device=device,
                        use_mmap=use_mmap,
                    )
                return load_file(target_file, device=device, use_mmap=use_mmap)

            # Check if sharded index exists in local directory
            if filename == "model.safetensors" and (local_dir / "model.safetensors.index.json").is_file():
                return self._load_sharded_from_index(
                    index_path=local_dir / "model.safetensors.index.json",
                    base_dir=local_dir,
                    device=device,
                    use_mmap=use_mmap,
                )

            raise FileNotFoundError(
                f"SafeTensors file '{filename}' not found in local directory '{repo_id}'"
            )

        # Case 3: Hugging Face remote repository
        if filename.endswith(".index.json"):
            index_path = self.download_file(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                force_download=force_download,
                expected_sha256=expected_sha256,
            )
            return self._load_sharded_from_hub(
                index_path=index_path,
                repo_id=repo_id,
                revision=revision,
                device=device,
                force_download=force_download,
                use_mmap=use_mmap,
            )

        # Attempt to download single file; if 404 and default filename, try sharded index
        try:
            local_path = self.download_file(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                force_download=force_download,
                expected_sha256=expected_sha256,
            )
            return load_file(local_path, device=device, use_mmap=use_mmap)
        except FileNotFoundError:
            if filename == "model.safetensors":
                try:
                    index_path = self.download_file(
                        repo_id=repo_id,
                        filename="model.safetensors.index.json",
                        revision=revision,
                        force_download=force_download,
                    )
                    return self._load_sharded_from_hub(
                        index_path=index_path,
                        repo_id=repo_id,
                        revision=revision,
                        device=device,
                        force_download=force_download,
                        use_mmap=use_mmap,
                    )
                except FileNotFoundError:
                    pass
            raise

    def open_safetensors(
        self,
        repo_id: str,
        filename: str = "model.safetensors",
        revision: str = "main",
        device: Union[Device, str] = "cpu",
        force_download: bool = False,
        expected_sha256: Optional[str] = None,
    ) -> safe_open:
        """Downloads or resolves local SafeTensors weights and opens them on-demand via safe_open."""
        if os.path.isfile(repo_id):
            local_path = Path(repo_id)
        elif os.path.isdir(repo_id):
            candidate = Path(repo_id) / filename
            if candidate.is_file():
                local_path = candidate
            else:
                raise FileNotFoundError(
                    f"SafeTensors file '{filename}' not found in local directory '{repo_id}'"
                )
        else:
            local_path = self.download_file(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                force_download=force_download,
                expected_sha256=expected_sha256,
            )

        return safe_open(local_path, framework="synapse", device=device)

    def clear_cache(self, repo_id: Optional[str] = None) -> None:
        """Clears cached files for a specific repository or the entire cache."""
        if repo_id is not None:
            clean_repo = repo_id.strip("/\\")
            repo_dir = (self.cache_dir / clean_repo).resolve()
            try:
                repo_dir.relative_to(self.cache_dir)
            except ValueError:
                raise ValueError(f"Path traversal detected in repo_id '{repo_id}'")
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir, ignore_errors=True)

    def list_cached_repos(self) -> List[str]:
        """Lists all repository IDs currently present in the cache."""
        if not self.cache_dir.is_dir():
            return []
        repos: List[str] = []
        for path in self.cache_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                subdirs = [s for s in path.iterdir() if s.is_dir() and not s.name.startswith(".")]
                has_nested_repos = any(
                    any(child.is_dir() for child in s.iterdir()) for s in subdirs if s.is_dir()
                )
                if has_nested_repos:
                    for sub in subdirs:
                        repos.append(f"{path.name}/{sub.name}")
                else:
                    repos.append(path.name)
        return sorted(repos)

    def list_cached_files(self, repo_id: Optional[str] = None) -> List[Path]:
        """Lists all cached model file paths, optionally filtered by repository ID."""
        if repo_id is not None:
            clean_repo = repo_id.strip("/\\")
            target_dir = (self.cache_dir / clean_repo).resolve()
            try:
                target_dir.relative_to(self.cache_dir)
            except ValueError:
                raise ValueError(f"Path traversal detected in repo_id '{repo_id}'")
            if not target_dir.is_dir():
                return []
            return sorted([f for f in target_dir.rglob("*") if f.is_file() and not f.name.startswith(".")])

        if not self.cache_dir.is_dir():
            return []
        return sorted([f for f in self.cache_dir.rglob("*") if f.is_file() and not f.name.startswith(".")])
