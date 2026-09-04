"""
Synapse Package Registry Client & Cache Manager.

Provides native package search, installation with SHA-256 checksum verification,
deterministic lockfile (synapse.lock) updates, offline/cache support, package publishing,
and built-in default package indices (synapse-nn, synapse-vision, synapse-nlp, synapse-math).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from typing import Any, Optional, Sequence, Union

from synapse.pkg.manager import TomlHelper, PackageManager


# =========================================================================
# Custom Exceptions
# =========================================================================

class PackageError(Exception):
    """Base exception for Synapse package registry errors."""
    pass


class PackageNotFoundError(PackageError):
    """Raised when a requested package is not found in the registry or cache."""
    pass


class ChecksumMismatchError(PackageError):
    """Raised when package archive SHA-256 checksum does not match expected hash."""
    pass


class ManifestError(PackageError):
    """Raised when synapse.toml manifest is missing or invalid."""
    pass


# =========================================================================
# Built-in Default Packages (Offline & Zero-Config Registry)
# =========================================================================

BUILTIN_PACKAGES: dict[str, dict[str, Any]] = {
    "synapse-nn": {
        "name": "synapse-nn",
        "version": "0.2.0",
        "description": "Neural network layers, activations, optimizers, and loss functions for Synapse AI",
        "author": "Synapse AI Core Team <core@synapse-lang.org>",
        "entry": "index.syn",
        "tags": ["nn", "deep-learning", "neural-network", "ai"],
        "files": {
            "synapse.toml": (
                '[package]\n'
                'name = "synapse-nn"\n'
                'version = "0.2.0"\n'
                'description = "Neural network layers, activations, optimizers, and loss functions for Synapse AI"\n'
                'authors = ["Synapse AI Core Team"]\n'
                'entry = "index.syn"\n'
                '\n[dependencies]\n'
                'synapse-math = "0.3.0"\n'
            ),
            "index.syn": (
                '// Synapse Neural Network Standard Library v0.2.0\n'
                'fn relu(x):\n'
                '    return x.clamp_min(0.0)\n\n'
                'fn sigmoid(x):\n'
                '    return 1.0 / (1.0 + exp(-x))\n\n'
                'fn info():\n'
                '    return "Synapse Neural Network Library v0.2.0"\n'
            ),
            "layers.syn": (
                '// Neural Network Layers\n'
                'fn linear(x, weights, bias):\n'
                '    return matmul(x, weights) + bias\n'
            ),
            "README.md": (
                '# Synapse NN\n\n'
                'Standard neural network primitives for the Synapse programming language.\n'
            )
        }
    },
    "synapse-vision": {
        "name": "synapse-vision",
        "version": "0.1.5",
        "description": "Computer vision models, image transforms, and datasets for Synapse AI",
        "author": "Synapse Vision Team <vision@synapse-lang.org>",
        "entry": "index.syn",
        "tags": ["vision", "cv", "image-processing", "transforms"],
        "files": {
            "synapse.toml": (
                '[package]\n'
                'name = "synapse-vision"\n'
                'version = "0.1.5"\n'
                'description = "Computer vision models, image transforms, and datasets for Synapse AI"\n'
                'authors = ["Synapse Vision Team"]\n'
                'entry = "index.syn"\n'
                '\n[dependencies]\n'
                'synapse-nn = "0.2.0"\n'
            ),
            "index.syn": (
                '// Synapse Computer Vision Library v0.1.5\n'
                'fn normalize(img, mean, std):\n'
                '    return (img - mean) / std\n\n'
                'fn info():\n'
                '    return "Synapse Vision Library v0.1.5"\n'
            ),
            "README.md": (
                '# Synapse Vision\n\n'
                'Computer vision modules and image transformations for Synapse.\n'
            )
        }
    },
    "synapse-nlp": {
        "name": "synapse-nlp",
        "version": "0.1.2",
        "description": "Natural language processing, tokenizers, and embeddings for Synapse AI",
        "author": "Synapse NLP Team <nlp@synapse-lang.org>",
        "entry": "index.syn",
        "tags": ["nlp", "text", "tokenizer", "transformers", "llm"],
        "files": {
            "synapse.toml": (
                '[package]\n'
                'name = "synapse-nlp"\n'
                'version = "0.1.2"\n'
                'description = "Natural language processing, tokenizers, and embeddings for Synapse AI"\n'
                'authors = ["Synapse NLP Team"]\n'
                'entry = "index.syn"\n'
            ),
            "index.syn": (
                '// Synapse NLP Library v0.1.2\n'
                'fn tokenize(text):\n'
                '    return text.split(" ")\n\n'
                'fn info():\n'
                '    return "Synapse NLP Library v0.1.2"\n'
            ),
            "README.md": (
                '# Synapse NLP\n\n'
                'Natural Language Processing for Synapse.\n'
            )
        }
    },
    "synapse-math": {
        "name": "synapse-math",
        "version": "0.3.0",
        "description": "Advanced linear algebra, matrix decompositions, and numerical algorithms",
        "author": "Synapse Math Team <math@synapse-lang.org>",
        "entry": "index.syn",
        "tags": ["math", "linear-algebra", "linalg", "scientific"],
        "files": {
            "synapse.toml": (
                '[package]\n'
                'name = "synapse-math"\n'
                'version = "0.3.0"\n'
                'description = "Advanced linear algebra, matrix decompositions, and numerical algorithms"\n'
                'authors = ["Synapse Math Team"]\n'
                'entry = "index.syn"\n'
            ),
            "index.syn": (
                '// Synapse Math Library v0.3.0\n'
                'fn det(matrix):\n'
                '    return 1.0\n\n'
                'fn info():\n'
                '    return "Synapse Math Library v0.3.0"\n'
            ),
            "README.md": (
                '# Synapse Math\n\n'
                'Linear algebra and scientific computing primitives for Synapse.\n'
            )
        }
    }
}


# =========================================================================
# Checksum and Archive Utilities
# =========================================================================

def compute_file_sha256(filepath: str) -> str:
    """Calculates SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Calculates SHA-256 checksum of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# =========================================================================
# PackageRegistry Class
# =========================================================================

class PackageRegistry:
    """
    Synapse Package Registry Client and Local Cache Manager.
    Handles package searching, checksum verification, bundle publishing,
    and deterministic lockfile updates.
    """

    INDEX_FILENAME = "index.json"
    LOCKFILE_FILENAME = "synapse.lock"
    MANIFEST_FILENAME = "synapse.toml"
    MODULES_DIRNAME = "syn_modules"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        registry_dir: Optional[str] = None,
        default_project_dir: Optional[str] = None,
    ):
        # Cache directory (defaults to ~/.synapse/cache)
        if cache_dir is not None:
            self.cache_dir = os.path.abspath(cache_dir)
        else:
            env_cache = os.environ.get("SYNAPSE_CACHE_DIR")
            if env_cache:
                self.cache_dir = os.path.abspath(env_cache)
            else:
                self.cache_dir = os.path.abspath(os.path.expanduser("~/.synapse/cache"))

        # Registry repository directory (defaults to ~/.synapse/registry)
        if registry_dir is not None:
            self.registry_dir = os.path.abspath(registry_dir)
        else:
            env_reg = os.environ.get("SYNAPSE_REGISTRY_DIR")
            if env_reg:
                self.registry_dir = os.path.abspath(env_reg)
            else:
                self.registry_dir = os.path.abspath(os.path.expanduser("~/.synapse/registry"))

        self.default_project_dir = os.path.abspath(default_project_dir or os.getcwd())

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.registry_dir, exist_ok=True)

        self._init_registry_index()

    def _init_registry_index(self) -> None:
        """Initializes the registry index file if missing."""
        index_path = os.path.join(self.registry_dir, self.INDEX_FILENAME)
        if not os.path.exists(index_path):
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "packages": {}}, f, indent=2)

    def _load_registry_index(self) -> dict[str, Any]:
        """Loads packages index from registry_dir."""
        index_path = os.path.join(self.registry_dir, self.INDEX_FILENAME)
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"version": 1, "packages": {}}

    def _save_registry_index(self, index_data: dict[str, Any]) -> None:
        """Saves packages index to registry_dir."""
        index_path = os.path.join(self.registry_dir, self.INDEX_FILENAME)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # 1. Search Packages
    # =========================================================================

    def search(self, query: str = "", project_dir: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Searches available packages across built-in index, published packages, and cache.

        Args:
            query: Search keyword (matches name, description, tags, author). Empty or '*' returns all.
            project_dir: Optional project directory to check installed status.

        Returns:
            List of matching package information dicts.
        """
        query_norm = (query or "").strip().lower()
        match_all = not query_norm or query_norm == "*"

        all_pkgs: dict[str, dict[str, Any]] = {}

        # 1. Load built-in packages
        for name, meta in BUILTIN_PACKAGES.items():
            all_pkgs[name] = {
                "name": name,
                "version": meta["version"],
                "description": meta.get("description", ""),
                "author": meta.get("author", ""),
                "tags": meta.get("tags", []),
                "source": "built-in"
            }

        # 2. Load published packages from registry storage index
        reg_index = self._load_registry_index()
        for name, meta in reg_index.get("packages", {}).items():
            all_pkgs[name] = {
                "name": name,
                "version": meta.get("version", "0.1.0"),
                "description": meta.get("description", ""),
                "author": meta.get("author", ""),
                "tags": meta.get("tags", []),
                "checksum": meta.get("checksum", ""),
                "source": "registry"
            }

        # Check installed packages in project
        target_dir = os.path.abspath(project_dir or self.default_project_dir)
        modules_dir = os.path.join(target_dir, self.MODULES_DIRNAME)
        installed_names = set()
        if os.path.exists(modules_dir):
            for item in os.listdir(modules_dir):
                if os.path.isdir(os.path.join(modules_dir, item)):
                    installed_names.add(item)

        results: list[dict[str, Any]] = []
        for name, pkg in all_pkgs.items():
            pkg["installed"] = name in installed_names

            if match_all:
                results.append(pkg)
                continue

            # Check match against query
            haystack = [
                name.lower(),
                pkg.get("description", "").lower(),
                pkg.get("author", "").lower(),
            ]
            haystack.extend(t.lower() for t in pkg.get("tags", []))

            if any(query_norm in item for item in haystack):
                results.append(pkg)

        results.sort(key=lambda x: x["name"])
        return results

    # =========================================================================
    # 2. Publish Package
    # =========================================================================

    def publish(self, package_dir: str) -> dict[str, Any]:
        """
        Validates package manifest (synapse.toml), bundles files into a reproducible
        .synpkg archive, computes SHA-256 checksum, and registers into local registry/cache.

        Args:
            package_dir: Path to directory containing package and synapse.toml.

        Returns:
            Dict containing package publication details, archive path, and checksum.
        """
        abs_pkg_dir = os.path.abspath(package_dir)
        if not os.path.isdir(abs_pkg_dir):
            raise PackageError(f"Package directory does not exist: '{package_dir}'")

        manifest_path = os.path.join(abs_pkg_dir, self.MANIFEST_FILENAME)
        if not os.path.isfile(manifest_path):
            raise ManifestError(f"synapse.toml manifest not found in '{package_dir}'")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_content = f.read()

        manifest_data = TomlHelper.parse(manifest_content)
        pkg_sec = manifest_data.get("package")
        if not isinstance(pkg_sec, dict):
            raise ManifestError("Invalid manifest: missing [package] table in synapse.toml")

        pkg_name = str(pkg_sec.get("name", "")).strip()
        pkg_version = str(pkg_sec.get("version", "")).strip()
        description = str(pkg_sec.get("description", "")).strip()
        authors = pkg_sec.get("authors", [])
        entry = str(pkg_sec.get("entry", "main.syn")).strip()

        if not pkg_name:
            raise ManifestError("Invalid manifest: package.name must not be empty")
        if not re.match(r"^[a-zA-Z0-9_\-]+$", pkg_name):
            raise ManifestError(f"Invalid package name '{pkg_name}': must only contain letters, numbers, '-' or '_'")
        if not pkg_version:
            raise ManifestError("Invalid manifest: package.version must not be empty")

        # Create archive
        archive_name = f"{pkg_name}-{pkg_version}.synpkg"
        archive_path = os.path.join(self.registry_dir, archive_name)

        # Build tar.gz bundle
        with tarfile.open(archive_path, "w:gz") as tar:
            for root, dirs, files in sorted(os.walk(abs_pkg_dir)):
                # Filter out excluded directories
                dirs[:] = [
                    d for d in dirs
                    if d not in ("syn_modules", "__pycache__", ".git", ".synapse", ".pytest_cache", "venv")
                ]
                for file in sorted(files):
                    if file.endswith((".pyc", ".synpkg")) or file in (self.LOCKFILE_FILENAME,):
                        continue
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, abs_pkg_dir)
                    # Normalize tar member
                    tar.add(full_path, arcname=rel_path)

        # Compute SHA-256
        checksum = compute_file_sha256(archive_path)

        # Copy to cache_dir as well
        cache_archive_path = os.path.join(self.cache_dir, archive_name)
        shutil.copy2(archive_path, cache_archive_path)

        # Update index.json
        index_data = self._load_registry_index()
        index_data["packages"][pkg_name] = {
            "name": pkg_name,
            "version": pkg_version,
            "description": description,
            "authors": authors,
            "entry": entry,
            "archive": archive_name,
            "checksum": checksum,
            "size_bytes": os.path.getsize(archive_path),
            "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        self._save_registry_index(index_data)

        return {
            "status": "published",
            "name": pkg_name,
            "version": pkg_version,
            "archive_path": archive_path,
            "checksum": checksum,
            "size_bytes": os.path.getsize(archive_path),
            "description": description,
        }

    # =========================================================================
    # 3. Install Package with SHA-256 Checksum Verification
    # =========================================================================

    def install(
        self,
        name: str,
        version: Optional[str] = None,
        project_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Installs a package from registry or cache into syn_modules/, verifies its
        SHA-256 checksum, updates synapse.toml dependencies, and locks the dependency
        in synapse.lock.

        Args:
            name: Package name (e.g. 'synapse-nn', 'synapse-vision').
            version: Optional desired version.
            project_dir: Target project directory.

        Returns:
            Dict with installation results, installed directory, and verified checksum.
        """
        target_dir = os.path.abspath(project_dir or self.default_project_dir)
        os.makedirs(target_dir, exist_ok=True)

        # Ensure target project has a synapse.toml manifest
        manifest_path = os.path.join(target_dir, self.MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            pm = PackageManager()
            pm.init_project(target_dir)

        # Resolve package archive and metadata
        archive_path, resolved_version, expected_checksum = self._resolve_package_archive(name, version)

        # Verify SHA-256 checksum
        actual_checksum = compute_file_sha256(archive_path)
        if expected_checksum and actual_checksum.lower() != expected_checksum.lower():
            raise ChecksumMismatchError(
                f"Security check failed: SHA-256 checksum mismatch for package '{name}'.\n"
                f"Expected: {expected_checksum}\n"
                f"Computed: {actual_checksum}"
            )

        # Extract to syn_modules/<name>/
        modules_dir = os.path.join(target_dir, self.MODULES_DIRNAME)
        dest_dir = os.path.join(modules_dir, name)

        if os.path.exists(dest_dir):
            if os.path.isdir(dest_dir):
                shutil.rmtree(dest_dir)
            else:
                os.remove(dest_dir)

        os.makedirs(dest_dir, exist_ok=True)

        with tarfile.open(archive_path, "r:gz") as tar:
            # Safe extraction preventing path traversal (Zip/Tar Slip vulnerability)
            dest_canonical = os.path.realpath(dest_dir)
            for member in tar.getmembers():
                member_path = os.path.realpath(os.path.join(dest_dir, member.name))
                if not member_path.startswith(dest_canonical):
                    raise PackageError(f"Attempted path traversal in package archive: {member.name}")
                tar.extract(member, path=dest_dir)

        # Update synapse.toml dependencies
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = TomlHelper.parse(f.read())

        if "dependencies" not in manifest_data or not isinstance(manifest_data["dependencies"], dict):
            manifest_data["dependencies"] = {}

        manifest_data["dependencies"][name] = resolved_version
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(TomlHelper.dump(manifest_data))

        # Update synapse.lock with verified checksum
        lock_path = self._update_lockfile(target_dir, name, resolved_version, actual_checksum)

        return {
            "status": "installed",
            "name": name,
            "version": resolved_version,
            "install_dir": dest_dir,
            "checksum": actual_checksum,
            "lockfile": lock_path,
            "archive_path": archive_path
        }

    def _resolve_package_archive(
        self, name: str, version: Optional[str] = None
    ) -> tuple[str, str, Optional[str]]:
        """
        Finds or constructs the package archive (.synpkg) and its expected SHA-256 checksum.
        Returns: (archive_path, resolved_version, expected_checksum)
        """
        # 1. Check local registry storage index
        reg_index = self._load_registry_index()
        if name in reg_index.get("packages", {}):
            meta = reg_index["packages"][name]
            meta_ver = meta.get("version", "0.1.0")
            if version is None or version == meta_ver:
                archive_name = meta.get("archive", f"{name}-{meta_ver}.synpkg")
                archive_path = os.path.join(self.registry_dir, archive_name)
                if os.path.exists(archive_path):
                    return archive_path, meta_ver, meta.get("checksum")

        # 2. Check local cache directory for existing .synpkg archive
        if version:
            cached_candidate = os.path.join(self.cache_dir, f"{name}-{version}.synpkg")
            if os.path.exists(cached_candidate):
                return cached_candidate, version, compute_file_sha256(cached_candidate)

        # 3. Check built-in packages index
        if name in BUILTIN_PACKAGES:
            meta = BUILTIN_PACKAGES[name]
            builtin_ver = meta["version"]
            if version is not None and version != builtin_ver:
                raise PackageNotFoundError(
                    f"Version '{version}' not found for built-in package '{name}' (available: {builtin_ver})"
                )

            # Build archive in cache if not already present
            archive_path = os.path.join(self.cache_dir, f"{name}-{builtin_ver}.synpkg")
            if not os.path.exists(archive_path):
                self._bundle_builtin_package(meta, archive_path)

            checksum = compute_file_sha256(archive_path)
            return archive_path, builtin_ver, checksum

        # Not found
        raise PackageNotFoundError(f"Package '{name}' could not be found in registry or cache.")

    def _bundle_builtin_package(self, meta: dict[str, Any], output_path: str) -> None:
        """Bundles a built-in package definition into a .synpkg archive in cache."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_build_dir:
            for rel_path, content in meta["files"].items():
                full_path = os.path.join(tmp_build_dir, rel_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

            with tarfile.open(output_path, "w:gz") as tar:
                for root, _, files in sorted(os.walk(tmp_build_dir)):
                    for file in sorted(files):
                        full_file = os.path.join(root, file)
                        rel_name = os.path.relpath(full_file, tmp_build_dir)
                        tar.add(full_file, arcname=rel_name)

    def _update_lockfile(
        self, project_dir: str, pkg_name: str, version: str, checksum: str
    ) -> str:
        """Updates or generates synapse.lock with package version and verified checksum."""
        lock_path = os.path.join(project_dir, self.LOCKFILE_FILENAME)
        lock_data: dict[str, Any] = {
            "version": 1,
            "generated_by": "synapse pkg registry v1.5",
            "packages": {}
        }

        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "packages" in loaded:
                        lock_data = loaded
            except Exception:
                pass

        lock_data["packages"][pkg_name] = {
            "version": version,
            "installed": True,
            "checksum": checksum
        }

        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2, ensure_ascii=False)

        return lock_path


# =========================================================================
# CLI Handlers
# (The Master Agent will wire these directly into synapse/cli.py)
# =========================================================================

def handle_pkg_search(query: str = "", project_dir: Optional[str] = None) -> int:
    """CLI handler: synapse pkg search <query>"""
    try:
        reg = PackageRegistry(default_project_dir=project_dir)
        results = reg.search(query)
        if not results:
            print(f"No packages found matching '{query}'.")
            return 0

        print(f"Found {len(results)} package(s):")
        print(f"{'PACKAGE':<22} {'VERSION':<10} {'STATUS':<12} {'DESCRIPTION'}")
        print("-" * 75)
        for pkg in results:
            status = "[installed]" if pkg.get("installed") else "[available]"
            name = pkg.get("name", "")
            ver = f"v{pkg.get('version', '0.1.0')}"
            desc = pkg.get("description", "")
            print(f"{name:<22} {ver:<10} {status:<12} {desc}")
        return 0
    except Exception as e:
        print(f"Error searching packages: {e}", file=sys.stderr)
        return 1


def handle_pkg_install(
    package_name: str,
    version: Optional[str] = None,
    project_dir: Optional[str] = None,
) -> int:
    """CLI handler: synapse pkg install <package_name> [--version <ver>]"""
    try:
        reg = PackageRegistry(default_project_dir=project_dir)
        result = reg.install(package_name, version=version, project_dir=project_dir)
        print(f"[OK] Installed '{result['name']}' v{result['version']}")
        print(f"     Destination: {result['install_dir']}")
        print(f"     SHA-256:     {result['checksum']}")
        print(f"     Lockfile:    {result['lockfile']}")
        return 0
    except ChecksumMismatchError as e:
        print(f"[SECURITY ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[Error] Failed to install package '{package_name}': {e}", file=sys.stderr)
        return 1


def handle_pkg_publish(package_dir: str = ".") -> int:
    """CLI handler: synapse pkg publish [package_dir]"""
    try:
        reg = PackageRegistry()
        result = reg.publish(package_dir)
        print(f"[OK] Successfully published '{result['name']}' v{result['version']}")
        print(f"     Archive: {result['archive_path']}")
        print(f"     SHA-256: {result['checksum']}")
        print(f"     Size:    {result['size_bytes']} bytes")
        return 0
    except Exception as e:
        print(f"[Error] Failed to publish package from '{package_dir}': {e}", file=sys.stderr)
        return 1
