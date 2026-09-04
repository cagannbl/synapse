"""
Unit and integration tests for Synapse Native C-FFI Bridge and Package Registry.

Covers:
- C runtime loading on Windows (msvcrt) and Unix (libc)
- Type resolution and CFunction binding
- Mathematical & standard C function calls (abs, ceil, sqrt, strlen)
- Zero-copy Synapse Tensor buffer passing with C memory operations (memcpy)
- Package Registry search, offline built-in index, install with SHA-256 checksum verification
- Determinstic synapse.lock generation and updates
- Package publish workflow and manifest validation
- CLI handlers: handle_pkg_search, handle_pkg_install, handle_pkg_publish
"""
import ctypes
import json
import os
import shutil
import sys
import numpy as np
import pytest

from synapse.core.tensor import Tensor
from synapse.interop.c_ffi import (
    CDynamicLibrary,
    CFunction,
    load_library,
    get_tensor_pointer,
    resolve_type,
    pointer_to,
    SynapseFFIError,
    int32,
    int64,
    float32,
    float64,
    char_p,
    void_p,
    size_t,
)
from synapse.pkg.registry import (
    PackageRegistry,
    PackageError,
    PackageNotFoundError,
    ChecksumMismatchError,
    ManifestError,
    compute_file_sha256,
    compute_bytes_sha256,
    handle_pkg_search,
    handle_pkg_install,
    handle_pkg_publish,
    BUILTIN_PACKAGES,
)
from synapse.pkg.manager import TomlHelper


# =========================================================================
# 1. Native C-FFI Bridge Tests
# =========================================================================

def test_c_ffi_load_default_c_runtime():
    """Verifies that load_library() without arguments loads the OS C runtime."""
    lib = load_library()
    assert lib is not None
    assert lib._lib is not None

    if os.name == "nt":
        # On Windows, must bind to msvcrt
        assert "msvcrt" in str(lib).lower() or lib._lib == ctypes.cdll.msvcrt


def test_c_ffi_bind_basic_math_functions():
    """Tests binding and invoking standard C functions (abs, ceil, sqrt, sin, cos)."""
    lib = load_library()

    # abs: int -> int
    abs_fn = lib.bind("abs", [int32], int32)
    assert isinstance(abs_fn, CFunction)
    assert abs_fn(-42) == 42
    assert abs_fn(100) == 100
    assert abs_fn(0) == 0

    # ceil: double -> double
    ceil_fn = lib.bind("ceil", [float64], float64)
    assert ceil_fn(3.2) == 4.0
    assert ceil_fn(-1.7) == -1.0
    assert ceil_fn(5.0) == 5.0

    # sqrt: double -> double
    sqrt_fn = lib.bind("sqrt", [float64], float64)
    assert sqrt_fn(64.0) == 8.0
    assert abs(sqrt_fn(2.0) - 1.41421356) < 1e-6

    # sin and cos
    sin_fn = lib.bind("sin", [float64], float64)
    cos_fn = lib.bind("cos", [float64], float64)
    assert sin_fn(0.0) == 0.0
    assert cos_fn(0.0) == 1.0


def test_c_ffi_string_functions():
    """Tests string passing to C function (strlen) with char_p and size_t."""
    lib = load_library()
    strlen_fn = lib.bind("strlen", [char_p], size_t)

    assert strlen_fn("synapse") == 7
    assert strlen_fn("AI-Native") == 9
    assert strlen_fn("") == 0


def test_c_ffi_type_mapping_and_resolution():
    """Tests type mapping helpers and string resolution."""
    assert resolve_type("int32") == ctypes.c_int32
    assert resolve_type("int64") == ctypes.c_int64
    assert resolve_type("float32") == ctypes.c_float
    assert resolve_type("float64") == ctypes.c_double
    assert resolve_type("void_p") == ctypes.c_void_p
    assert resolve_type("char_p") == ctypes.c_char_p
    assert resolve_type("size_t") == ctypes.c_size_t
    assert resolve_type("void") is None
    assert resolve_type(None) is None

    # Pointer resolution
    double_ptr = resolve_type("float64*")
    assert double_ptr._type_ == ctypes.c_double

    int_ptr = pointer_to("int32")
    assert int_ptr._type_ == ctypes.c_int32

    # Invalid type
    with pytest.raises(SynapseFFIError):
        resolve_type("nonexistent_c_type_xyz")


def test_c_ffi_zero_copy_tensor_memcpy():
    """
    Validates zero-copy C interoperability using C memcpy on Synapse Tensor buffers.
    Data is copied directly between memory buffers at the C level without Python loops.
    """
    lib = load_library()
    memcpy_fn = lib.bind("memcpy", [void_p, void_p, size_t], void_p)

    # Create source and destination tensors
    src = Tensor([10.5, 20.25, 30.125, 40.0625])
    dst = Tensor([0.0, 0.0, 0.0, 0.0])

    assert not np.array_equal(src.data, dst.data)

    # 1. Zero-copy: Pass Tensor instances directly (automatic pointer extraction)
    num_bytes = src.size * 8  # float64 = 8 bytes each
    memcpy_fn(dst, src, num_bytes)

    assert np.allclose(dst.data, src.data)
    assert dst.data[0] == 10.5
    assert dst.data[3] == 40.0625

    # 2. Zero-copy: Pass explicit raw ctypes pointers
    src2 = Tensor([1.1, 2.2, 3.3])
    dst2 = Tensor([9.9, 9.9, 9.9])

    raw_src_ptr = src2.c_pointer()
    raw_dst_ptr = dst2.data.ctypes.data_as(void_p)

    memcpy_fn(raw_dst_ptr, raw_src_ptr, src2.size * 8)
    assert np.allclose(dst2.data, [1.1, 2.2, 3.3])


def test_c_ffi_tensor_data_ptr_and_helper():
    """Tests tensor data_ptr() and get_tensor_pointer()."""
    t = Tensor([1.0, 2.0, 3.0])
    assert isinstance(t.data_ptr(), int)
    assert t.data_ptr() == int(t.data.ctypes.data)

    ptr = get_tensor_pointer(t, void_p)
    assert ptr is not None

    np_arr = np.array([4.0, 5.0])
    ptr_np = get_tensor_pointer(np_arr, void_p)
    assert ptr_np is not None

    with pytest.raises(TypeError):
        get_tensor_pointer("not_a_tensor_or_array")


def test_c_ffi_symbol_not_found_error():
    """Verifies that attempting to bind a non-existent C symbol raises SynapseFFIError."""
    lib = load_library()
    with pytest.raises(SynapseFFIError) as exc_info:
        lib.bind("this_c_function_definitely_does_not_exist_xyz123", [], None)
    assert "Symbol" in str(exc_info.value)


# =========================================================================
# 2. Package Registry Tests
# =========================================================================

@pytest.fixture
def isolated_registry(tmp_path):
    """Creates a temporary isolated PackageRegistry instance."""
    cache_dir = str(tmp_path / "cache")
    registry_dir = str(tmp_path / "registry")
    return PackageRegistry(cache_dir=cache_dir, registry_dir=registry_dir)


def test_registry_search_builtin(isolated_registry):
    """Tests searching built-in packages."""
    # 1. Search all
    all_pkgs = isolated_registry.search("*")
    names = [p["name"] for p in all_pkgs]
    assert "synapse-nn" in names
    assert "synapse-vision" in names
    assert "synapse-nlp" in names
    assert "synapse-math" in names

    # 2. Search specific keyword
    vision_pkgs = isolated_registry.search("vision")
    assert len(vision_pkgs) == 1
    assert vision_pkgs[0]["name"] == "synapse-vision"

    # 3. Search by tag
    tag_pkgs = isolated_registry.search("transformers")
    assert any(p["name"] == "synapse-nlp" for p in tag_pkgs)

    # 4. Search no match
    none_pkgs = isolated_registry.search("nonexistent_package_query_xyz")
    assert len(none_pkgs) == 0


def test_registry_install_builtin_with_sha256_and_lockfile(tmp_path, isolated_registry):
    """
    Tests installing a built-in package:
    - Bundles and caches .synpkg archive
    - Verifies SHA-256 checksum
    - Extracts into syn_modules/
    - Updates synapse.toml dependencies
    - Writes deterministic synapse.lock with checksum
    """
    project_dir = str(tmp_path / "my_project")

    # Install synapse-nn
    result = isolated_registry.install("synapse-nn", project_dir=project_dir)

    assert result["status"] == "installed"
    assert result["name"] == "synapse-nn"
    assert result["version"] == "0.2.0"
    assert len(result["checksum"]) == 64  # valid SHA-256 hex string

    # Verify syn_modules extraction
    mod_dir = os.path.join(project_dir, "syn_modules", "synapse-nn")
    assert os.path.isdir(mod_dir)
    assert os.path.isfile(os.path.join(mod_dir, "index.syn"))
    assert os.path.isfile(os.path.join(mod_dir, "layers.syn"))
    assert os.path.isfile(os.path.join(mod_dir, "synapse.toml"))

    # Verify synapse.toml dependencies updated
    manifest_path = os.path.join(project_dir, "synapse.toml")
    assert os.path.isfile(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = TomlHelper.parse(f.read())
    assert manifest_data.get("dependencies", {}).get("synapse-nn") == "0.2.0"

    # Verify synapse.lock contains verified SHA-256 checksum
    lock_path = os.path.join(project_dir, "synapse.lock")
    assert os.path.isfile(lock_path)
    with open(lock_path, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    assert lock_data["version"] == 1
    assert "synapse-nn" in lock_data["packages"]
    pkg_entry = lock_data["packages"]["synapse-nn"]
    assert pkg_entry["version"] == "0.2.0"
    assert pkg_entry["installed"] is True
    assert pkg_entry["checksum"] == result["checksum"]


def test_registry_install_checksum_verification_failure(tmp_path, isolated_registry):
    """
    Tests that a corrupted package archive triggers ChecksumMismatchError.
    """
    project_dir = str(tmp_path / "project_corrupt")
    # Pre-install to generate archive in cache
    res = isolated_registry.install("synapse-math", project_dir=project_dir)
    archive_path = res["archive_path"]
    assert os.path.isfile(archive_path)

    # Publish a package record with an intentionally mismatched checksum
    tampered_index = isolated_registry._load_registry_index()
    tampered_index["packages"]["tampered-pkg"] = {
        "name": "tampered-pkg",
        "version": "1.0.0",
        "archive": os.path.basename(archive_path),
        "checksum": "0000000000000000000000000000000000000000000000000000000000000000",
    }
    isolated_registry._save_registry_index(tampered_index)
    # Copy archive to registry_dir
    shutil.copy2(archive_path, os.path.join(isolated_registry.registry_dir, os.path.basename(archive_path)))

    with pytest.raises(ChecksumMismatchError) as exc_info:
        isolated_registry.install("tampered-pkg", project_dir=project_dir)

    assert "checksum mismatch" in str(exc_info.value).lower()


def test_registry_publish_and_install_workflow(tmp_path, isolated_registry):
    """
    Tests publishing a custom user package and installing it in another project.
    """
    # 1. Create a custom package
    pkg_src_dir = str(tmp_path / "custom_nlp_addon")
    os.makedirs(pkg_src_dir)

    manifest_content = (
        '[package]\n'
        'name = "custom-nlp-addon"\n'
        'version = "1.0.0"\n'
        'description = "Custom sentiment and embedding addon"\n'
        'authors = ["Test Author <test@synapse.org>"]\n'
        'entry = "addon.syn"\n'
    )
    with open(os.path.join(pkg_src_dir, "synapse.toml"), "w", encoding="utf-8") as f:
        f.write(manifest_content)

    with open(os.path.join(pkg_src_dir, "addon.syn"), "w", encoding="utf-8") as f:
        f.write('fn analyze(text): return 0.95\n')

    # 2. Publish package
    pub_result = isolated_registry.publish(pkg_src_dir)
    assert pub_result["status"] == "published"
    assert pub_result["name"] == "custom-nlp-addon"
    assert pub_result["version"] == "1.0.0"
    assert os.path.isfile(pub_result["archive_path"])
    assert len(pub_result["checksum"]) == 64

    # 3. Search and find published package
    search_res = isolated_registry.search("custom")
    assert any(p["name"] == "custom-nlp-addon" for p in search_res)

    # 4. Install into consumer project
    consumer_proj = str(tmp_path / "consumer_app")
    inst_result = isolated_registry.install("custom-nlp-addon", project_dir=consumer_proj)

    assert inst_result["status"] == "installed"
    assert inst_result["checksum"] == pub_result["checksum"]

    # Verify extracted files
    installed_addon = os.path.join(consumer_proj, "syn_modules", "custom-nlp-addon", "addon.syn")
    assert os.path.isfile(installed_addon)
    with open(installed_addon, "r", encoding="utf-8") as f:
        assert "analyze" in f.read()


def test_registry_publish_validation_errors(tmp_path, isolated_registry):
    """Tests that publishing invalid package directories raises ManifestError or PackageError."""
    # Nonexistent directory
    with pytest.raises(PackageError):
        isolated_registry.publish(str(tmp_path / "does_not_exist"))

    # Missing synapse.toml
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir)
    with pytest.raises(ManifestError):
        isolated_registry.publish(empty_dir)

    # Invalid manifest (missing [package] table)
    bad_toml_dir = str(tmp_path / "bad_toml")
    os.makedirs(bad_toml_dir)
    with open(os.path.join(bad_toml_dir, "synapse.toml"), "w") as f:
        f.write("[other_table]\nkey = 'val'\n")
    with pytest.raises(ManifestError):
        isolated_registry.publish(bad_toml_dir)


def test_registry_install_nonexistent_package(tmp_path, isolated_registry):
    """Verifies that requesting an unknown package raises PackageNotFoundError."""
    proj = str(tmp_path / "proj")
    with pytest.raises(PackageNotFoundError):
        isolated_registry.install("ghost-package-unknown-xyz", project_dir=proj)


# =========================================================================
# 3. CLI Handlers Integration Tests
# =========================================================================

def test_cli_handlers(tmp_path, monkeypatch):
    """Tests CLI handler functions exposed for synapse/cli.py."""
    proj_dir = str(tmp_path / "cli_test_proj")
    os.makedirs(proj_dir, exist_ok=True)
    monkeypatch.chdir(proj_dir)

    # Test search handler
    search_code = handle_pkg_search("vision")
    assert search_code == 0

    # Test install handler
    install_code = handle_pkg_install("synapse-math", project_dir=proj_dir)
    assert install_code == 0

    # Verify installed
    mod_path = os.path.join(proj_dir, "syn_modules", "synapse-math")
    assert os.path.isdir(mod_path)

    # Test publish handler
    pub_pkg = str(tmp_path / "cli_pkg")
    os.makedirs(pub_pkg)
    with open(os.path.join(pub_pkg, "synapse.toml"), "w") as f:
        f.write('[package]\nname = "cli-test-pkg"\nversion = "0.1.0"\n')
    with open(os.path.join(pub_pkg, "main.syn"), "w") as f:
        f.write('fn test(): return true\n')

    pub_code = handle_pkg_publish(pub_pkg)
    assert pub_code == 0
