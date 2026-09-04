import os
import pytest
from synapse.pkg.manager import PackageManager, TomlHelper


def test_toml_helper_roundtrip():
    data = {
        "package": {
            "name": "super_net",
            "version": "1.2.3",
            "active": True,
            "stars": 42
        },
        "dependencies": {
            "synapse-math": "0.1.0"
        }
    }
    dumped = TomlHelper.dump(data)
    assert 'name = "super_net"' in dumped
    assert "active = true" in dumped
    assert "stars = 42" in dumped
    assert "[dependencies]" in dumped

    parsed = TomlHelper.parse(dumped)
    assert parsed["package"]["name"] == "super_net"
    assert parsed["package"]["version"] == "1.2.3"
    assert parsed["package"]["active"] is True
    assert parsed["package"]["stars"] == 42
    assert parsed["dependencies"]["synapse-math"] == "0.1.0"


def test_package_manager_project_lifecycle(tmp_path):
    pm = PackageManager()
    proj_dir = str(tmp_path / "my_synapse_project")

    # 1. Initialize project
    manifest_path = pm.init_project(proj_dir, name="ai_project")
    assert os.path.isfile(manifest_path)
    assert os.path.isdir(os.path.join(proj_dir, "syn_modules"))
    assert os.path.isfile(os.path.join(proj_dir, "main.syn"))

    # 2. Read manifest
    manifest = pm.read_manifest(proj_dir)
    assert manifest["package"]["name"] == "ai_project"
    assert manifest["package"]["version"] == "0.1.0"
    assert manifest["package"]["entry"] == "main.syn"

    # 3. Install local/mock package
    success = pm.install_package(proj_dir, "neural-vision")
    assert success is True

    # Check manifest updated
    updated_manifest = pm.read_manifest(proj_dir)
    assert "neural-vision" in updated_manifest["dependencies"]

    # Check syn_modules directory
    mod_path = os.path.join(proj_dir, "syn_modules", "neural-vision")
    assert os.path.isdir(mod_path)
    assert os.path.isfile(os.path.join(mod_path, "index.syn"))

    # 4. List packages
    pkgs = pm.list_packages(proj_dir)
    assert len(pkgs) >= 1
    names = [p["name"] for p in pkgs]
    assert "neural-vision" in names
    item = next(p for p in pkgs if p["name"] == "neural-vision")
    assert item["installed"] is True

    # 5. Uninstall package
    rm_success = pm.uninstall_package(proj_dir, "neural-vision")
    assert rm_success is True

    post_rm_manifest = pm.read_manifest(proj_dir)
    assert "neural-vision" not in post_rm_manifest.get("dependencies", {})
    assert not os.path.exists(mod_path)


def test_package_manager_missing_manifest(tmp_path):
    pm = PackageManager()
    empty_dir = str(tmp_path / "empty")
    os.makedirs(empty_dir, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        pm.read_manifest(empty_dir)
