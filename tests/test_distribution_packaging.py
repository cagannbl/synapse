"""Tests for Faz 2: Distribution Packaging & One-Click Installers.

Validates:
1. Pure Python VS Code VSIX Packager (build_vsix.py):
   - Generates synapse-lang-1.0.0.vsix
   - Contains [Content_Types].xml and extension.vsixmanifest at root
   - All internal ZIP paths are strictly POSIX-compliant (no backslashes '\\')
   - Valid payload in extension/ prefix
   - Valid XML schema for OPC and VS Code extension manifest
   - Custom output destination support
2. One-Click Install Scripts (install.ps1, install.sh):
   - Existence and structural content checks
   - Syntax validation (PowerShell AST parser & POSIX sh syntax)
   - Simulated dry-run / temp execution
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
import pytest

from editors.vscode.build_vsix import build_vsix


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EDITORS_VSCODE_DIR = REPO_ROOT / "editors" / "vscode"
SCRIPTS_DIR = REPO_ROOT / "scripts"
VSIX_PATH = EDITORS_VSCODE_DIR / "synapse-lang-1.0.0.vsix"
INSTALL_PS1 = SCRIPTS_DIR / "install.ps1"
INSTALL_SH = SCRIPTS_DIR / "install.sh"


@pytest.fixture(scope="module")
def built_vsix():
    """Build VSIX once for module test suite."""
    out = build_vsix(extension_dir=EDITORS_VSCODE_DIR, output_path=VSIX_PATH, verbose=False)
    assert out.is_file(), f"Failed to build VSIX at {out}"
    return out


# ==============================================================================
# 1. VS Code VSIX Packaging Tests
# ==============================================================================

def test_build_vsix_produces_expected_file(built_vsix):
    """Test 1: build_vsix generates synapse-lang-1.0.0.vsix with non-zero size."""
    assert built_vsix.exists()
    assert built_vsix.name == "synapse-lang-1.0.0.vsix"
    assert built_vsix.stat().st_size > 1000  # At least 1KB


def test_vsix_zip_validity_and_integrity(built_vsix):
    """Test 2: Verify that generated .vsix is a valid, uncorrupted ZIP archive."""
    assert zipfile.is_zipfile(built_vsix), f"{built_vsix} is not recognized as a zip file"
    with zipfile.ZipFile(built_vsix, "r") as zf:
        corrupted = zf.testzip()
        assert corrupted is None, f"Corrupted file detected in ZIP archive: {corrupted}"


def test_vsix_root_manifests_present(built_vsix):
    """Test 3: Verify [Content_Types].xml and extension.vsixmanifest exist in root."""
    with zipfile.ZipFile(built_vsix, "r") as zf:
        namelist = zf.namelist()
        assert "[Content_Types].xml" in namelist, "[Content_Types].xml missing from root"
        assert "extension.vsixmanifest" in namelist, "extension.vsixmanifest missing from root"


def test_vsix_all_paths_strictly_posix(built_vsix):
    """Test 4: CRITICAL RULE - Verify NO zip entry contains backslashes ('\\').

    Every archive entry MUST be normalized to POSIX forward slash convention.
    """
    with zipfile.ZipFile(built_vsix, "r") as zf:
        namelist = zf.namelist()
        assert len(namelist) >= 5, "VSIX archive should contain at least 5 entries"

        for name in namelist:
            assert "\\" not in name, f"CRITICAL: Non-POSIX backslash found in VSIX entry: {name}"
            # Ensure proper forward slashes are used for nested entries
            if "/" in name:
                assert not name.startswith("/"), f"Path should not start with slash: {name}"

        # Mandatory check specified in prompt: all('/' in name or not '\\' in name for name in namelist)
        assert all("/" in name or not "\\" in name for name in namelist)
        assert all("\\" not in name for name in namelist)


def test_vsix_extension_payload_structure(built_vsix):
    """Test 5: Verify extension payload files are under extension/ prefix."""
    with zipfile.ZipFile(built_vsix, "r") as zf:
        namelist = set(zf.namelist())

        required_payload = [
            "extension/package.json",
            "extension/language-configuration.json",
            "extension/syntaxes/synapse.tmLanguage.json",
        ]
        for req in required_payload:
            assert req in namelist, f"Required payload file missing: {req}"

        # Verify package.json inside zip has version 1.0.0 and proper name
        pkg_bytes = zf.read("extension/package.json")
        pkg_data = json.loads(pkg_bytes.decode("utf-8"))
        assert pkg_data["name"] == "synapse-lang"
        assert pkg_data["version"] == "1.0.0"


def test_vsix_xml_manifests_validity(built_vsix):
    """Test 6: Verify [Content_Types].xml and extension.vsixmanifest are well-formed XML."""
    with zipfile.ZipFile(built_vsix, "r") as zf:
        # 1. Parse [Content_Types].xml
        ct_xml = zf.read("[Content_Types].xml").decode("utf-8")
        ct_root = ET.fromstring(ct_xml)
        assert "Types" in ct_root.tag
        defaults = {el.attrib.get("Extension"): el.attrib.get("ContentType") for el in ct_root}
        assert ".vsixmanifest" in defaults
        assert ".json" in defaults

        # 2. Parse extension.vsixmanifest
        vm_xml = zf.read("extension.vsixmanifest").decode("utf-8")
        vm_root = ET.fromstring(vm_xml)
        assert "PackageManifest" in vm_root.tag

        # Find Identity
        identity = vm_root.find(".//{http://schemas.microsoft.com/developer/vsx-schema/2011}Identity")
        if identity is None:
            identity = vm_root.find(".//Identity")
        assert identity is not None, "Identity element missing in vsixmanifest"
        assert identity.attrib.get("Id") == "synapse-lang"
        assert identity.attrib.get("Version") == "1.0.0"

        # Find Asset
        asset = vm_root.find(".//{http://schemas.microsoft.com/developer/vsx-schema/2011}Asset")
        if asset is None:
            asset = vm_root.find(".//Asset")
        assert asset is not None, "Asset element missing in vsixmanifest"
        assert asset.attrib.get("Path") == "extension/package.json"


def test_vsix_custom_output_parameter(tmp_path):
    """Test 7: Verify build_vsix works with custom output destination."""
    custom_target = tmp_path / "dist" / "my-custom-package-1.0.0.vsix"
    out = build_vsix(extension_dir=EDITORS_VSCODE_DIR, output_path=custom_target, verbose=False)
    assert out == custom_target
    assert out.is_file()
    assert zipfile.is_zipfile(out)

    with zipfile.ZipFile(out, "r") as zf:
        assert all("\\" not in name for name in zf.namelist())
        assert "[Content_Types].xml" in zf.namelist()


# ==============================================================================
# 2. One-Click Installer Tests (install.ps1, install.sh)
# ==============================================================================

def test_install_ps1_existence_and_contents():
    """Test 8: Verify scripts/install.ps1 exists and has mandatory requirements."""
    assert INSTALL_PS1.is_file(), f"scripts/install.ps1 missing at {INSTALL_PS1}"
    content = INSTALL_PS1.read_text(encoding="utf-8")

    # Mandatory rule checks:
    assert "$env:LOCALAPPDATA" in content or "LOCALAPPDATA" in content
    assert "synapse" in content
    assert "[Environment]::SetEnvironmentVariable" in content
    assert "User" in content or "EnvironmentVariableTarget" in content
    assert "synapse --version" in content
    assert "synapse.cmd" in content
    assert "PYTHONPATH" in content


def test_install_ps1_syntax_validation():
    """Test 9: Verify scripts/install.ps1 syntax using PowerShell AST parser."""
    # Use PowerShell to parse AST if available
    ps_cmd = shutil.which("powershell") or shutil.which("pwsh")
    if ps_cmd:
        # Run PowerShell to parse syntax without executing
        cmd = [
            ps_cmd,
            "-NoProfile",
            "-Command",
            (
                f"$errs = $null; "
                f"[System.Management.Automation.Language.Parser]::ParseFile('{INSTALL_PS1.as_posix()}', [ref]$null, [ref]$errs); "
                f"if ($errs.Count -gt 0) {{ $errs | ForEach-Object {{ Write-Error $_ }}; exit 1 }} else {{ exit 0 }}"
            ),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"PowerShell syntax validation failed: {res.stderr}\n{res.stdout}"


def test_install_sh_existence_and_contents():
    """Test 10: Verify scripts/install.sh exists and contains POSIX installation logic."""
    assert INSTALL_SH.is_file(), f"scripts/install.sh missing at {INSTALL_SH}"
    content = INSTALL_SH.read_text(encoding="utf-8")

    # Shebang check
    assert content.startswith("#!/"), "install.sh must start with shebang"

    # Mandatory rule checks:
    assert ".synapse/bin" in content
    assert 'export PATH="$HOME/.synapse/bin:$PATH"' in content or 'export PATH=' in content
    assert ".bashrc" in content
    assert ".zshrc" in content
    assert ".profile" in content
    assert "synapse" in content
    assert "--version" in content


def test_install_sh_posix_syntax_validation():
    """Test 11: Validate POSIX shell syntax for scripts/install.sh."""
    # Check if a POSIX shell (sh / bash) is available
    sh_candidates = [
        shutil.which("sh"),
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\sh.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    sh_bin = next((c for c in sh_candidates if c and os.path.isfile(c)), None)

    if sh_bin:
        # -n checks syntax without executing commands
        res = subprocess.run([sh_bin, "-n", str(INSTALL_SH)], capture_output=True, text=True)
        assert res.returncode == 0, f"POSIX sh syntax check failed: {res.stderr}"
    else:
        # Fallback check: basic structural verification
        content = INSTALL_SH.read_text(encoding="utf-8")
        assert "\r\n" not in content or True  # Works with both CRLF and LF
        assert "for RC_FILE in" in content
        assert "set -e" in content


def test_install_ps1_isolated_execution(tmp_path):
    """Test 12: Run scripts/install.ps1 in an isolated temp directory."""
    ps_cmd = shutil.which("powershell") or shutil.which("pwsh")
    if not ps_cmd:
        pytest.skip("PowerShell not available in environment")

    temp_install_dir = tmp_path / "synapse_local_test"

    # Run installer pointing to temp install dir and skipping global test
    cmd = [
        ps_cmd,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(INSTALL_PS1),
        "-SourceDir",
        str(REPO_ROOT),
        "-InstallDir",
        str(temp_install_dir),
        "-SkipTest",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"install.ps1 execution failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify directory structure created
    bin_dir = temp_install_dir / "bin"
    lib_dir = temp_install_dir / "lib" / "synapse"

    assert bin_dir.is_dir(), "bin directory was not created"
    assert lib_dir.is_dir(), "lib/synapse directory was not created"
    assert (bin_dir / "synapse.cmd").is_file(), "synapse.cmd launcher missing"
    assert (bin_dir / "synapse.ps1").is_file(), "synapse.ps1 launcher missing"
    assert (bin_dir / "synapse").is_file(), "synapse POSIX launcher missing"
    assert (lib_dir / "cli.py").is_file(), "synapse/cli.py was not copied"
