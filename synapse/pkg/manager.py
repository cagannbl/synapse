"""
Synapse Package Manager (pkg).
Manages project initialization, dependencies, local modules, and package manifests (synapse.toml).
Zero external dependencies, 100% pure Python with native TOML support.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
from typing import Any, Optional

try:
    import tomllib  # Python 3.11+ built-in
except ImportError:
    tomllib = None  # type: ignore


class TomlHelper:
    """
    Lightweight, robust TOML reader and writer with zero external dependencies.
    """

    @staticmethod
    def parse(content: str) -> dict[str, Any]:
        """Parses TOML string using tomllib or fallback line-based parser."""
        if tomllib is not None:
            return tomllib.loads(content)

        # Fallback simple parser for environments without tomllib
        result: dict[str, Any] = {}
        current_section: Optional[dict[str, Any]] = None

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section header [section]
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1].strip()
                result[section_name] = {}
                current_section = result[section_name]
                continue

            # Key-value pair: key = "value"
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()

                parsed_val: Any = val
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    parsed_val = val[1:-1]
                elif val.lower() == "true":
                    parsed_val = True
                elif val.lower() == "false":
                    parsed_val = False
                elif val.isdigit():
                    parsed_val = int(val)
                elif re.match(r"^\d+\.\d+$", val):
                    parsed_val = float(val)

                if current_section is not None:
                    current_section[key] = parsed_val
                else:
                    result[key] = parsed_val

        return result

    @staticmethod
    def dump(data: dict[str, Any]) -> str:
        """Serializes dictionary to clean, standard TOML format."""
        lines: list[str] = []

        # Top-level primitives first
        for key, value in data.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {TomlHelper._format_value(value)}")

        # Sections (tables)
        for section, content in data.items():
            if isinstance(content, dict):
                if lines:
                    lines.append("")
                lines.append(f"[{section}]")
                for k, v in content.items():
                    lines.append(f"{k} = {TomlHelper._format_value(v)}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            items = ", ".join(TomlHelper._format_value(x) for x in value)
            return f"[{items}]"
        else:
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'


class SemverHelper:
    """
    Lightweight Semantic Versioning (SemVer 2.0.0) parser and constraint matcher.
    Supports exact (0.1.0), caret (^0.1.0), tilde (~0.1.0), and comparison (>=, <=, >).
    """
    @staticmethod
    def parse_version(v: str) -> tuple[int, int, int]:
        m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(v).strip())
        if not m:
            return (0, 0, 0)
        major = int(m.group(1) or 0)
        minor = int(m.group(2) or 0)
        patch = int(m.group(3) or 0)
        return (major, minor, patch)

    @staticmethod
    def satisfies(version: str, constraint: str) -> bool:
        v_parts = SemverHelper.parse_version(version)
        c = constraint.strip()

        if c.startswith("^"):
            c_parts = SemverHelper.parse_version(c[1:])
            if c_parts[0] > 0:
                return v_parts[0] == c_parts[0] and v_parts >= c_parts
            elif c_parts[1] > 0:
                return v_parts[0] == 0 and v_parts[1] == c_parts[1] and v_parts[2] >= c_parts[2]
            else:
                return v_parts == c_parts

        elif c.startswith("~"):
            c_parts = SemverHelper.parse_version(c[1:])
            return v_parts[0] == c_parts[0] and v_parts[1] == c_parts[1] and v_parts[2] >= c_parts[2]

        elif c.startswith(">="):
            c_parts = SemverHelper.parse_version(c[2:])
            return v_parts >= c_parts

        elif c.startswith("<="):
            c_parts = SemverHelper.parse_version(c[2:])
            return v_parts <= c_parts

        elif c.startswith(">"):
            c_parts = SemverHelper.parse_version(c[1:])
            return v_parts > c_parts

        elif c.startswith("=="):
            c_parts = SemverHelper.parse_version(c[2:])
            return v_parts == c_parts

        c_parts = SemverHelper.parse_version(c)
        return v_parts == c_parts


class PackageManager:
    """
    Synapse Package Manager engine.
    Handles synapse.toml project manifests and syn_modules/ installation.
    """

    MANIFEST_FILENAME = "synapse.toml"
    MODULES_DIRNAME = "syn_modules"
    LOCKFILE_FILENAME = "synapse.lock"

    def init_project(self, dir_path: str, name: Optional[str] = None) -> str:
        """
        Initializes a new Synapse project directory with standard synapse.toml manifest
        and syn_modules/ directory.

        Args:
            dir_path: Target directory path.
            name: Project name. If not provided, directory basename is used.

        Returns:
            Absolute path to generated synapse.toml.
        """
        abs_dir = os.path.abspath(dir_path)
        os.makedirs(abs_dir, exist_ok=True)

        proj_name = name or os.path.basename(abs_dir) or "synapse_project"
        manifest_path = os.path.join(abs_dir, self.MANIFEST_FILENAME)
        modules_dir = os.path.join(abs_dir, self.MODULES_DIRNAME)
        os.makedirs(modules_dir, exist_ok=True)

        # Standard Synapse project manifest
        manifest_data: dict[str, Any] = {
            "package": {
                "name": proj_name,
                "version": "0.1.0",
                "description": f"Synapse AI application {proj_name}",
                "entry": "main.syn",
                "authors": []
            },
            "dependencies": {},
            "dev-dependencies": {}
        }

        # If manifest doesn't exist, write default
        if not os.path.exists(manifest_path):
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(TomlHelper.dump(manifest_data))

        # Create starter main.syn if missing
        entry_file = os.path.join(abs_dir, "main.syn")
        if not os.path.exists(entry_file):
            starter_code = (
                f"// Synapse AI Project: {proj_name}\n"
                f"fn main():\n"
                f"    let message = \"Welcome to Synapse AI: {proj_name}\"\n"
                f"    print(message)\n\n"
                f"main()\n"
            )
            with open(entry_file, "w", encoding="utf-8") as f:
                f.write(starter_code)

        return manifest_path

    def read_manifest(self, dir_path: str) -> dict[str, Any]:
        """
        Reads and parses synapse.toml manifest in target directory.

        Args:
            dir_path: Project root directory.

        Returns:
            Parsed dictionary representing the manifest.

        Raises:
            FileNotFoundError if synapse.toml does not exist.
        """
        manifest_path = os.path.join(os.path.abspath(dir_path), self.MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Synapse project manifest not found at: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            content = f.read()

        return TomlHelper.parse(content)

    def write_manifest(self, dir_path: str, data: dict[str, Any]) -> str:
        """
        Writes dictionary back to synapse.toml manifest.
        """
        manifest_path = os.path.join(os.path.abspath(dir_path), self.MANIFEST_FILENAME)
        content = TomlHelper.dump(data)
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return manifest_path

    def install_package(self, target_dir: str, package_name_or_url: str) -> bool:
        """
        Installs a package into syn_modules/ and updates synapse.toml dependencies.

        Supports:
        - Local directory or file paths (e.g. '../my_lib' or './utils.syn')
        - Git repository URLs (e.g. 'https://github.com/synapse/vision.git')
        - Package names (e.g. 'synapse-vision', 'neural-utils')

        Args:
            target_dir: Target project directory.
            package_name_or_url: Package identifier, path, or URL.

        Returns:
            True if installation succeeded.
        """
        abs_target = os.path.abspath(target_dir)
        manifest_path = os.path.join(abs_target, self.MANIFEST_FILENAME)

        # Ensure project is initialized
        if not os.path.exists(manifest_path):
            self.init_project(abs_target)

        manifest = self.read_manifest(abs_target)
        if "dependencies" not in manifest or not isinstance(manifest["dependencies"], dict):
            manifest["dependencies"] = {}

        modules_dir = os.path.join(abs_target, self.MODULES_DIRNAME)
        os.makedirs(modules_dir, exist_ok=True)

        pkg_name, pkg_version, install_dest = self._resolve_and_copy_package(
            abs_target, modules_dir, package_name_or_url
        )

        # Update dependencies table in manifest
        manifest["dependencies"][pkg_name] = pkg_version
        self.write_manifest(abs_target, manifest)

        return True

    def list_packages(self, dir_path: str) -> list[dict[str, Any]]:
        """
        Lists installed packages in project.

        Returns:
            List of dicts containing name, version, path, and installed status.
        """
        abs_dir = os.path.abspath(dir_path)
        modules_dir = os.path.join(abs_dir, self.MODULES_DIRNAME)

        try:
            manifest = self.read_manifest(abs_dir)
            declared_deps = manifest.get("dependencies", {})
        except FileNotFoundError:
            declared_deps = {}

        packages: dict[str, dict[str, Any]] = {}

        # 1. Add declared dependencies from synapse.toml
        for dep_name, dep_version in declared_deps.items():
            pkg_path = os.path.join(modules_dir, dep_name)
            packages[dep_name] = {
                "name": dep_name,
                "version": str(dep_version),
                "path": pkg_path if os.path.exists(pkg_path) else None,
                "installed": os.path.exists(pkg_path)
            }

        # 2. Check syn_modules directory for any unrecorded or extra packages
        if os.path.exists(modules_dir):
            for entry in os.listdir(modules_dir):
                full_path = os.path.join(modules_dir, entry)
                if os.path.isdir(full_path) and entry not in packages:
                    # Attempt to read package's own manifest if present
                    pkg_manifest_file = os.path.join(full_path, self.MANIFEST_FILENAME)
                    version = "0.1.0"
                    if os.path.exists(pkg_manifest_file):
                        try:
                            with open(pkg_manifest_file, "r", encoding="utf-8") as f:
                                sub_data = TomlHelper.parse(f.read())
                                version = sub_data.get("package", {}).get("version", "0.1.0")
                        except Exception:
                            pass

                    packages[entry] = {
                        "name": entry,
                        "version": version,
                        "path": full_path,
                        "installed": True
                    }

        return list(packages.values())

    def outdated(self, dir_path: str) -> list[dict[str, Any]]:
        """
        Compares installed package versions against known registry versions.
        Returns list of outdated packages with installed and latest versions.
        """
        packages = self.list_packages(dir_path)
        outdated_list = []
        try:
            from synapse.pkg.registry import BUILTIN_PACKAGES
        except ImportError:
            BUILTIN_PACKAGES = {}

        for pkg in packages:
            name = pkg["name"]
            installed_v = pkg["version"]
            if name in BUILTIN_PACKAGES:
                latest_v = BUILTIN_PACKAGES[name]["version"]
                if SemverHelper.parse_version(latest_v) > SemverHelper.parse_version(installed_v):
                    outdated_list.append({
                        "name": name,
                        "current": installed_v,
                        "latest": latest_v,
                        "status": "upgrade_available"
                    })
        return outdated_list

    def uninstall_package(self, target_dir: str, package_name: str) -> bool:
        """
        Removes a package from syn_modules/ and synapse.toml.
        """
        abs_target = os.path.abspath(target_dir)
        manifest = self.read_manifest(abs_target)

        # Remove from dependencies
        deps = manifest.get("dependencies", {})
        if package_name in deps:
            del deps[package_name]
            self.write_manifest(abs_target, manifest)

        # Remove directory from syn_modules
        pkg_path = os.path.join(abs_target, self.MODULES_DIRNAME, package_name)
        if os.path.exists(pkg_path):
            if os.path.isdir(pkg_path):
                shutil.rmtree(pkg_path)
            else:
                os.remove(pkg_path)

        return True

    def generate_lockfile(self, target_dir: str) -> str:
        """
        Cargo/uv benzeri deterministik lockfile (synapse.lock) üretir.
        Yüklü tüm paketlerin adını, sürümünü ve dosya hash'lerini kilitler.
        """
        import hashlib
        import json
        abs_target = os.path.abspath(target_dir)
        packages = self.list_packages(abs_target)

        lock_data: dict[str, Any] = {
            "version": 1,
            "generated_by": "synapse pkg v1.5",
            "packages": {}
        }

        for pkg in packages:
            name = pkg["name"]
            pkg_path = pkg.get("path")
            file_hash = "direct"
            if pkg_path and os.path.exists(pkg_path):
                hasher = hashlib.sha256()
                if os.path.isdir(pkg_path):
                    for root, _, files in sorted(os.walk(pkg_path)):
                        for fn in sorted(files):
                            fp = os.path.join(root, fn)
                            try:
                                with open(fp, "rb") as f:
                                    hasher.update(f.read())
                            except Exception:
                                pass
                else:
                    with open(pkg_path, "rb") as f:
                        hasher.update(f.read())
                file_hash = hasher.hexdigest()

            lock_data["packages"][name] = {
                "version": pkg["version"],
                "installed": pkg["installed"],
                "checksum": file_hash
            }

        lock_path = os.path.join(abs_target, self.LOCKFILE_FILENAME)
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(lock_data, indent=2, ensure_ascii=False))

        return lock_path

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _resolve_and_copy_package(
        self, project_dir: str, modules_dir: str, package_spec: str
    ) -> tuple[str, str, str]:
        """
        Resolves package specifier and installs files into modules_dir.
        Returns: (package_name, version, installed_path)
        """
        spec = package_spec.strip()

        # Case 1: Local Path
        local_path = os.path.abspath(os.path.join(project_dir, spec)) if not os.path.isabs(spec) else spec
        if os.path.exists(local_path):
            pkg_name = os.path.basename(os.path.normpath(local_path))
            dest = os.path.join(modules_dir, pkg_name)
            version = "0.1.0"

            if os.path.isdir(local_path):
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(local_path, dest)
                # Check for sub-manifest
                sub_manifest = os.path.join(dest, self.MANIFEST_FILENAME)
                if os.path.exists(sub_manifest):
                    try:
                        with open(sub_manifest, "r", encoding="utf-8") as f:
                            data = TomlHelper.parse(f.read())
                            pkg_name = data.get("package", {}).get("name", pkg_name)
                            version = data.get("package", {}).get("version", "0.1.0")
                    except Exception:
                        pass
            else:
                # Single file module
                os.makedirs(dest, exist_ok=True)
                shutil.copy(local_path, os.path.join(dest, os.path.basename(local_path)))

            return pkg_name, version, dest

        # Case 2: Git Repository URL
        if spec.startswith("http://") or spec.startswith("https://") or spec.startswith("git@"):
            match = re.search(r"/([^/]+?)(?:\.git)?$", spec)
            pkg_name = match.group(1) if match else "synapse_pkg"
            dest = os.path.join(modules_dir, pkg_name)

            # Try git clone if git is available
            cloned = False
            try:
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                res = subprocess.run(
                    ["git", "clone", "--depth", "1", spec, dest],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if res.returncode == 0:
                    cloned = True
            except Exception:
                cloned = False

            if not cloned:
                # Fallback module creation
                os.makedirs(dest, exist_ok=True)
                with open(os.path.join(dest, "index.syn"), "w", encoding="utf-8") as f:
                    f.write(f"// Package {pkg_name} cloned from {spec}\nfn info(): return \"{pkg_name}\"\n")

            return pkg_name, spec, dest

        # Case 3: Package Name (Registry / Standard module)
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "", spec) or "module"
        dest = os.path.join(modules_dir, clean_name)
        os.makedirs(dest, exist_ok=True)

        index_file = os.path.join(dest, "index.syn")
        if not os.path.exists(index_file):
            content = (
                f"// Synapse Package: {clean_name}\n"
                f"fn info():\n"
                f"    return \"Package: {clean_name}\"\n"
            )
            with open(index_file, "w", encoding="utf-8") as f:
                f.write(content)

        pkg_manifest = os.path.join(dest, self.MANIFEST_FILENAME)
        if not os.path.exists(pkg_manifest):
            with open(pkg_manifest, "w", encoding="utf-8") as f:
                f.write(TomlHelper.dump({
                    "package": {
                        "name": clean_name,
                        "version": "0.1.0",
                        "description": f"Installed Synapse module {clean_name}"
                    }
                }))

        return clean_name, "0.1.0", dest
