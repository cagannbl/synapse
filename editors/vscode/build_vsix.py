#!/usr/bin/env python3
"""Pure Python VS Code VSIX Packager (OpenVSX / Microsoft VSIX Spec Compliant).

Builds a standards-compliant .vsix archive without requiring Node.js, npm, or vsce.
Enforces POSIX path normalization (strict forward slashes) across all ZIP entries.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Optional, Set


CONTENT_TYPE_MAPPINGS: Dict[str, str] = {
    ".vsixmanifest": "text/xml",
    ".xml": "text/xml",
    ".json": "application/json",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".js": "application/javascript",
    ".ts": "application/typescript",
    ".css": "text/css",
    ".html": "text/html",
}

IGNORED_PATTERNS: Set[str] = {
    ".git",
    ".gitignore",
    ".vscode-test",
    ".DS_Store",
    "Thumbs.db",
    "__pycache__",
    "build_vsix.py",
}


def generate_content_types_xml(extensions: Set[str]) -> str:
    """Generate [Content_Types].xml following Open Packaging Conventions."""
    types_el = ET.Element(
        "Types",
        xmlns="http://schemas.openxmlformats.org/package/2006/content-types",
    )

    # Standard default entries
    defaults = dict(CONTENT_TYPE_MAPPINGS)
    for ext in sorted(extensions):
        if ext and ext not in defaults:
            defaults[ext] = "application/octet-stream"

    for ext, content_type in sorted(defaults.items()):
        if not ext.startswith("."):
            ext = f".{ext}"
        ET.SubElement(
            types_el,
            "Default",
            Extension=ext,
            ContentType=content_type,
        )

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(types_el, encoding="utf-8").decode("utf-8")
    )


def generate_vsix_manifest(package_json: dict, assets: List[Dict[str, str]]) -> str:
    """Generate extension.vsixmanifest following VS Code VSIX schema."""
    name = package_json.get("name", "synapse-lang")
    version = package_json.get("version", "1.0.0")
    publisher = package_json.get("publisher", "synapse")
    display_name = package_json.get("displayName", name)
    description = package_json.get("description", "")
    engines_vscode = package_json.get("engines", {}).get("vscode", "^1.75.0")
    categories = ",".join(package_json.get("categories", ["Programming Languages"]))

    manifest_el = ET.Element(
        "PackageManifest",
        Version="2.0.0",
        xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011",
        attrib={"xmlns:d": "http://schemas.microsoft.com/developer/vsx-schema-design/2011"},
    )

    metadata_el = ET.SubElement(manifest_el, "Metadata")
    ET.SubElement(
        metadata_el,
        "Identity",
        Id=name,
        Version=version,
        Publisher=publisher,
        Language="en-US",
    )
    disp_el = ET.SubElement(metadata_el, "DisplayName")
    disp_el.text = display_name
    desc_el = ET.SubElement(metadata_el, "Description", attrib={"xml:space": "preserve"})
    desc_el.text = description

    if categories:
        cat_el = ET.SubElement(metadata_el, "Categories")
        cat_el.text = categories

    flags_el = ET.SubElement(metadata_el, "GalleryFlags")
    flags_el.text = "Public"

    properties_el = ET.SubElement(metadata_el, "Properties")
    ET.SubElement(properties_el, "Property", Id="Microsoft.VisualStudio.Code.Engine", Value=engines_vscode)
    ET.SubElement(properties_el, "Property", Id="Microsoft.VisualStudio.Code.ExtensionDependencies", Value="")
    ET.SubElement(properties_el, "Property", Id="Microsoft.VisualStudio.Code.ExtensionPack", Value="")
    ET.SubElement(properties_el, "Property", Id="Microsoft.VisualStudio.Code.LocalizedLanguages", Value="")
    ET.SubElement(properties_el, "Property", Id="Microsoft.VisualStudio.Code.ExtensionKind", Value="workspace,web")

    installation_el = ET.SubElement(manifest_el, "Installation")
    ET.SubElement(installation_el, "InstallationTarget", Id="Microsoft.VisualStudio.Code")

    ET.SubElement(manifest_el, "Dependencies")

    assets_el = ET.SubElement(manifest_el, "Assets")
    for asset in assets:
        ET.SubElement(
            assets_el,
            "Asset",
            Type=asset["Type"],
            Path=asset["Path"],
            Addressable="true",
        )

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(manifest_el, encoding="utf-8").decode("utf-8")
    )


def should_ignore(path: pathlib.Path, base_dir: pathlib.Path) -> bool:
    """Check if file or directory should be excluded from package."""
    rel = path.relative_to(base_dir)
    for part in rel.parts:
        if part in IGNORED_PATTERNS or part.endswith(".vsix") or part.endswith(".pyc"):
            return True
    return False


def build_vsix(
    extension_dir: Optional[pathlib.Path | str] = None,
    output_path: Optional[pathlib.Path | str] = None,
    verbose: bool = True,
) -> pathlib.Path:
    """Pack the extension directory into a compliant .vsix file.

    Args:
        extension_dir: Root directory of the VS Code extension containing package.json.
        output_path: Target .vsix file path. Defaults to <extension_dir>/<name>-<version>.vsix.
        verbose: If True, prints packaging details.

    Returns:
        pathlib.Path of the generated .vsix archive.
    """
    if extension_dir is None:
        extension_dir = pathlib.Path(__file__).resolve().parent
    else:
        extension_dir = pathlib.Path(extension_dir).resolve()

    pkg_json_file = extension_dir / "package.json"
    if not pkg_json_file.is_file():
        raise FileNotFoundError(f"Missing package.json in extension directory: {extension_dir}")

    with open(pkg_json_file, "r", encoding="utf-8") as f:
        pkg_data = json.load(f)

    name = pkg_data.get("name", "synapse-lang")
    version = pkg_data.get("version", "1.0.0")

    if output_path is None:
        output_path = extension_dir / f"{name}-{version}.vsix"
    else:
        output_path = pathlib.Path(output_path).resolve()

    # Ensure output parent dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather files to package
    files_to_pack: List[tuple[pathlib.Path, str]] = []
    discovered_extensions: Set[str] = set()

    for root, dirs, files in os.walk(extension_dir):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if not should_ignore(pathlib.Path(root) / d, extension_dir)]

        for file_name in files:
            full_path = pathlib.Path(root) / file_name
            if should_ignore(full_path, extension_dir):
                continue
            if full_path.suffix == ".vsix":
                continue

            rel_path = full_path.relative_to(extension_dir)
            # CRITICAL: Always use POSIX forward slashes
            posix_rel = rel_path.as_posix().replace("\\", "/")
            arcname = f"extension/{posix_rel}"
            # Assert strict POSIX compliance
            assert "\\" not in arcname, f"Path contains backslash: {arcname}"

            files_to_pack.append((full_path, arcname))
            if full_path.suffix:
                discovered_extensions.add(full_path.suffix.lower())

    # Build assets list
    assets: List[Dict[str, str]] = [
        {"Type": "Microsoft.VisualStudio.Code.Manifest", "Path": "extension/package.json"}
    ]

    readme_path = extension_dir / "README.md"
    if readme_path.is_file():
        assets.append({"Type": "Microsoft.VisualStudio.Services.Content.Details", "Path": "extension/README.md"})

    # Generate manifests
    content_types_xml = generate_content_types_xml(discovered_extensions)
    vsix_manifest_xml = generate_vsix_manifest(pkg_data, assets)

    # Create ZIP archive
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write root OPC manifests with strict POSIX arcnames
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("extension.vsixmanifest", vsix_manifest_xml)

        # Write extension payload
        for src_file, arcname in files_to_pack:
            # Re-verify arcname POSIX conformity
            clean_arcname = arcname.replace("\\", "/")
            zf.write(src_file, arcname=clean_arcname)

    # Integrity verification
    with zipfile.ZipFile(output_path, "r") as zf:
        namelist = zf.namelist()
        for name_in_zip in namelist:
            if "\\" in name_in_zip:
                raise ValueError(f"CRITICAL ERROR: Archive path contains backslash: {name_in_zip}")
        if "[Content_Types].xml" not in namelist:
            raise ValueError("Missing [Content_Types].xml in archive root")
        if "extension.vsixmanifest" not in namelist:
            raise ValueError("Missing extension.vsixmanifest in archive root")
        if "extension/package.json" not in namelist:
            raise ValueError("Missing extension/package.json in archive")

    if verbose:
        size_kb = output_path.stat().st_size / 1024
        print(f"[VSIX Builder] Successfully built: {output_path} ({size_kb:.1f} KB, {len(namelist)} items)")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure Python VS Code VSIX Packager")
    parser.add_argument(
        "--dir",
        "-d",
        default=str(pathlib.Path(__file__).resolve().parent),
        help="Path to extension root directory (default: script dir)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Target output .vsix filepath",
    )
    args = parser.parse_args()

    try:
        out = build_vsix(extension_dir=args.dir, output_path=args.output, verbose=True)
        print(f"Generated VSIX package: {out}")
    except Exception as e:
        print(f"Build failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
