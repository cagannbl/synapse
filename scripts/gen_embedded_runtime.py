import gzip
import base64
import os

h_path = os.path.join("synapse", "runtime", "synapse_runtime.h")
c_path = os.path.join("synapse", "runtime", "synapse_runtime.c")

with open(h_path, "rb") as f:
    h_bytes = f.read()
with open(c_path, "rb") as f:
    c_bytes = f.read()

h_b64 = base64.b64encode(gzip.compress(h_bytes)).decode("ascii")
c_b64 = base64.b64encode(gzip.compress(c_bytes)).decode("ascii")

target_file = os.path.join("synapse", "codegen", "embedded_runtime.py")

content = f'''"""
Synapse AI Embedded C Runtime Payload
Automatically extracted when physical runtime files are absent (e.g. standalone binary mode).
"""
import os
import sys
import gzip
import base64
import tempfile
from typing import Tuple, Optional

RUNTIME_H_GZ_B64 = "{h_b64}"
RUNTIME_C_GZ_B64 = "{c_b64}"

_EXTRACTED_DIR: Optional[str] = None


def get_embedded_runtime_content() -> Tuple[str, str]:
    h_decomp = gzip.decompress(base64.b64decode(RUNTIME_H_GZ_B64))
    c_decomp = gzip.decompress(base64.b64decode(RUNTIME_C_GZ_B64))
    return h_decomp.decode("utf-8"), c_decomp.decode("utf-8")


def ensure_runtime_extracted(target_dir: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Returns (runtime_dir, runtime_c_path, runtime_h_path).
    If target_dir is None, extracts to system temp directory under 'synapse_rt_embedded'.
    """
    global _EXTRACTED_DIR
    if _EXTRACTED_DIR and os.path.isdir(_EXTRACTED_DIR) and target_dir is None:
        return _EXTRACTED_DIR, os.path.join(_EXTRACTED_DIR, "synapse_runtime.c"), os.path.join(_EXTRACTED_DIR, "synapse_runtime.h")

    if target_dir is None:
        target_dir = os.path.join(tempfile.gettempdir(), "synapse_rt_embedded")

    os.makedirs(target_dir, exist_ok=True)
    h_path = os.path.join(target_dir, "synapse_runtime.h")
    c_path = os.path.join(target_dir, "synapse_runtime.c")

    h_content, c_content = get_embedded_runtime_content()

    if not os.path.isfile(h_path) or os.path.getsize(h_path) != len(h_content.encode("utf-8")):
        with open(h_path, "w", encoding="utf-8") as f:
            f.write(h_content)

    if not os.path.isfile(c_path) or os.path.getsize(c_path) != len(c_content.encode("utf-8")):
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(c_content)

    _EXTRACTED_DIR = target_dir
    return target_dir, c_path, h_path
'''

with open(target_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Successfully generated {target_file} (H size: {len(h_bytes)}, C size: {len(c_bytes)})")
