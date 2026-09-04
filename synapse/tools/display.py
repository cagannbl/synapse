"""
Synapse Rich Display Formatter & Visual Engine (Jupyter & REPL)
===============================================================
Generates sleek, minimalist Linear/Vercel-styled HTML and MIME bundles for
Synapse primitives (DataFrame, Tensor, Models, Plots) in JupyterLab, VS Code,
and interactive notebooks.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional, Tuple, Union


def format_html_dataframe(df: Any, max_rows: int = 10) -> str:
    """Renders a Synapse DataFrame as a modern minimalist dark-mode HTML table."""
    try:
        columns = list(getattr(df, "columns", []))
        dtypes = getattr(df, "dtypes", {})
        total_rows = len(df)
        
        # Extract rows for display
        if hasattr(df, "head"):
            preview_df = df.head(max_rows)
            if hasattr(preview_df, "rows"):
                rows = preview_df.rows
            elif hasattr(preview_df, "to_dict"):
                d = preview_df.to_dict()
                rows = [[d[c][i] for c in columns] for i in range(len(preview_df))]
            else:
                rows = getattr(df, "rows", [])[:max_rows]
        else:
            rows = getattr(df, "rows", [])[:max_rows]
    except Exception:
        return f"<pre>{html.escape(repr(df))}</pre>"

    html_parts = [
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, monospace; '
        'background: #09090b; color: #f4f4f5; border: 1px solid #27272a; border-radius: 8px; '
        'padding: 12px; margin: 8px 0; overflow-x: auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">'
    ]

    # Header summary
    html_parts.append(
        '<div style="display: flex; justify-content: space-between; align-items: center; '
        'margin-bottom: 8px; font-size: 12px; color: #a1a1aa; border-bottom: 1px solid #27272a; padding-bottom: 6px;">'
        f'<span><strong>Synapse DataFrame:</strong> {total_rows} rows &times; {len(columns)} cols</span>'
        f'<span style="background: #18181b; padding: 2px 8px; border-radius: 4px; border: 1px solid #27272a;">Memory-Mapped / Arrow Ready</span>'
        '</div>'
    )

    # Table
    html_parts.append(
        '<table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">'
    )

    # Thead
    html_parts.append('<thead><tr style="border-bottom: 1px solid #3f3f46;">')
    html_parts.append('<th style="padding: 6px 12px; color: #71717a; font-size: 11px;">#</th>')
    for col in columns:
        col_type = dtypes.get(col, "")
        type_badge = f' <span style="font-size: 10px; color: #71717a; font-weight: normal;">({html.escape(str(col_type))})</span>' if col_type else ""
        html_parts.append(f'<th style="padding: 6px 12px; color: #e4e4e7; font-weight: 600;">{html.escape(str(col))}{type_badge}</th>')
    html_parts.append('</tr></thead>')

    # Tbody
    html_parts.append('<tbody>')
    for i, row in enumerate(rows):
        bg = "#09090b" if i % 2 == 0 else "#121215"
        html_parts.append(f'<tr style="background: {bg}; border-bottom: 1px solid #18181b;">')
        html_parts.append(f'<td style="padding: 6px 12px; color: #52525b; font-size: 11px;">{i}</td>')
        for val in row:
            val_str = html.escape(str(val))
            color = "#38bdf8" if isinstance(val, (int, float)) else "#f4f4f5"
            html_parts.append(f'<td style="padding: 6px 12px; color: {color};">{val_str}</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody>')

    html_parts.append('</table>')

    if total_rows > max_rows:
        html_parts.append(
            f'<div style="font-size: 11px; color: #71717a; margin-top: 6px; text-align: right;">'
            f'Showing first {max_rows} of {total_rows} rows...</div>'
        )

    html_parts.append('</div>')
    return "".join(html_parts)


def format_html_tensor(tensor: Any) -> str:
    """Renders a Synapse Tensor with sleek badges, shape, and metadata."""
    try:
        shape = getattr(tensor, "shape", ())
        dtype = str(getattr(tensor, "dtype", "float32"))
        device = str(getattr(tensor, "device", "cpu"))
        requires_grad = bool(getattr(tensor, "requires_grad", False))

        raw_data = getattr(tensor, "data", None)
        stats = ""
        if raw_data is not None and hasattr(raw_data, "min") and hasattr(raw_data, "max"):
            try:
                min_val = float(raw_data.min())
                max_val = float(raw_data.max())
                mean_val = float(raw_data.mean())
                stats = f'<div style="margin-top: 6px; font-size: 11px; color: #71717a;">min: {min_val:.4f} | max: {max_val:.4f} | mean: {mean_val:.4f}</div>'
            except Exception:
                pass

        repr_text = html.escape(str(tensor)[:200])
    except Exception:
        return f"<pre>{html.escape(repr(tensor))}</pre>"

    grad_badge = '<span style="background: #064e3b; color: #34d399; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 4px;">grad</span>' if requires_grad else ""

    return f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #09090b; color: #f4f4f5; border: 1px solid #27272a; border-radius: 8px; padding: 12px; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 12px; border-bottom: 1px solid #27272a; padding-bottom: 6px;">
    <div>
      <span style="font-weight: 600; color: #e4e4e7;">Synapse Tensor</span>
      <span style="background: #18181b; color: #38bdf8; border: 1px solid #27272a; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 6px;">shape: {list(shape)}</span>
      <span style="background: #18181b; color: #a1a1aa; border: 1px solid #27272a; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 4px;">{dtype}</span>
      <span style="background: #18181b; color: #a1a1aa; border: 1px solid #27272a; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 4px;">{device}</span>
      {grad_badge}
    </div>
    <span style="font-size: 11px; color: #71717a;">DLPack Zero-Copy Ready</span>
  </div>
  <pre style="margin: 0; font-size: 12px; color: #d4d4d8; overflow-x: auto;">{repr_text}</pre>
  {stats}
</div>
"""


def render_mime_bundle(obj: Any) -> Dict[str, Any]:
    """Generates a complete Jupyter MIME bundle for a Synapse object."""
    bundle: Dict[str, Any] = {
        "text/plain": repr(obj)
    }

    # Check for DataFrame
    type_name = type(obj).__name__
    if type_name == "DataFrame" or hasattr(obj, "to_arrow_ipc") or (hasattr(obj, "columns") and hasattr(obj, "rows")):
        bundle["text/html"] = format_html_dataframe(obj)
        return bundle

    # Check for Tensor
    if type_name in ("Tensor", "QuantizedTensor") or (hasattr(obj, "shape") and hasattr(obj, "requires_grad")):
        bundle["text/html"] = format_html_tensor(obj)
        return bundle

    # Check for custom _repr_html_ or _repr_mimebundle_
    if hasattr(obj, "_repr_html_"):
        try:
            bundle["text/html"] = obj._repr_html_()
            return bundle
        except Exception:
            pass

    if hasattr(obj, "_repr_mimebundle_"):
        try:
            custom_bundle = obj._repr_mimebundle_()
            if isinstance(custom_bundle, dict):
                bundle.update(custom_bundle)
                return bundle
        except Exception:
            pass

    return bundle
