"""
Static HTML assembly for the comparison dashboard.

Fully self-contained by default: Plotly is inlined directly into the page so
the output HTML has zero dependencies (no server, no sibling files, no
network) -- it can be opened straight from disk via file://, emailed, copied
anywhere, and will still render. This trades a larger file (~6MB, dominated
by Plotly) for never depending on anything external being reachable, which
matters more than load speed when it's unclear whether a given environment's
HTTP server is actually reachable from the browser viewing it.
"""

from __future__ import annotations

import json
from pathlib import Path

_SKYDISCOVER_ROOT = Path(__file__).resolve().parents[2]  # html_utils.py -> compare_dashboard -> tools -> skydiscover
PLOTLY_PATH = _SKYDISCOVER_ROOT / ".venv/lib/python3.12/site-packages/plotly/package_data/plotly.min.js"
TEMPLATE_PATH = Path(__file__).resolve().parent / "dashboard_template.html"

PLOTLY_PLACEHOLDER = "<!--PLOTLY_SCRIPT-->"
DATA_PLACEHOLDER = "/*__COMPARISON_DATA__*/"


def inline_plotly(html: str) -> str:
    if PLOTLY_PATH.exists():
        plotly_js = PLOTLY_PATH.read_text(encoding="utf-8")
        return html.replace(PLOTLY_PLACEHOLDER, f"<script>\n{plotly_js}\n</script>", 1)
    # Fall back to CDN if the vendored copy isn't present in this environment.
    return html.replace(
        PLOTLY_PLACEHOLDER,
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
        1,
    )


def inject_data(html: str, data: dict) -> str:
    # Escape '</script' so embedded data (error messages, code previews) can
    # never prematurely close the surrounding <script> tag -- e.g. a program
    # that errors with a message mentioning HTML/script content, or (once
    # --include-full-code is used) a docstring containing that substring.
    payload = json.dumps(data).replace("</script", "<\\/script")
    return html.replace(DATA_PLACEHOLDER, payload, 1)


def build_dashboard_html(data: dict, output_dir: Path) -> str:
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = inject_data(html, data)
    html = inline_plotly(html)
    return html
