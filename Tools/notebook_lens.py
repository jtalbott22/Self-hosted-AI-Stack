"""
title: Notebook Lens
author: Josh Talbott
author_url: https://joshuaallentalbott.com
description: >
    Run any Jupyter notebook on demand and view every output - text, tables,
    plots, HTML, embedded video, and errors - as a swipeable carousel rendered
    inline in chat. By default each heading section of the notebook becomes ONE
    page (heading + notes + code + outputs stacked, scroll down inside the page),
    like collapsing a heading in Jupyter. Companion to the Lens / Data Lens
    tools; same pattern (LLM infers intent + confirms, Python does the work,
    HTML renders the result without being re-typed by the model).
version: 0.3.0
requirements: nbformat,pydantic,aiohttp,websockets
"""

import re
import json
import time
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any
from urllib.parse import quote

from pydantic import BaseModel, Field

import nbformat
import aiohttp
import websockets

# ============================================================================
# Small utilities
# ============================================================================

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def clean_stream_text(text: str, max_chars: int = 20000) -> str:
    """Collapse carriage-return-overwritten progress-bar spam (tqdm, yfinance,
    pip, etc.) down to the last state of each line, then trim blank runs."""
    if not text:
        return ""
    text = strip_ansi(text)
    lines_out = []
    for raw_line in text.split("\n"):
        if "\r" in raw_line:
            raw_line = raw_line.split("\r")[-1]
        lines_out.append(raw_line)
    cleaned = "\n".join(lines_out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + f"\n\n... [truncated, {len(text)} chars total]"
    return cleaned


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def human_ago(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def safe_json_dumps(obj: Any) -> str:
    """json.dumps that is also safe to drop inside a <script type=application/json>
    block (escapes closing-script sequences)."""
    return json.dumps(obj, default=str).replace("</", "<\\/")


# ============================================================================
# Jupyter Server client (REST + kernel WebSocket) - talks to your existing
# JupyterLab server instead of spawning a local kernel, so execution happens
# where your packages actually live and no filesystem mount is needed.
# ============================================================================


class JupyterClient:
    def __init__(self, base_url: str, token: str, verify_ssl: bool = True):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.verify_ssl = verify_ssl

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"token {self.token}"
        return h

    def _ws_base(self) -> str:
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url[len("https://") :]
        if self.base_url.startswith("http://"):
            return "ws://" + self.base_url[len("http://") :]
        return self.base_url

    def _contents_url(self, path: str) -> str:
        return f"{self.base_url}/api/contents/{quote(path, safe='/')}"

    async def get_contents(
        self, session, path: str, content: bool = True, type_hint: Optional[str] = None
    ):
        params = {}
        if content:
            params["content"] = "1"
        if type_hint:
            params["type"] = type_hint
        async with session.get(
            self._contents_url(path),
            headers=self._headers(),
            params=params,
            ssl=self.verify_ssl,
        ) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return await resp.json()

    async def put_contents(self, session, path: str, body: dict):
        async with session.put(
            self._contents_url(path),
            headers=self._headers(),
            json=body,
            ssl=self.verify_ssl,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def start_kernel(self, session, kernel_name: str) -> dict:
        async with session.post(
            f"{self.base_url}/api/kernels",
            headers=self._headers(),
            json={"name": kernel_name},
            ssl=self.verify_ssl,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def delete_kernel(self, session, kernel_id: str) -> None:
        try:
            async with session.delete(
                f"{self.base_url}/api/kernels/{kernel_id}",
                headers=self._headers(),
                ssl=self.verify_ssl,
            ):
                pass
        except Exception:
            pass  # best-effort cleanup

    def kernel_ws_url(self, kernel_id: str, session_id: str) -> str:
        qs = f"session_id={session_id}"
        if self.token:
            qs += f"&token={self.token}"
        return f"{self._ws_base()}/api/kernels/{kernel_id}/channels?{qs}"


async def _scan_root(
    client: "JupyterClient", session, root_path: str, found: dict
) -> Optional[str]:
    """Recursively lists a content-root-relative folder via the Contents API.
    Returns None on success, or a human-readable problem description."""
    data = await client.get_contents(
        session, root_path, content=True, type_hint="directory"
    )
    if data is None:
        return (
            f'"{root_path or "/"}" was not found on the Jupyter server (404) - check the '
            f"path is relative to the server's own content root, not a host filesystem path."
        )
    if data.get("type") != "directory":
        return f'"{root_path}" exists on the Jupyter server but is not a directory.'
    for item in data.get("content", []):
        name = item.get("name", "")
        if name.startswith(".") or name == ".ipynb_checkpoints":
            continue
        if item.get("type") == "directory":
            await _scan_root(client, session, item["path"], found)
        elif item.get("type") == "notebook":
            found[item["path"]] = {
                "name": Path(name).stem,
                "path": item["path"],
                "last_modified": item.get("last_modified"),
                "size": item.get("size"),
            }
    return None


async def discover_notebooks_remote(
    client: "JupyterClient", session, roots: list, history: dict
):
    found = {}
    problems = []
    for root in roots:
        err = await _scan_root(client, session, root, found)
        if err:
            problems.append(err)
    out = []
    for nb in found.values():
        hist = history.get(nb["path"], {})
        nb = dict(nb)
        nb["last_run"] = hist.get("last_run")
        nb["last_duration"] = hist.get("last_duration")
        nb["last_ok"] = hist.get("last_ok")
        out.append(nb)
    return sorted(out, key=lambda n: n["name"].lower()), problems


def resolve_notebook_remote(name: str, notebooks: list) -> tuple:
    """Returns (matching_notebook_dict, candidate_names_if_ambiguous_or_notfound)."""
    name = name.strip()
    stem_wanted = name[:-6] if name.lower().endswith(".ipynb") else name
    want_path = name if name.endswith(".ipynb") else name + ".ipynb"

    # 1) exact content-path match
    for nb in notebooks:
        if (
            nb["path"] == name
            or nb["path"] == want_path
            or nb["path"].lstrip("/") == want_path.lstrip("/")
        ):
            return nb, []

    # 2) exact stem match (case-insensitive), unique
    exact = [nb for nb in notebooks if nb["name"].lower() == stem_wanted.lower()]
    if len(exact) == 1:
        return exact[0], []
    if len(exact) > 1:
        return None, [nb["path"] for nb in exact]

    # 3) fuzzy substring match on stem
    fuzzy = [nb for nb in notebooks if stem_wanted.lower() in nb["name"].lower()]
    if len(fuzzy) == 1:
        return fuzzy[0], []
    if len(fuzzy) > 1:
        return None, [nb["name"] for nb in fuzzy]

    return None, []


# ============================================================================
# Run history (so list_notebooks can say "last run 2h ago, took 45s")
# ============================================================================


def load_history(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_history(path: Path, history: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2, default=str))
    except Exception:
        pass


def record_run(path: Path, key: str, duration: float, ok: bool) -> None:
    history = load_history(path)
    history[key] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "last_duration": duration,
        "last_ok": ok,
    }
    save_history(path, history)


# ============================================================================
# Notebook outputs -> carousel "pages"
# ============================================================================

IMAGE_MIME_PRIORITY = ["image/svg+xml", "image/png", "image/jpeg", "image/gif"]
VIDEO_MIMES = ["video/mp4", "video/webm", "video/ogg"]


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _heading_text(markdown_source: str) -> Optional[str]:
    for line in markdown_source.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def _leading_heading(markdown_source: str) -> tuple:
    """If the first non-blank line of a markdown cell is an ATX heading, return
    (level, cleaned_heading_text, remaining_body). Otherwise (0, None, source).
    Only a LEADING heading starts a section - that's what Jupyter's collapse
    caret keys off too."""
    lines = markdown_source.splitlines()
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        m = re.match(r"^\s*(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            text = re.sub(r"[*`]", "", m.group(2)).strip()
            return len(m.group(1)), text, "\n".join(lines[n + 1 :])
        break
    return 0, None, markdown_source


def _tiny_markdown_to_html(md: str) -> str:
    """Minimal, dependency-free markdown renderer covering what notebook
    markdown cells actually use: headings, bold/italic, inline code, links,
    bullet/numbered lists, blockquotes, and paragraphs."""
    lines = md.splitlines()
    html_parts = []
    in_list = None  # 'ul' | 'ol' | None
    para_buf = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            html_parts.append("<p>" + inline(" ".join(para_buf)) + "</p>")
            para_buf = []

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append(f"</{in_list}>")
            in_list = None

    def inline(text: str) -> str:
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            r'<a href="\2" target="_blank" rel="noopener">\1</a>',
            text,
        )
        return text

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        m_h = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        m_ul = re.match(r"^[-*]\s+(.*)$", stripped)
        m_ol = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        m_bq = re.match(r"^>\s?(.*)$", stripped)

        if not stripped:
            flush_para()
            close_list()
            continue
        if m_h:
            flush_para()
            close_list()
            level = len(m_h.group(1))
            html_parts.append(f"<h{level}>{inline(m_h.group(2))}</h{level}>")
        elif m_ul:
            flush_para()
            if in_list != "ul":
                close_list()
                html_parts.append("<ul>")
                in_list = "ul"
            html_parts.append(f"<li>{inline(m_ul.group(1))}</li>")
        elif m_ol:
            flush_para()
            if in_list != "ol":
                close_list()
                html_parts.append("<ol>")
                in_list = "ol"
            html_parts.append(f"<li>{inline(m_ol.group(1))}</li>")
        elif m_bq:
            flush_para()
            close_list()
            html_parts.append(f"<blockquote>{inline(m_bq.group(1))}</blockquote>")
        elif stripped == "---":
            flush_para()
            close_list()
            html_parts.append("<hr>")
        else:
            close_list()
            para_buf.append(stripped)

    flush_para()
    close_list()
    return "\n".join(html_parts)


def _output_data_dict(output: dict) -> dict:
    return output.get("data", {}) or {}


def _best_rich_mime(data: dict) -> Optional[str]:
    for m in IMAGE_MIME_PRIORITY:
        if m in data:
            return m
    for m in VIDEO_MIMES:
        if m in data:
            return m
    if "text/html" in data:
        return "text/html"
    if "application/vnd.plotly.v1+json" in data:
        return "application/vnd.plotly.v1+json"
    if "text/plain" in data:
        return "text/plain"
    return None


def _mime_value(data: dict, mime: str) -> str:
    v = data.get(mime, "")
    if isinstance(v, list):
        return "".join(v)
    return v


def cell_to_pages(
    cell, index: int, section: Optional[str], include_markdown: bool
) -> list[dict]:
    pages = []
    source = (
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else cell.get("source", "")
    )

    if cell["cell_type"] == "markdown":
        heading = _heading_text(source)
        body_lines = [l for l in source.splitlines() if not l.strip().startswith("#")]
        has_body = any(l.strip() and l.strip() != "---" for l in body_lines)
        if include_markdown and (has_body or not heading):
            pages.append(
                {
                    "type": "markdown",
                    "title": heading or f"Notes - cell {index}",
                    "section": section,
                    "content": _tiny_markdown_to_html(source),
                    "cell_index": index,
                }
            )
        return pages

    if cell["cell_type"] != "code":
        return pages

    stream_text = []
    stderr_present = False
    rich_pages = []
    error_page = None

    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if out.get("name") == "stderr":
                stderr_present = True
                stream_text.append(f"[stderr]\n{text}")
            else:
                stream_text.append(text)
        elif otype in ("display_data", "execute_result"):
            data = _output_data_dict(out)
            mime = _best_rich_mime(data)
            if not mime:
                continue
            value = _mime_value(data, mime)
            if mime in IMAGE_MIME_PRIORITY:
                if mime == "image/svg+xml":
                    rich_pages.append({"type": "image_svg", "content": value})
                else:
                    rich_pages.append({"type": "image", "mime": mime, "content": value})
            elif mime in VIDEO_MIMES:
                rich_pages.append({"type": "video", "mime": mime, "content": value})
            elif mime == "text/html":
                rich_pages.append({"type": "html", "content": value})
            elif mime == "application/vnd.plotly.v1+json":
                # Prefer a plain image fallback if one was also captured
                png = data.get("image/png")
                if png:
                    rich_pages.append(
                        {"type": "image", "mime": "image/png", "content": png}
                    )
                else:
                    rich_pages.append(
                        {
                            "type": "plotly",
                            "content": (
                                json.dumps(value)
                                if not isinstance(value, str)
                                else value
                            ),
                        }
                    )
            elif mime == "text/plain":
                text = value if isinstance(value, str) else "".join(value)
                rich_pages.append({"type": "text", "content": text})
        elif otype == "error":
            tb = "\n".join(strip_ansi(l) for l in out.get("traceback", []))
            error_page = {
                "type": "error",
                "ename": out.get("ename", "Error"),
                "evalue": out.get("evalue", ""),
                "content": tb,
            }

    base_title = f"Cell {index}"
    code_snippet = source.strip()

    if stream_text:
        pages.append(
            {
                "type": "stream",
                "title": base_title,
                "section": section,
                "content": clean_stream_text("\n".join(stream_text)),
                "has_stderr": stderr_present,
                "code": code_snippet,
                "cell_index": index,
            }
        )

    for i, rp in enumerate(rich_pages):
        rp["title"] = base_title if len(rich_pages) == 1 else f"{base_title}.{i+1}"
        rp["section"] = section
        rp["code"] = code_snippet
        rp["cell_index"] = index
        pages.append(rp)

    if error_page:
        error_page["title"] = f"{base_title} - Error"
        error_page["section"] = section
        error_page["code"] = code_snippet
        error_page["cell_index"] = index
        pages.append(error_page)

    return pages


def notebook_to_pages(nb, include_markdown: bool = True) -> list[dict]:
    """Original layout: one carousel page per cell output."""
    pages = []
    current_section = None
    for i, cell in enumerate(nb.cells):
        if cell["cell_type"] == "markdown":
            src = (
                "".join(cell.get("source", []))
                if isinstance(cell.get("source"), list)
                else cell.get("source", "")
            )
            h = _heading_text(src)
            if h:
                current_section = h
        pages.extend(cell_to_pages(cell, i, current_section, include_markdown))
    return pages


def notebook_to_sections(
    nb,
    include_markdown: bool = True,
    section_level: int = 2,
    code_mode: str = "collapsed",
) -> list[dict]:
    """Section layout: one carousel page per heading section - the same span a
    Jupyter collapse caret would fold. A markdown cell whose FIRST line is a
    heading at level <= section_level starts a new page; deeper headings stay
    inline. Every cell up to the next such heading (notes, code, outputs) is
    stacked on that one page, in notebook order.

    code_mode: "collapsed" (default; every code block folded behind a
    "view code" button), "expanded" (code shown; setup cells with no output
    stay folded), or "hidden" (no code at all)."""
    section_level = max(1, min(6, int(section_level or 2)))
    if code_mode not in ("expanded", "collapsed", "hidden"):
        code_mode = "collapsed"

    pages: list[dict] = []
    stack: dict = {}  # heading level -> heading text, for the "parent" pill

    def new_page(title, level, parent, cell_index):
        return {
            "type": "section",
            "title": title,
            "level": level,
            "parent": parent,
            "blocks": [],
            "cell_index": cell_index,
        }

    def flush(page):
        if page["blocks"]:
            pages.append(page)

    current = new_page("Overview", 0, None, None)

    for i, cell in enumerate(nb.cells):
        ctype = cell.get("cell_type")
        src = _cell_source(cell)

        if ctype == "markdown":
            level, text, body = _leading_heading(src)
            if level and level <= section_level:
                flush(current)
                for k in [k for k in stack if k >= level]:
                    del stack[k]
                parent = stack[max(stack)] if stack else None
                stack[level] = text
                current = new_page(text, level, parent, i)
                if include_markdown and body.strip():
                    current["blocks"].append(
                        {"type": "md", "content": _tiny_markdown_to_html(body)}
                    )
            elif include_markdown and src.strip():
                current["blocks"].append(
                    {"type": "md", "content": _tiny_markdown_to_html(src)}
                )
            continue

        if ctype != "code":
            continue

        code = src.strip()
        outputs = cell_to_pages(cell, i, None, False)  # output pages for this cell

        if code_mode != "hidden" and code:
            current["blocks"].append(
                {
                    "type": "codesrc",
                    "content": code,
                    "lines": code.count("\n") + 1,
                    # setup cells (imports etc.) with nothing to show stay folded
                    "open": code_mode == "expanded" and bool(outputs),
                }
            )
        for p in outputs:
            block = {
                k: v
                for k, v in p.items()
                if k not in ("title", "section", "code", "cell_index")
            }
            current["blocks"].append(block)
        if current["cell_index"] is None:
            current["cell_index"] = i

    flush(current)
    return pages


def _flat_outputs(pages: list[dict]):
    """Yield every renderable unit, whether pages are per-cell or per-section."""
    for p in pages:
        if p.get("type") == "section":
            for b in p.get("blocks", []):
                yield b
        else:
            yield p


# ============================================================================
# Carousel HTML template
# ============================================================================

_CAROUSEL_TEMPLATE = r"""
<div class="nblens-root" data-theme="__THEME__">
<style>
.nblens-root{
  /* dark is the base; light is applied below via [data-theme] / prefers-color-scheme */
  --bg:#0f1115;--panel:#161920;--panel2:#1e222b;--raised:#252a35;
  --text:#e9ecf1;--muted:#98a2b3;--accent:#5eb0ff;--border:#2a2f3a;
  --err:#ff7b7b;--errbg:#3a1f22;--stderr:#e8b84b;--shadow:rgba(0,0,0,.45);
  color-scheme:dark;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
  color:var(--text);background:var(--bg);
  /* fill the screen: dvh tracks mobile browser chrome as it hides/shows */
  height:100vh;height:100dvh;width:100%;
  display:flex;flex-direction:column;overflow:hidden;position:relative;
  /* keep content clear of notch / home indicator */
  padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
  box-sizing:border-box;
}
.nblens-root[data-theme="light"]{
  --bg:#f5f6f8;--panel:#ffffff;--panel2:#eceff3;--raised:#e2e6ec;
  --text:#14171c;--muted:#5b6472;--accent:#0b6bcb;--border:#d7dce3;
  --err:#c0332f;--errbg:#fdecec;--stderr:#8a6100;--shadow:rgba(0,0,0,.12);
  color-scheme:light;
}
@media (prefers-color-scheme:light){
  .nblens-root[data-theme="auto"]{
    --bg:#f5f6f8;--panel:#ffffff;--panel2:#eceff3;--raised:#e2e6ec;
    --text:#14171c;--muted:#5b6472;--accent:#0b6bcb;--border:#d7dce3;
    --err:#c0332f;--errbg:#fdecec;--stderr:#8a6100;--shadow:rgba(0,0,0,.12);
    color-scheme:light;
  }
}
.nblens-root *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}

/* ---------- header ---------- */
.nblens-head{
  flex:0 0 auto;display:flex;align-items:center;gap:8px;
  padding:10px 12px;background:var(--panel2);border-bottom:1px solid var(--border);
}
.nblens-title{font-weight:650;font-size:15px;letter-spacing:-.01em;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1 1 auto;min-width:0;}
.nblens-meta{font-size:11px;color:var(--muted);white-space:nowrap;flex:0 0 auto;}
.nblens-jump{
  flex:0 0 auto;background:var(--panel);color:var(--text);border:1px solid var(--border);
  border-radius:9px;padding:8px 9px;font-size:12px;max-width:34vw;min-height:36px;
  font-family:inherit;
}
/* on narrow screens the meta line is noise - the counter already says where you are */
@media (max-width:560px){ .nblens-meta{display:none;} .nblens-jump{max-width:42vw;} }

/* ---------- viewport / track ---------- */
.nblens-viewport{position:relative;flex:1 1 auto;min-height:0;overflow:hidden;background:var(--panel);}
.nblens-track{display:flex;height:100%;will-change:transform;}
.nblens-track.animate{transition:transform .3s cubic-bezier(.22,.9,.32,1);}
.nblens-page{
  flex:0 0 100%;width:100%;height:100%;min-height:0;
  display:flex;flex-direction:column;padding:12px 14px 6px;
}

/* ---------- page chrome ---------- */
.nblens-section-pill{
  display:inline-block;align-self:flex-start;background:var(--raised);
  border:1px solid var(--border);border-radius:999px;padding:3px 11px;
  font-size:11px;color:var(--accent);margin-bottom:6px;max-width:100%;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:0 0 auto;
}
.nblens-page-label{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin-bottom:7px;flex:0 0 auto;}
.nblens-code-toggle{
  flex:0 0 auto;align-self:flex-start;background:none;border:1px solid var(--border);
  color:var(--muted);border-radius:7px;font-size:11px;padding:6px 10px;margin-bottom:8px;
  cursor:pointer;min-height:32px;font-family:inherit;
}
.nblens-code-toggle:active{background:var(--raised);}
.nblens-code-drawer{
  display:none;flex:0 0 auto;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.5px;line-height:1.5;background:var(--panel2);border:1px solid var(--border);
  border-radius:8px;padding:10px;margin-bottom:8px;white-space:pre;
  max-height:30%;overflow:auto;overscroll-behavior:contain;-webkit-overflow-scrolling:touch;
}
.nblens-code-drawer.open{display:block;}

/* ---------- the scrollable body of each page ----------
   min-height:0 is what actually lets these scroll inside a flex column      */
.nblens-body{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;}
.nblens-scroll{
  flex:1 1 auto;min-height:0;overflow:auto;
  overscroll-behavior:contain;-webkit-overflow-scrolling:touch;
}

/* text / stream output */
.nblens-pre{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word;
  background:var(--panel2);border:1px solid var(--border);border-radius:10px;
  padding:12px;margin:0;
}
.nblens-pre .stderr-line{color:var(--stderr);}

/* markdown */
.nblens-md{padding-right:2px;}
.nblens-md h1,.nblens-md h2,.nblens-md h3{margin:.45em 0 .3em;line-height:1.25;}
.nblens-md h1{font-size:1.4em;} .nblens-md h2{font-size:1.2em;} .nblens-md h3{font-size:1.05em;}
.nblens-md p{line-height:1.6;margin:.55em 0;}
.nblens-md ul,.nblens-md ol{padding-left:1.3em;line-height:1.6;}
.nblens-md code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:.9em;
  font-family:ui-monospace,Menlo,monospace;}
.nblens-md a{color:var(--accent);}
.nblens-md blockquote{border-left:3px solid var(--accent);margin:.6em 0;padding:.2em .9em;color:var(--muted);}
.nblens-md hr{border:none;border-top:1px solid var(--border);margin:1em 0;}

/* images: fit by default; pinch or double-tap to zoom, drag to pan.
   Gestures are handled in JS (transform-based) rather than by making the
   container scroll - touch-action:none hands us every touch, which is what
   lets panning work at all once zoomed. */
.nblens-img-wrap{
  flex:1 1 auto;min-height:0;position:relative;overflow:hidden;
  display:flex;align-items:center;justify-content:center;
  touch-action:none;cursor:zoom-in;
}
.nblens-img-wrap img,.nblens-img-wrap svg{
  max-width:100%;max-height:100%;object-fit:contain;border-radius:8px;display:block;
  transform-origin:0 0;will-change:transform;
}
.nblens-img-wrap.zoomed{cursor:grab;}
.nblens-img-wrap.zoomed:active{cursor:grabbing;}
.nblens-zoom-badge{
  position:absolute;top:8px;right:8px;background:var(--raised);border:1px solid var(--border);
  color:var(--text);border-radius:999px;padding:4px 10px;font-size:11px;
  font-variant-numeric:tabular-nums;pointer-events:none;opacity:0;transition:opacity .18s;
  box-shadow:0 2px 8px var(--shadow);
}
.nblens-zoom-badge.show{opacity:1;}
.nblens-img-hint{flex:0 0 auto;font-size:10.5px;color:var(--muted);text-align:center;padding-top:4px;}

/* html output (pandas tables etc) - scrolls in BOTH axes without moving the page */
.nblens-html-wrap{
  flex:1 1 auto;min-height:0;overflow:auto;overscroll-behavior:contain;
  -webkit-overflow-scrolling:touch;background:var(--panel2);
  border:1px solid var(--border);border-radius:10px;padding:10px;
}
.nblens-html-wrap table{border-collapse:collapse;width:max-content;min-width:100%;font-size:12.5px;}
.nblens-html-wrap th,.nblens-html-wrap td{
  border:1px solid var(--border);padding:6px 10px;text-align:right;white-space:nowrap;}
.nblens-html-wrap th{position:sticky;top:0;background:var(--panel2);text-align:center;z-index:1;}
.nblens-html-wrap tr:nth-child(even) td{background:rgba(127,127,127,.07);}
.nblens-html-wrap img{max-width:100%;height:auto;}

/* video */
.nblens-video{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;}
.nblens-video video{max-width:100%;max-height:100%;border-radius:10px;}

/* errors */
.nblens-error{
  flex:1 1 auto;min-height:0;overflow:auto;overscroll-behavior:contain;
  -webkit-overflow-scrolling:touch;background:var(--errbg);
  border:1px solid var(--err);border-radius:10px;padding:14px;
}
.nblens-error h4{color:var(--err);margin:0 0 8px;font-size:14px;}
.nblens-error pre{white-space:pre-wrap;word-break:break-word;
  font-family:ui-monospace,Menlo,monospace;font-size:12px;line-height:1.5;margin:0;}

/* ---------- section pages: notes + code + outputs stacked on ONE page ----------
   The whole page scrolls vertically. Big outputs (long logs, wide tables, long
   code) are height-capped and scroll inside their own box so one huge output
   can't bury everything below it. Images sit inline; tap one to open the
   full-screen zoom viewer (inline pinch would fight vertical scrolling). */
.nblens-page.sec{padding-bottom:0;}
.nblens-sec-scroll{
  flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;
  overscroll-behavior:contain;-webkit-overflow-scrolling:touch;padding:0 2px 18px 0;
}
.nblens-sec-title{font-size:19px;font-weight:650;letter-spacing:-.01em;line-height:1.25;margin:0 0 10px;}
.nblens-blk{margin:0 0 12px;}
.nblens-blk > .nblens-md > :first-child{margin-top:0;}
.nblens-sec .nblens-code-toggle{margin-bottom:6px;}
.nblens-sec .nblens-code-drawer{max-height:280px;}
.nblens-sec .nblens-pre{max-height:360px;overflow:auto;overscroll-behavior:contain;}
.nblens-sec .nblens-html-wrap{max-height:420px;}
.nblens-sec .nblens-error{max-height:360px;}
.nblens-sec-media{
  display:block;width:100%;background:var(--panel2);border:1px solid var(--border);
  border-radius:10px;padding:6px;cursor:zoom-in;
}
.nblens-sec-media img,.nblens-sec-media svg{
  display:block;max-width:100%;height:auto;margin:0 auto;border-radius:6px;
}
.nblens-sec-media-hint{font-size:10.5px;color:var(--muted);text-align:center;margin-top:4px;}

/* ---------- full-screen image viewer (opened by tapping an inline image) ---------- */
.nblens-lb{
  display:none;position:absolute;left:0;right:0;top:0;bottom:0;z-index:20;
  background:var(--bg);flex-direction:column;outline:none;
  padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
}
.nblens-lb.open{display:flex;}
.nblens-lb-bar{
  flex:0 0 auto;display:flex;align-items:center;gap:10px;padding:10px 12px;
  background:var(--panel2);border-bottom:1px solid var(--border);
  font-size:11.5px;color:var(--muted);
}
.nblens-lb-close{
  background:var(--panel);color:var(--text);border:1px solid var(--border);
  border-radius:9px;padding:0 14px;min-height:38px;font-size:13px;cursor:pointer;font-family:inherit;
}
.nblens-lb-close:active{background:var(--raised);}
.nblens-lb-body{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;padding:10px 12px;}

/* ---------- nav ---------- */
.nblens-nav{
  flex:0 0 auto;display:flex;align-items:center;gap:12px;
  padding:10px 12px;background:var(--panel2);border-top:1px solid var(--border);
}
.nblens-btn{
  flex:0 0 auto;background:var(--panel);color:var(--text);border:1px solid var(--border);
  border-radius:10px;width:44px;height:44px;font-size:19px;line-height:1;cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-family:inherit;
}
.nblens-btn:active{background:var(--raised);}
.nblens-btn:disabled{opacity:.3;cursor:default;}
.nblens-progress-wrap{flex:1 1 auto;height:6px;background:var(--panel);
  border:1px solid var(--border);border-radius:4px;overflow:hidden;}
.nblens-progress-bar{height:100%;background:var(--accent);width:0%;transition:width .25s;}
.nblens-count{flex:0 0 auto;font-size:12.5px;color:var(--muted);
  min-width:52px;text-align:center;font-variant-numeric:tabular-nums;}
</style>

<div class="nblens-head">
  <div class="nblens-title">__TITLE__</div>
  <div class="nblens-meta">__META__</div>
  <select class="nblens-jump" id="nblens-jump-__UID__" aria-label="Jump to section"></select>
</div>
<div class="nblens-viewport" id="nblens-vp-__UID__">
  <div class="nblens-track" id="nblens-track-__UID__"></div>
</div>
<div class="nblens-nav">
  <button class="nblens-btn" id="nblens-prev-__UID__" aria-label="Previous page">&#8249;</button>
  <div class="nblens-progress-wrap"><div class="nblens-progress-bar" id="nblens-bar-__UID__"></div></div>
  <div class="nblens-count" id="nblens-count-__UID__"></div>
  <button class="nblens-btn" id="nblens-next-__UID__" aria-label="Next page">&#8250;</button>
</div>

<script type="application/json" id="nblens-data-__UID__">__PAGES_JSON__</script>
<script>
(function(){
  var uid = "__UID__";
  var pages = JSON.parse(document.getElementById("nblens-data-" + uid).textContent);
  var track = document.getElementById("nblens-track-" + uid);
  var vp    = document.getElementById("nblens-vp-" + uid);
  var root  = vp.closest(".nblens-root");
  var jump  = document.getElementById("nblens-jump-" + uid);
  var prevBtn = document.getElementById("nblens-prev-" + uid);
  var nextBtn = document.getElementById("nblens-next-" + uid);
  var bar   = document.getElementById("nblens-bar-" + uid);
  var count = document.getElementById("nblens-count-" + uid);
  var idx = 0;

  function esc(s){ return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  function streamLines(content){
    return esc(content).split("\n").map(function(l){
      return l.indexOf("[stderr]") === 0 ? '<span class="stderr-line">'+l+'</span>' : l;
    }).join("\n");
  }

  function streamHtml(p){
    return '<div class="nblens-scroll"><pre class="nblens-pre">' + streamLines(p.content) + '</pre></div>';
  }

  var PLOTLY_NOTE = 'Interactive Plotly output. Set the PLOTLY_CDN_URL valve to render these interactively; otherwise a static PNG is used when the notebook produced one.';

  function bodyHtml(p){
    if(p.type === "image"){
      return '<div class="nblens-img-wrap"><img src="data:'+p.mime+';base64,'+p.content+'" loading="lazy" alt=""></div>'
           + '<div class="nblens-img-hint">pinch or double-tap to zoom &middot; drag to pan</div>';
    }
    if(p.type === "image_svg"){
      return '<div class="nblens-img-wrap">'+p.content+'</div>'
           + '<div class="nblens-img-hint">pinch or double-tap to zoom &middot; drag to pan</div>';
    }
    if(p.type === "video"){
      return '<div class="nblens-video"><video controls playsinline src="data:'+p.mime+';base64,'+p.content+'"></video></div>';
    }
    if(p.type === "html"){
      return '<div class="nblens-html-wrap">' + p.content + '</div>';
    }
    if(p.type === "markdown"){
      return '<div class="nblens-scroll"><div class="nblens-md">' + p.content + '</div></div>';
    }
    if(p.type === "error"){
      return '<div class="nblens-error"><h4>'+esc(p.ename)+': '+esc(p.evalue)+'</h4><pre>'+esc(p.content)+'</pre></div>';
    }
    if(p.type === "plotly"){
      return '<div class="nblens-scroll"><pre class="nblens-pre">' + PLOTLY_NOTE + '</pre></div>';
    }
    return streamHtml(p);
  }

  /* one block of a section page (notes / code / a single output) */
  function blockHtml(b){
    var t = b.type;
    if(t === "md"){
      return '<div class="nblens-blk"><div class="nblens-md">' + b.content + '</div></div>';
    }
    if(t === "codesrc"){
      return '<div class="nblens-blk">'
           + '<button class="nblens-code-toggle" type="button">&lt;/&gt; view code</button>'
           + '<div class="nblens-code-drawer' + (b.open ? ' open' : '') + '">' + esc(b.content) + '</div></div>';
    }
    if(t === "image"){
      return '<div class="nblens-blk"><div class="nblens-sec-media"><img src="data:'+b.mime+';base64,'+b.content+'" loading="lazy" alt=""></div>'
           + '<div class="nblens-sec-media-hint">tap to zoom</div></div>';
    }
    if(t === "image_svg"){
      return '<div class="nblens-blk"><div class="nblens-sec-media">'+b.content+'</div>'
           + '<div class="nblens-sec-media-hint">tap to zoom</div></div>';
    }
    if(t === "video"){
      return '<div class="nblens-blk"><video controls playsinline style="width:100%;display:block;border-radius:10px" src="data:'+b.mime+';base64,'+b.content+'"></video></div>';
    }
    if(t === "html"){
      return '<div class="nblens-blk"><div class="nblens-html-wrap">' + b.content + '</div></div>';
    }
    if(t === "error"){
      return '<div class="nblens-blk"><div class="nblens-error"><h4>'+esc(b.ename)+': '+esc(b.evalue)+'</h4><pre>'+esc(b.content)+'</pre></div></div>';
    }
    if(t === "plotly"){
      return '<div class="nblens-blk"><pre class="nblens-pre">' + PLOTLY_NOTE + '</pre></div>';
    }
    return '<div class="nblens-blk"><pre class="nblens-pre">' + streamLines(b.content) + '</pre></div>';
  }

  function pageInner(p){
    if(p.type === "section"){
      var s = '<div class="nblens-sec-scroll">';
      if(p.parent){ s += '<div class="nblens-section-pill">'+esc(p.parent)+'</div>'; }
      s += '<div class="nblens-sec-title">'+esc(p.title||"")+'</div>';
      (p.blocks||[]).forEach(function(b){ s += blockHtml(b); });
      return s + '</div>';
    }
    var h = "";
    if(p.section){ h += '<div class="nblens-section-pill">'+esc(p.section)+'</div>'; }
    h += '<div class="nblens-page-label">' + esc(p.title||"") + '</div>';
    if(p.code){
      h += '<button class="nblens-code-toggle" type="button">&lt;/&gt; view code</button>'
        +  '<div class="nblens-code-drawer">' + esc(p.code) + '</div>';
    }
    h += '<div class="nblens-body">' + bodyHtml(p) + '</div>';
    return h;
  }

  pages.forEach(function(p){
    var div = document.createElement("div");
    div.className = "nblens-page" + (p.type === "section" ? " sec" : "");
    div.innerHTML = pageInner(p);
    track.appendChild(div);
  });

  /* ---- image zoom: transform-based, so panning works once zoomed ----
     Positions are tracked as the media's absolute top-left (px,py) inside the
     wrapper, which makes the pinch-anchor and clamping math straightforward.
     Gestures stopPropagation so a pan doesn't also turn the carousel page.
     Returns a destroy() that unhooks the window-level listeners (needed
     because the full-screen viewer creates a fresh one each time it opens). */
  function setupZoom(wrap){
    var media = wrap.querySelector("img, svg");
    if(!media) return function(){};
    var badge = document.createElement("div");
    badge.className = "nblens-zoom-badge";
    wrap.appendChild(badge);

    var s = 1, px = 0, py = 0;
    var fitW = 0, fitH = 0, x0 = 0, y0 = 0, cw = 0, ch = 0;
    var MAXS = 8;

    function measure(){
      var wr = wrap.getBoundingClientRect();
      cw = wr.width; ch = wr.height;
      var prev = media.style.transform;
      media.style.transform = "none";
      var mr = media.getBoundingClientRect();
      fitW = mr.width; fitH = mr.height;
      media.style.transform = prev;
      x0 = (cw - fitW) / 2; y0 = (ch - fitH) / 2;
    }
    function clamp(){
      if(fitW * s <= cw) px = (cw - fitW * s) / 2;
      else px = Math.min(0, Math.max(cw - fitW * s, px));
      if(fitH * s <= ch) py = (ch - fitH * s) / 2;
      else py = Math.min(0, Math.max(ch - fitH * s, py));
    }
    function apply(){
      clamp();
      media.style.transform = "translate(" + (px - x0) + "px," + (py - y0) + "px) scale(" + s + ")";
      var z = s > 1.01;
      wrap.classList.toggle("zoomed", z);
      badge.textContent = s.toFixed(1) + "\u00d7";
      badge.classList.toggle("show", z);
    }
    function zoomAt(newS, cx, cy){
      newS = Math.max(1, Math.min(MAXS, newS));
      var u = (cx - px) / s, v = (cy - py) / s;
      s = newS;
      px = cx - u * s; py = cy - v * s;
      apply();
    }
    function reset(){ s = 1; measure(); px = x0; py = y0; apply(); }
    function local(t){
      var r = wrap.getBoundingClientRect();
      return [t.clientX - r.left, t.clientY - r.top];
    }

    if(media.tagName.toLowerCase() === "img" && !media.complete){
      media.addEventListener("load", reset);
    }
    setTimeout(reset, 0);
    function onResize(){
      var wasZoomed = s > 1.01;
      measure();
      if(wasZoomed) apply(); else reset();
    }
    window.addEventListener("resize", onResize);

    var mode = null, startDist = 0, startS = 1, lastX = 0, lastY = 0, lastTap = 0;

    wrap.addEventListener("touchstart", function(e){
      if(e.touches.length === 2){
        mode = "pinch";
        var a = local(e.touches[0]), b = local(e.touches[1]);
        startDist = Math.hypot(a[0]-b[0], a[1]-b[1]) || 1;
        startS = s;
        lastX = (a[0]+b[0])/2; lastY = (a[1]+b[1])/2;
        e.stopPropagation();
      } else if(e.touches.length === 1){
        var p = local(e.touches[0]);
        lastX = p[0]; lastY = p[1];
        var now = Date.now();
        if(now - lastTap < 300){
          if(s > 1.01) reset(); else zoomAt(2.5, p[0], p[1]);
          lastTap = 0; mode = null;
          e.stopPropagation();
          if(e.cancelable) e.preventDefault();
          return;
        }
        lastTap = now;
        mode = s > 1.01 ? "pan" : null;
        if(mode) e.stopPropagation();
      }
    }, {passive:false});

    wrap.addEventListener("touchmove", function(e){
      if(mode === "pinch" && e.touches.length === 2){
        var a = local(e.touches[0]), b = local(e.touches[1]);
        var d = Math.hypot(a[0]-b[0], a[1]-b[1]) || 1;
        var mx = (a[0]+b[0])/2, my = (a[1]+b[1])/2;
        px += mx - lastX; py += my - lastY;
        lastX = mx; lastY = my;
        zoomAt(startS * (d / startDist), mx, my);
        e.stopPropagation();
        if(e.cancelable) e.preventDefault();
      } else if(mode === "pan" && e.touches.length === 1){
        var p = local(e.touches[0]);
        px += p[0] - lastX; py += p[1] - lastY;
        lastX = p[0]; lastY = p[1];
        apply();
        e.stopPropagation();
        if(e.cancelable) e.preventDefault();
      }
    }, {passive:false});

    function endTouch(e){
      if(mode) e.stopPropagation();
      if(e.touches.length === 0){ mode = null; }
      else if(e.touches.length === 1){
        mode = s > 1.01 ? "pan" : null;
        var p = local(e.touches[0]);
        lastX = p[0]; lastY = p[1];
      }
    }
    wrap.addEventListener("touchend", endTouch, {passive:false});
    wrap.addEventListener("touchcancel", endTouch, {passive:false});

    // desktop: wheel to zoom, double-click to toggle, drag to pan
    wrap.addEventListener("wheel", function(e){
      var r = wrap.getBoundingClientRect();
      zoomAt(s * (e.deltaY < 0 ? 1.15 : 1/1.15), e.clientX - r.left, e.clientY - r.top);
      if(e.cancelable) e.preventDefault();
    }, {passive:false});
    wrap.addEventListener("dblclick", function(e){
      var r = wrap.getBoundingClientRect();
      if(s > 1.01) reset(); else zoomAt(2.5, e.clientX - r.left, e.clientY - r.top);
      e.preventDefault();
    });
    var mDown = false;
    wrap.addEventListener("mousedown", function(e){
      if(s > 1.01){ mDown = true; lastX = e.clientX; lastY = e.clientY; e.preventDefault(); }
    });
    function onMouseMove(e){
      if(!mDown) return;
      px += e.clientX - lastX; py += e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      apply();
    }
    function onMouseUp(){ mDown = false; }
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    return function destroy(){
      window.removeEventListener("resize", onResize);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }

  // per-cell layout: images are zoomable right in the page
  Array.prototype.forEach.call(track.querySelectorAll(".nblens-img-wrap"), function(w){ setupZoom(w); });

  /* ---- full-screen image viewer, used by section pages ---- */
  var lb = document.createElement("div");
  lb.className = "nblens-lb";
  lb.tabIndex = -1;
  lb.innerHTML = '<div class="nblens-lb-bar"><button class="nblens-lb-close" type="button">&#10005; Close</button>'
               + '<span>pinch or double-tap to zoom &middot; drag to pan</span></div>'
               + '<div class="nblens-lb-body"></div>';
  root.appendChild(lb);
  var lbBody = lb.querySelector(".nblens-lb-body");
  var lbDestroy = null;

  function openLightbox(media){
    var src = media.querySelector("img, svg");
    if(!src) return;
    var el;
    if(src.tagName.toLowerCase() === "img"){
      el = document.createElement("img");
      el.src = src.src; el.alt = "";
    } else {
      el = src.cloneNode(true);
    }
    var wrap = document.createElement("div");
    wrap.className = "nblens-img-wrap";
    wrap.appendChild(el);
    lbBody.innerHTML = "";
    lbBody.appendChild(wrap);
    lb.classList.add("open");
    lbDestroy = setupZoom(wrap);
    lb.focus();
  }
  function closeLightbox(){
    if(lbDestroy){ lbDestroy(); lbDestroy = null; }
    lb.classList.remove("open");
    lbBody.innerHTML = "";
  }
  lb.querySelector(".nblens-lb-close").addEventListener("click", closeLightbox);
  lb.addEventListener("keydown", function(e){
    if(e.key === "Escape"){ closeLightbox(); e.preventDefault(); }
  });

  // delegated clicks: code drawer toggle + tap-to-zoom on inline images
  track.addEventListener("click", function(e){
    var t = e.target.closest ? e.target.closest(".nblens-code-toggle") : null;
    if(t){ var d = t.nextElementSibling; if(d) d.classList.toggle("open"); return; }
    var m = e.target.closest ? e.target.closest(".nblens-sec-media") : null;
    if(m) openLightbox(m);
  });

  // section jump menu
  var seen = {};
  var opt0 = document.createElement("option");
  opt0.value = "-1"; opt0.textContent = "Jump to\u2026";
  jump.appendChild(opt0);
  pages.forEach(function(p, i){
    if(p.type === "section"){
      var so = document.createElement("option");
      so.value = String(i);
      so.textContent = (p.parent ? "\u2003" : "") + (p.title || ("Section " + (i+1)));
      jump.appendChild(so);
    } else if(p.section && !seen[p.section]){
      seen[p.section] = 1;
      var o = document.createElement("option");
      o.value = String(i); o.textContent = p.section;
      jump.appendChild(o);
    }
  });
  if(jump.options.length <= 1){ jump.style.display = "none"; }
  jump.addEventListener("change", function(){
    var v = parseInt(jump.value, 10);
    if(v >= 0) goTo(v);
    jump.value = "-1";
  });

  function update(animate){
    track.classList.toggle("animate", animate !== false);
    track.style.transform = "translateX(-" + (idx*100) + "%)";
    bar.style.width = pages.length > 1 ? ((idx/(pages.length-1))*100) + "%" : "100%";
    count.textContent = (idx+1) + " / " + pages.length;
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === pages.length - 1;
  }
  function goTo(i){
    idx = Math.max(0, Math.min(pages.length-1, i));
    update(true);
  }

  prevBtn.addEventListener("click", function(){ goTo(idx-1); });
  nextBtn.addEventListener("click", function(){ goTo(idx+1); });

  /* ---- swipe: follows the finger, and yields to inner scrolling ----
     Axis is locked on the first real movement. A horizontal drag that starts
     inside something scrollable horizontally (a wide table, a code block)
     scrolls that instead of turning the page, until it hits its own edge.  */
  var startX=null, startY=null, axis=null, dragging=false, width=1, hScroller=null;

  function horizontalScroller(el){
    while(el && el !== track){
      if(el.scrollWidth > el.clientWidth + 1){
        var s = getComputedStyle(el).overflowX;
        if(s === "auto" || s === "scroll") return el;
      }
      el = el.parentElement;
    }
    return null;
  }

  vp.addEventListener("touchstart", function(e){
    if(e.touches.length !== 1) return;
    startX = e.touches[0].clientX; startY = e.touches[0].clientY;
    axis = null; dragging = true;
    width = vp.clientWidth || 1;
    hScroller = horizontalScroller(e.target);
  }, {passive:true});

  vp.addEventListener("touchmove", function(e){
    if(!dragging || startX === null || e.touches.length !== 1) return;
    var dx = e.touches[0].clientX - startX;
    var dy = e.touches[0].clientY - startY;

    if(axis === null){
      if(Math.abs(dx) < 8 && Math.abs(dy) < 8) return;
      axis = Math.abs(dx) > Math.abs(dy) ? "x" : "y";
      if(axis === "x" && hScroller){
        // let the inner element scroll unless it's already at the relevant edge
        var atStart = hScroller.scrollLeft <= 0;
        var atEnd = hScroller.scrollLeft + hScroller.clientWidth >= hScroller.scrollWidth - 1;
        if((dx > 0 && !atStart) || (dx < 0 && !atEnd)){ axis = "inner"; }
      }
    }
    if(axis !== "x") return;           // vertical / inner scrolling: hands off

    if(e.cancelable) e.preventDefault();
    var resist = ((idx === 0 && dx > 0) || (idx === pages.length-1 && dx < 0)) ? 0.32 : 1;
    track.classList.remove("animate");
    track.style.transform = "translateX(calc(-" + (idx*100) + "% + " + (dx*resist) + "px))";
  }, {passive:false});

  function endDrag(e){
    if(!dragging) return;
    dragging = false;
    if(axis !== "x"){ startX = null; axis = null; return; }
    var dx = (e.changedTouches ? e.changedTouches[0].clientX : startX) - startX;
    var threshold = Math.min(70, width * 0.18);
    if(dx <= -threshold) idx = Math.min(pages.length-1, idx+1);
    else if(dx >= threshold) idx = Math.max(0, idx-1);
    update(true);
    startX = null; axis = null;
  }
  vp.addEventListener("touchend", endDrag);
  vp.addEventListener("touchcancel", endDrag);

  // keyboard (desktop)
  vp.tabIndex = 0;
  vp.addEventListener("keydown", function(e){
    if(e.key === "ArrowRight"){ goTo(idx+1); e.preventDefault(); }
    if(e.key === "ArrowLeft"){ goTo(idx-1); e.preventDefault(); }
  });

  window.addEventListener("resize", function(){ width = vp.clientWidth || 1; update(false); });
  update(false);
})();
</script>
</div>
"""


def build_carousel_html(
    pages: list[dict], title: str, meta_line: str, theme: str = "auto"
) -> str:
    uid = f"nb{int(time.time()*1000) % 10_000_000}"
    if not pages:
        pages = [
            {
                "type": "text",
                "title": "No output",
                "content": "This notebook produced no captured output.",
                "section": None,
            }
        ]
    if theme not in ("dark", "light", "auto"):
        theme = "auto"
    html = _CAROUSEL_TEMPLATE
    html = html.replace("__UID__", uid)
    html = html.replace("__THEME__", theme)
    html = html.replace("__TITLE__", title.replace("<", "&lt;"))
    html = html.replace("__META__", meta_line.replace("<", "&lt;"))
    html = html.replace("__PAGES_JSON__", safe_json_dumps(pages))
    return html


def build_standalone_page(carousel_html: str, title: str, theme: str = "auto") -> str:
    """Wrap the carousel in a complete HTML document for opening directly in a
    browser tab. The viewport meta (with viewport-fit=cover) is the load-bearing
    part on phones - without it mobile Safari lays the page out at ~980px and
    scales it down, which is why a carousel saved as a bare fragment looks like
    a tiny panel floating in a sea of blank page."""
    safe_title = title.replace("<", "&lt;").replace("&", "&amp;")
    scheme = {"dark": "dark", "light": "light"}.get(theme, "light dark")
    return (
        "<!doctype html>\n"
        f'<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        f'<meta name="color-scheme" content="{scheme}">\n'
        f"<title>{safe_title}</title>\n"
        "<style>\n"
        "  html,body{height:100%;margin:0;padding:0;overflow:hidden;background:#0f1115;}\n"
        f"  @media (prefers-color-scheme:light){{ html,body{{background:{'#0f1115' if theme == 'dark' else '#f5f6f8'};}} }}\n"
        "</style>\n</head>\n<body>\n"
        f"{carousel_html}\n"
        "</body>\n</html>\n"
    )


def wrap_as_iframe(html: str, height_px: int = 640) -> str:
    """Sandbox the self-contained carousel in an iframe so its CSS/JS never
    collides with the host page. Inside the iframe the carousel's 100dvh
    resolves to the iframe's own height, so the same markup serves both the
    embedded panel and the full-screen standalone page."""
    doc = build_standalone_page(html, "Notebook output")
    srcdoc = doc.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{srcdoc}" style="width:100%;height:{height_px}px;'
        f'border:none;display:block;border-radius:12px;" loading="lazy"></iframe>'
    )


# ============================================================================
# Execution (against a remote kernel over the Jupyter Server WebSocket API)
# ============================================================================


async def _execute_one_cell(
    ws, session_id: str, source: str, cell: dict, timeout: int
) -> bool:
    """Sends one execute_request and collects iopub outputs into cell["outputs"]
    until that request's kernel goes idle. Returns True if the cell errored."""
    msg_id = uuid.uuid4().hex
    msg = {
        "header": {
            "msg_id": msg_id,
            "username": "notebook-lens",
            "session": session_id,
            "msg_type": "execute_request",
            "version": "5.3",
            "date": datetime.now(timezone.utc).isoformat(),
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": source,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "buffers": [],
        "channel": "shell",
    }
    await ws.send(json.dumps(msg))

    had_error = False

    async def _drain():
        nonlocal had_error
        while True:
            raw = await ws.recv()
            data = json.loads(raw)
            if data.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            if data.get("channel") != "iopub":
                continue
            mtype = data.get("msg_type") or data.get("header", {}).get("msg_type")
            content = data.get("content", {})
            if mtype == "stream":
                cell["outputs"].append(
                    {
                        "output_type": "stream",
                        "name": content.get("name", "stdout"),
                        "text": content.get("text", ""),
                    }
                )
            elif mtype == "display_data":
                cell["outputs"].append(
                    {
                        "output_type": "display_data",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                    }
                )
            elif mtype == "execute_result":
                cell["execution_count"] = content.get("execution_count")
                cell["outputs"].append(
                    {
                        "output_type": "execute_result",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                        "execution_count": content.get("execution_count"),
                    }
                )
            elif mtype == "error":
                had_error = True
                cell["outputs"].append(
                    {
                        "output_type": "error",
                        "ename": content.get("ename", ""),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    }
                )
            elif mtype == "clear_output":
                if not content.get("wait"):
                    cell["outputs"] = []
            elif mtype == "status" and content.get("execution_state") == "idle":
                return

    await asyncio.wait_for(_drain(), timeout=timeout)
    return had_error


async def execute_notebook_remote(
    client: "JupyterClient",
    session,
    nb,
    kernel_name: str,
    cell_timeout: int,
    kernel_start_timeout: int,
    progress_cb=None,
):
    """Starts a real kernel on your Jupyter server, runs each code cell over
    its WebSocket channel, and always tears the kernel down afterward - same
    stop-on-first-error-but-keep-partial-results behavior as before, just
    executed where your packages actually live instead of locally."""
    ok = True
    err_msg = None
    start = time.time()
    code_cells = [i for i, c in enumerate(nb.cells) if c.get("cell_type") == "code"]
    total = len(code_cells)

    try:
        kernel_info = await asyncio.wait_for(
            client.start_kernel(session, kernel_name), timeout=kernel_start_timeout
        )
    except Exception as e:
        return (
            nb,
            0.0,
            False,
            f'Couldn\'t start a "{kernel_name}" kernel on the Jupyter server: {e}',
        )

    kernel_id = kernel_info["id"]
    session_id = uuid.uuid4().hex
    ws_url = client.kernel_ws_url(kernel_id, session_id)

    try:
        async with websockets.connect(
            ws_url, open_timeout=kernel_start_timeout, max_size=None
        ) as ws:
            for n, i in enumerate(code_cells):
                if progress_cb:
                    await progress_cb(n + 1, total, i)
                cell = nb.cells[i]
                cell["outputs"] = []
                cell["execution_count"] = None
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                try:
                    had_error = await _execute_one_cell(
                        ws, session_id, source, cell, cell_timeout
                    )
                except asyncio.TimeoutError:
                    ok = False
                    err_msg = f"Cell {i} timed out after {cell_timeout}s waiting on the kernel"
                    break
                if had_error:
                    ok = False
                    err_msg = f"Cell {i} raised an error (execution stopped there)"
                    break
    except Exception as e:
        ok = False
        err_msg = f"Lost connection to the Jupyter kernel: {e}"
    finally:
        await client.delete_kernel(session, kernel_id)

    duration = time.time() - start
    return nb, duration, ok, err_msg


# ============================================================================
# Open WebUI Tool
# ============================================================================


class Tools:
    class Valves(BaseModel):
        JUPYTER_BASE_URL: str = Field(
            default="http://jupyter:8888",
            description="Base URL of your Jupyter Server, reachable from wherever this tool's "
            "Python code runs (a Docker service name/port if it's a sibling "
            "container, not necessarily localhost). Same server your JupyterLab "
            "integration already talks to.",
        )
        JUPYTER_TOKEN: str = Field(
            default="",
            description="Jupyter Server auth token (the same one used to open JupyterLab in a "
            "browser, or from `jupyter server list`). Leave empty only if the "
            "server has auth disabled.",
        )
        VERIFY_SSL: bool = Field(
            default=True,
            description="Verify TLS certificates if JUPYTER_BASE_URL is https. Turn off only "
            "for a self-signed cert on a trusted internal server.",
        )
        NOTEBOOKS_PATH: str = Field(
            default="",
            description="Folder to scan for notebooks, as a path RELATIVE TO THE JUPYTER "
            "SERVER'S OWN CONTENT ROOT - not a host filesystem path. Leave empty "
            "to scan from that server's root.",
        )
        EXTRA_PATHS: str = Field(
            default="",
            description="Comma-separated list of additional content-root-relative folders to scan.",
        )
        EXECUTED_PATH: str = Field(
            default="",
            description="Content-root-relative folder to save timestamped executed copies "
            "into, if SAVE_EXECUTED_COPY is on. Defaults to the notebook's own "
            "folder + /executed. (Jupyter's Contents API rejects dot-prefixed "
            "directory names, so this can't be a hidden folder.)",
        )
        SAVE_EXECUTED_COPY: bool = Field(
            default=True,
            description="Save a timestamped copy of the notebook with fresh outputs back to "
            "the Jupyter server after each run.",
        )
        JUPYTER_KERNEL_NAME: str = Field(
            default="python3",
            description="Kernel name to start when a notebook doesn't specify one in its metadata.",
        )
        CELL_TIMEOUT_SECONDS: int = Field(
            default=180,
            description="Max seconds any single cell may run before the run is aborted.",
        )
        KERNEL_START_TIMEOUT_SECONDS: int = Field(
            default=30,
            description="Max seconds to wait for a new kernel to start and connect.",
        )
        HISTORY_FILE: str = Field(
            default="/tmp/notebook_lens_history.json",
            description="Local path (in this tool's own environment) for the small run-history "
            "log used by list_notebooks - unrelated to where notebooks/kernels live.",
        )
        INCLUDE_MARKDOWN_PAGES: bool = Field(
            default=True,
            description="Include markdown cells with real body text in the carousel (in "
            "section mode they appear at the top of / inside their section's page; "
            "in cell mode as their own pages). Section headings always label pages "
            "regardless.",
        )
        PAGE_MODE: str = Field(
            default="section",
            description='Carousel layout. "section" = one page per heading section: the '
            "heading, notes, code, and every output between that heading and the next "
            "are stacked on one page you scroll down (like collapsing a heading with "
            'the caret in Jupyter). "cell" = the original layout, one page per cell '
            "output.",
        )
        SECTION_HEADING_LEVEL: int = Field(
            default=2,
            description="Section mode only: a markdown cell whose first line is a heading at "
            "this level or higher starts a new page (1 = only '#', 2 = '#' and '##', "
            "3 = down to '###', ...). Deeper headings stay inline on the page. Raise "
            "it for finer pages, lower it for fewer, longer ones.",
        )
        CODE_MODE: str = Field(
            default="collapsed",
            description='Section mode only: "collapsed" folds every cell\'s code behind a '
            '"view code" button, "expanded" shows code above its output (cells with no '
            'output, like imports, stay folded), "hidden" leaves code out entirely.',
        )
        THEME: str = Field(
            default="auto",
            description="Carousel theme: auto (follow the viewing device's light/dark setting), "
            "dark, or light. Auto means the same saved page renders dark on a phone "
            "in dark mode and light on one in light mode.",
        )
        PANEL_HEIGHT_PX: int = Field(
            default=640,
            description="Height in pixels of the rendered carousel panel.",
        )
        PLOTLY_CDN_URL: str = Field(
            default="",
            description="Optional plotly.js CDN URL to enable interactive Plotly pages. "
            "Left empty by default to keep the panel fully self-contained/offline-safe; "
            "when empty, Plotly outputs fall back to their static PNG if the notebook "
            "produced one.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # -- helpers -------------------------------------------------------

    def _roots(self) -> list:
        roots = [self.valves.NOTEBOOKS_PATH.strip("/")]
        for p in self.valves.EXTRA_PATHS.split(","):
            p = p.strip().strip("/")
            if p:
                roots.append(p)
        seen = set()
        out = []
        for r in roots:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def _history_path(self) -> Path:
        return Path(self.valves.HISTORY_FILE).expanduser()

    def _client(self) -> "JupyterClient":
        return JupyterClient(
            self.valves.JUPYTER_BASE_URL,
            self.valves.JUPYTER_TOKEN,
            self.valves.VERIFY_SSL,
        )

    def _conn_error(self, e: Exception) -> str:
        return (
            f"Couldn't reach the Jupyter server at {self.valves.JUPYTER_BASE_URL}: {e}. "
            f"Check JUPYTER_BASE_URL is correct and reachable from wherever this tool's code "
            f"runs, and that JUPYTER_TOKEN is set if the server requires auth."
        )

    # -- tool methods ----------------------------------------------------

    async def list_notebooks(self) -> str:
        """
        List every Jupyter notebook (.ipynb) found on the configured Jupyter
        server, with when each was last modified and, if it's been run
        through this tool before, when it last ran and how long it took.

        Use this to find out what notebooks exist and to get the exact name
        to pass to run_notebook. Read the results back to the user naturally
        (this works fine in voice mode) - don't just dump the raw list.

        If the notebook the user wants isn't in this list, do not go looking
        for it yourself (searching other knowledge bases, chat history,
        memory, or the raw filesystem) and do not read/re-execute its code
        through a different tool. Tell the user it isn't in a folder this
        tool can see on that Jupyter server, and that its folder needs to be
        added to the configuration - this tool is the only supported way to
        execute a notebook here, by design.
        """
        history = load_history(self._history_path())
        client = self._client()
        try:
            async with aiohttp.ClientSession() as session:
                notebooks, problems = await discover_notebooks_remote(
                    client, session, self._roots(), history
                )
        except Exception as e:
            return self._conn_error(e)

        if not notebooks:
            diagnostics = (
                "\n".join(f"- {p}" for p in problems)
                if problems
                else (
                    "- The configured folder(s) exist and were read successfully, but contain no .ipynb files."
                )
            )
            return (
                f"No notebooks found via {self.valves.JUPYTER_BASE_URL}.\n{diagnostics}\n\n"
                f"Relay this diagnosis to the user plainly so they can fix NOTEBOOKS_PATH/EXTRA_PATHS "
                f"(remember: these are paths relative to the Jupyter server's own content root, not "
                f'host filesystem paths) - don\'t just say "no notebooks found."'
            )

        lines = [f"Found {len(notebooks)} notebook(s):"]
        for nb in notebooks:
            entry = f"- {nb['name']} ({nb['path']})"
            if nb.get("last_modified"):
                try:
                    mod_dt = datetime.fromisoformat(
                        nb["last_modified"].replace("Z", "+00:00")
                    )
                    entry += f" - modified {human_ago(mod_dt)}"
                except Exception:
                    pass
            if nb.get("last_run"):
                try:
                    last_run_dt = datetime.fromisoformat(nb["last_run"])
                    entry += f", last run {human_ago(last_run_dt)}"
                    if nb.get("last_duration"):
                        entry += f" (took {human_duration(nb['last_duration'])})"
                    if nb.get("last_ok") is False:
                        entry += " [last run had an error]"
                except Exception:
                    pass
            else:
                entry += ", never run through this tool"
            lines.append(entry)
        lines.append(
            "\nTo run one, confirm the exact notebook name with the user first, "
            "then call run_notebook with confirmed=True."
        )
        return "\n".join(lines)

    async def run_notebook(
        self,
        notebook: str,
        confirmed: bool = False,
        __event_emitter__=None,
    ) -> str:
        """
        Execute a Jupyter notebook top to bottom on a fresh kernel (started on
        your actual Jupyter server, not locally) and display every output
        (plots, tables, text, video, errors) as a swipeable carousel in chat.

        IMPORTANT: never call this with confirmed=True unless the user has
        explicitly told you to run it (or explicitly confirmed) earlier in
        this conversation. If confirmed is left False, call this once first
        to look up the notebook, tell the user what you're about to run, and
        wait for them to say go ahead - then call again with confirmed=True.
        This step matters in voice mode where there's no button to click.

        If this returns "not found" or "ambiguous", stop there. Do not fall
        back to searching knowledge bases, chat history, memory, or the raw
        filesystem for the file, and do not read the notebook's source and
        re-execute its code cell-by-cell through a separate code-interpreter
        tool. Tell the user plainly that this tool can't see that notebook
        on the configured Jupyter server and that its folder needs to be
        added to the configuration - that is the fix, not working around
        this tool.

        :param notebook: Notebook name (with or without .ipynb) or content
            path, as returned by list_notebooks. Partial/fuzzy names are
            accepted if unambiguous.
        :param confirmed: Must be explicitly set True by the assistant only
            after the user has verbally/textually agreed to run it.
        """
        client = self._client()
        history = load_history(self._history_path())

        try:
            async with aiohttp.ClientSession() as session:
                notebooks, problems = await discover_notebooks_remote(
                    client, session, self._roots(), history
                )
                nb_ref, candidates = resolve_notebook_remote(notebook, notebooks)

                if nb_ref is None:
                    if candidates:
                        return (
                            f"\"{notebook}\" is ambiguous - it could mean: {', '.join(candidates)}. "
                            f"Ask the user which one, then call run_notebook again with the exact name."
                        )
                    detail = " ".join(problems) if problems else ""
                    return (
                        f'No notebook matching "{notebook}" was found via {self.valves.JUPYTER_BASE_URL}. '
                        f"{detail} Call list_notebooks to see what's available. If the user knows this "
                        f"notebook exists on that server, tell them its folder needs to be added to "
                        f"NOTEBOOKS_PATH/EXTRA_PATHS - do not try to locate or execute it through any "
                        f"other tool."
                    )

                if not confirmed:
                    return (
                        f"Found \"{nb_ref['name']}\" at {nb_ref['path']} on the Jupyter server. This "
                        f"will start a fresh kernel and run it top to bottom, which can take anywhere "
                        f"from a few seconds to a few minutes depending on what it does. Ask the user "
                        f"to confirm before proceeding, then call "
                        f"run_notebook(notebook=\"{nb_ref['name']}\", confirmed=True)."
                    )

                async def emit_status(desc: str, done: bool = False):
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": desc, "done": done},
                            }
                        )

                await emit_status(f"Loading {nb_ref['name']}...")
                data = await client.get_contents(
                    session, nb_ref["path"], content=True, type_hint="notebook"
                )
                if data is None or "content" not in data:
                    return f"{nb_ref['path']} disappeared from the Jupyter server between listing and loading it."
                try:
                    nb = nbformat.from_dict(data["content"])
                except Exception as e:
                    return f"Couldn't parse {nb_ref['path']} as a notebook: {e}"

                kernel_name = (
                    nb.metadata.get("kernelspec", {}).get("name")
                    or self.valves.JUPYTER_KERNEL_NAME
                )

                async def progress_cb(n, total, cell_i):
                    await emit_status(f"Running {nb_ref['name']}: cell {n}/{total}...")

                try:
                    nb, duration, ok, err_msg = await execute_notebook_remote(
                        client,
                        session,
                        nb,
                        kernel_name,
                        self.valves.CELL_TIMEOUT_SECONDS,
                        self.valves.KERNEL_START_TIMEOUT_SECONDS,
                        progress_cb,
                    )
                except Exception as e:
                    ok, err_msg, duration = (
                        False,
                        f"Unexpected error running the notebook: {e}",
                        0.0,
                    )

                # execute_notebook_remote appends plain dict outputs (fine for our own dict-style
                # rendering below); nbformat.writes() needs real NotebookNode objects throughout,
                # so normalize once here before anything downstream touches nb.
                nb = nbformat.from_dict(nb)

                record_run(self._history_path(), nb_ref["path"], duration, ok)

                parent = (
                    nb_ref["path"].rsplit("/", 1)[0] if "/" in nb_ref["path"] else ""
                )
                exec_dir = self.valves.EXECUTED_PATH.strip("/") or (
                    f"{parent}/executed" if parent else "executed"
                )
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                if self.valves.SAVE_EXECUTED_COPY:
                    try:
                        await _ensure_dir_remote(client, session, exec_dir)
                        nb_target_path = f"{exec_dir}/{nb_ref['name']}.{stamp}.ipynb"
                        body = {
                            "type": "notebook",
                            "format": "json",
                            "content": json.loads(nbformat.writes(nb)),
                        }
                        await client.put_contents(session, nb_target_path, body)
                    except Exception:
                        pass  # best-effort, never blocks showing results

                page_mode = (self.valves.PAGE_MODE or "section").strip().lower()
                if page_mode == "cell":
                    pages = notebook_to_pages(
                        nb, include_markdown=self.valves.INCLUDE_MARKDOWN_PAGES
                    )
                    unit = "pages"
                else:
                    pages = notebook_to_sections(
                        nb,
                        include_markdown=self.valves.INCLUDE_MARKDOWN_PAGES,
                        section_level=self.valves.SECTION_HEADING_LEVEL,
                        code_mode=(self.valves.CODE_MODE or "collapsed").strip().lower(),
                    )
                    unit = "sections"

                meta_bits = [
                    human_duration(duration),
                    f"{len(nb.cells)} cells",
                    f"{len(pages)} {unit}",
                ]
                if not ok:
                    meta_bits.append("stopped on error")
                meta_line = " - ".join(meta_bits)

                html = build_carousel_html(
                    pages,
                    title=nb_ref["name"],
                    meta_line=meta_line,
                    theme=self.valves.THEME,
                )
                panel = wrap_as_iframe(html, height_px=self.valves.PANEL_HEIGHT_PX)

                # Best-effort inline delivery - whether this actually renders depends on
                # your chat frontend supporting "message"-type emitter events; many
                # bridges/pipes only forward "status" events, so this can silently do
                # nothing. Never rely on it alone (see carousel_url below).
                if __event_emitter__:
                    await emit_status(
                        f"Ran {nb_ref['name']} in {human_duration(duration)}", done=True
                    )
                    try:
                        await __event_emitter__(
                            {"type": "message", "data": {"content": panel}}
                        )
                    except Exception:
                        pass

                # Guaranteed-to-work fallback: save the carousel as a real .html file on the
                # Jupyter server and hand back a direct link, since that only depends on the
                # Contents API (already proven to work) rather than any chat-side rendering.
                carousel_url = None
                try:
                    await _ensure_dir_remote(client, session, exec_dir)
                    html_target_path = (
                        f"{exec_dir}/{nb_ref['name']}.{stamp}.carousel.html"
                    )
                    standalone = build_standalone_page(
                        html, nb_ref["name"], self.valves.THEME
                    )
                    await client.put_contents(
                        session,
                        html_target_path,
                        {"type": "file", "format": "text", "content": standalone},
                    )
                    base = self.valves.JUPYTER_BASE_URL.rstrip("/")
                    url_path = quote(html_target_path, safe="/")
                    carousel_url = f"{base}/files/{url_path}"
                    if self.valves.JUPYTER_TOKEN:
                        carousel_url += f"?token={self.valves.JUPYTER_TOKEN}"
                except Exception:
                    pass

                flat = list(_flat_outputs(pages))
                n_images = sum(1 for p in flat if p["type"] in ("image", "image_svg"))
                n_tables = sum(1 for p in flat if p["type"] == "html")
                summary = (
                    f"Ran {nb_ref['name']} in {human_duration(duration)} ({len(nb.cells)} cells), "
                    f"producing {len(pages)} carousel {unit} ({n_images} plots, {n_tables} tables)."
                )
                if carousel_url:
                    summary += (
                        f" Share this link with the user so they can open the carousel directly - "
                        f"it always works regardless of whether it also rendered inline in this chat: {carousel_url}"
                    )
                else:
                    summary += (
                        " An inline carousel panel was attempted but couldn't also be saved as a "
                        "fallback link this time - if nothing rendered above, that's why; tell the user."
                    )
                if not ok:
                    summary += f" Execution stopped early: {err_msg}. Everything up to that point is still shown/saved."
                summary += (
                    " Do not restate specific numbers from the notebook yourself - "
                    "the carousel is the source of truth; just point the user to it."
                )
                return summary
        except aiohttp.ClientError as e:
            return self._conn_error(e)


async def _ensure_dir_remote(client: "JupyterClient", session, path: str) -> None:
    """Creates each segment of a content-root-relative folder path that
    doesn't already exist, one level at a time (the Contents API can't
    create nested folders in a single call)."""
    if not path:
        return
    segments = [s for s in path.split("/") if s]
    built = ""
    for seg in segments:
        built = f"{built}/{seg}" if built else seg
        existing = await client.get_contents(session, built, content=False)
        if existing is None:
            await client.put_contents(session, built, {"type": "directory"})
