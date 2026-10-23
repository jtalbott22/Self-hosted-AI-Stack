"""
title: Diagram Design
author: jtalbott22
version: 0.1.0
license: MIT
description: Gives the model the diagram-design skill (cathrynlavery/diagram-design, MIT) as tools. The model reads the design rules, the reference for one of 45 diagram types and a worked example, writes one self-contained HTML + SVG file, and the tool checks it, saves it, and renders it inline in chat.
required_open_webui_version: 0.6.0
"""

# ---------------------------------------------------------------------------
# Diagram Design
#
#   prompt -> model calls diagram_guide()            (rules + type list)
#          -> model calls diagram_reference(type)    (type reference + example)
#          -> model writes ONE self-contained .html file
#          -> model calls render_diagram(html)
#          -> Python: optional self_check.py, save to disk,
#             HTMLResponse(Content-Disposition: inline) => inline iframe panel
#
# The skill files come from a local checkout of the upstream repo, mounted
# into the Open WebUI container (see README → Diagram Design). Without a
# checkout they are fetched read-only from raw.githubusercontent.com at a
# pinned ref and cached in memory. The upstream self-check script is only run
# from a local checkout, never from downloaded code.
# ---------------------------------------------------------------------------

import asyncio
import difflib
import html as _html
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

UPSTREAM_REPO = "cathrynlavery/diagram-design"
SKILL_SUBPATH = "skills/diagram-design"

# SKILL.md sections handed to the model by diagram_guide(). The rest (setup
# gate, import, onboarding, export) assumes a coding agent with a shell.
GUIDE_SECTIONS = ("1.", "4.", "6.", "7.", "12.")
STYLE_SECTIONS = ("### Semantic roles", "## Typography", "## Node type → treatment")

ALIASES = {
    "state-machine": "state",
    "erd": "er",
    "entity-relationship": "er",
    "data-model": "er",
    "spider": "radar",
    "funnel": "pyramid",
    "org": "org-chart",
    "flow": "flowchart",
    "flow-chart": "flowchart",
    "layer-stack": "layers",
    "stack": "layers",
    "schema": "db-schema",
    "database-schema": "db-schema",
    "class": "uml-class",
    "uml": "uml-class",
    "slopegraph": "line",
    "ridgeline": "line",
    "streamgraph": "line",
    "bump": "line",
    "bubble": "scatter",
    "beeswarm": "scatter",
    "marimekko": "treemap",
    "dumbbell": "bar",
    "flywheel": "loop",
    "cycle": "loop",
    "user-journey": "journey",
    "blueprint": "service-blueprint",
    "dependencies": "dependency",
    "dependency-graph": "dependency",
    "root-cause": "fishbone",
    "ishikawa": "fishbone",
    "board": "kanban",
}

VARIANT_SUFFIX = {"light": "", "dark": "-dark", "full": "-full"}

WORKFLOW = """\
# Diagram Design — how to use these tools

1. Pick the visual type from the table below. If a table or a paragraph would
   say the same thing, don't draw; say so instead.
2. In one short sentence, state the type and anything the complexity budget
   forces out, then call `diagram_reference(diagram_type)`. It returns that
   type's layout rules and a finished example file to adapt.
3. Write ONE complete, self-contained HTML document: inline CSS, inline SVG,
   Google Fonts as the only external resource, no JavaScript. Resolve every
   role placeholder such as {ink} or {accent} to the hex values in the style
   tokens below.
4. Call `render_diagram(html=..., title=...)` with the whole document. Never
   paste the HTML into the chat; the tool renders it. If it reports problems,
   fix them and call it again.
"""


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def _sections(markdown: str, wanted: tuple) -> str:
    """Return the markdown sections whose heading starts with one of `wanted`
    (heading text after the hashes, e.g. "1." or "## Typography"). A section
    runs until the next heading of the same or higher level."""
    lines = markdown.splitlines()
    out, keep, level = [], False, 0
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            depth, text = len(m.group(1)), m.group(2)
            if keep and depth <= level:
                keep = False
            if not keep:
                for w in wanted:
                    if w.startswith("#"):
                        hit = line.startswith(w)
                    else:
                        hit = depth == 2 and text.startswith(w)
                    if hit:
                        keep, level = True, depth
                        break
        if keep:
            out.append(line)
    return "\n".join(out).strip()


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:html)?\s*\n(.*?)```", text or "", re.DOTALL)
    return (m.group(1) if m else text or "").strip()


def _fault_panel(what: str, how: str) -> HTMLResponse:
    return HTMLResponse(
        content=(
            "<div style=\"font:13px/1.5 system-ui,sans-serif;padding:12px 14px;"
            "border:1px solid #d9a38f;border-radius:6px;background:#fbf3f0;"
            "color:#2d3142\"><b>Diagram Design</b> — "
            f"{_html.escape(what)}<br><span style=\"color:#4f5d75\">"
            f"{_html.escape(how)}</span></div>"
        ),
        headers={"Content-Disposition": "inline"},
    )


class Tools:
    class Valves(BaseModel):
        SKILL_DIR: str = Field(
            default="/app/backend/data/diagram-design/skills/diagram-design",
            description="Path inside the Open WebUI container to the skill "
            "folder (the one holding SKILL.md, references/, assets/, "
            "scripts/). Used first when it exists.",
        )
        GITHUB_FALLBACK: bool = Field(
            default=True,
            description="When SKILL_DIR is missing, fetch skill files "
            "read-only from raw.githubusercontent.com.",
        )
        GITHUB_REF: str = Field(
            default="main",
            description="Branch, tag, or commit of the upstream repo to fetch "
            "from when falling back to GitHub. Pin a tag or commit for "
            "repeatable output.",
        )
        OUTPUT_DIR: str = Field(
            default="/app/backend/data/diagrams",
            description="Where render_diagram saves each .html file.",
        )
        RUN_SELF_CHECK: bool = Field(
            default=True,
            description="Run the skill's scripts/self_check.py on every "
            "rendered diagram. Only runs from a local SKILL_DIR checkout.",
        )
        MAX_EXAMPLE_CHARS: int = Field(
            default=30000,
            description="Truncate example HTML handed to the model beyond "
            "this many characters. Lower it for small-context models.",
        )
        TIMEOUT: float = Field(default=10.0, description="Fetch timeout, seconds.")

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

    # -- plumbing -----------------------------------------------------------

    _client: Optional[httpx.AsyncClient] = None
    _cache: dict = {}

    def _local_dir(self) -> Optional[Path]:
        p = Path(self.valves.SKILL_DIR)
        return p if (p / "SKILL.md").is_file() else None

    def _http(self) -> httpx.AsyncClient:
        cls = type(self)
        if cls._client is None or cls._client.is_closed:
            cls._client = httpx.AsyncClient(
                timeout=self.valves.TIMEOUT, follow_redirects=True
            )
        return cls._client

    async def _read(self, rel: str) -> Optional[str]:
        """Read a file relative to the skill folder: local checkout first,
        then the pinned GitHub ref. Returns None when it doesn't exist."""
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*", rel) or ".." in rel:
            return None
        local = self._local_dir()
        if local:
            f = local / rel
            return f.read_text(encoding="utf-8") if f.is_file() else None
        if not self.valves.GITHUB_FALLBACK:
            return None
        url = (
            f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
            f"{self.valves.GITHUB_REF}/{SKILL_SUBPATH}/{rel}"
        )
        if url in self._cache:
            return self._cache[url]
        r = await self._http().get(url)
        if r.status_code == 404:
            self._cache[url] = None
            return None
        r.raise_for_status()
        self._cache[url] = r.text
        return r.text

    async def _types(self) -> list[tuple[str, str, str]]:
        """(when-to-use, name, slug) rows from SKILL.md's visual-type guide."""
        skill = await self._read("SKILL.md") or ""
        rows = re.findall(
            r"^\|\s*(.+?)\s*\|\s*\*\*(.+?)\*\*\s*\|\s*\[type-([a-z0-9-]+)\.md\]",
            skill,
            re.MULTILINE,
        )
        return [(when, name, slug) for when, name, slug in rows]

    async def _resolve(self, diagram_type: str) -> Optional[tuple[str, str]]:
        types = await self._types()
        by_slug = {slug: name for _, name, slug in types}
        by_name = {_slugify(name): slug for _, name, slug in types}
        key = _slugify(diagram_type)
        key = re.sub(r"-(diagram|chart|plot|map|graph)$", "", key) or key
        key = ALIASES.get(key, key)
        slug = key if key in by_slug else by_name.get(key)
        if not slug:
            near = difflib.get_close_matches(
                key, list(by_slug) + list(by_name), n=1, cutoff=0.6
            )
            if near:
                slug = near[0] if near[0] in by_slug else by_name[near[0]]
        return (slug, by_slug[slug]) if slug else None

    async def _status(self, emitter, text: str, done: bool = False) -> None:
        if emitter:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": text, "done": done, "hidden": done},
                }
            )

    def _unavailable(self) -> str:
        return (
            "The diagram-design skill files are not available. Clone "
            f"https://github.com/{UPSTREAM_REPO} so that SKILL.md sits at "
            f"{self.valves.SKILL_DIR}/SKILL.md inside the Open WebUI "
            "container, or enable GITHUB_FALLBACK in this tool's valves."
        )

    # -- tools --------------------------------------------------------------

    async def diagram_guide(self, __event_emitter__=None) -> str:
        """
        Load the diagram-design rules. Call this FIRST, once per conversation,
        whenever the user asks for any diagram, chart, schematic, flow, map, or
        visual explanation. Returns the workflow, the list of 45 diagram types
        with when to use each, the design rules, and the style tokens.
        """
        await self._status(__event_emitter__, "Loading diagram-design rules…")
        try:
            skill = await self._read("SKILL.md")
            style = await self._read("references/style-guide.md")
        except httpx.HTTPError as e:
            return f"Could not fetch the skill from GitHub: {e}. {self._unavailable()}"
        if not skill:
            return self._unavailable()
        types = await self._types()
        table = "\n".join(
            f"| `{slug}` | {name} | {when} |" for when, name, slug in types
        )
        body = _sections(skill, GUIDE_SECTIONS)
        # Relative links point into the skill folder; the model reaches those
        # files through diagram_reference, not by path.
        body = re.sub(r"\[([^\]]+)\]\((?:references|assets)/[^)]+\)", r"\1", body)
        tokens = _sections(style or "", STYLE_SECTIONS)
        await self._status(__event_emitter__, "Diagram rules loaded", done=True)
        return (
            f"{WORKFLOW}\n## Diagram types\n\n| diagram_type | Name | Use when you're showing… |\n"
            f"|---|---|---|\n{table}\n\n{body}\n\n# Style tokens\n\n{tokens}\n"
        )

    async def diagram_reference(
        self,
        diagram_type: str,
        variant: str = "light",
        include_example: bool = True,
        __event_emitter__=None,
    ) -> str:
        """
        Load the layout rules for one diagram type plus a finished example file
        to adapt. Call this after diagram_guide and before writing the HTML.
        :param diagram_type: The type slug from the diagram_guide table, e.g. "architecture", "flowchart", "sequence", "state", "er", "timeline", "quadrant", "sankey".
        :param variant: "light" (default, minimal), "dark" (minimal dark), or "full" (editorial page with summary cards).
        :param include_example: Include the example HTML file. Leave true unless context is very tight.
        """
        await self._status(__event_emitter__, f"Loading {diagram_type} reference…")
        try:
            hit = await self._resolve(diagram_type)
            if not hit:
                if not await self._read("SKILL.md"):
                    return self._unavailable()
                slugs = ", ".join(s for _, _, s in await self._types())
                return f"Unknown diagram_type '{diagram_type}'. Use one of: {slugs}."
            slug, name = hit
            ref = await self._read(f"references/type-{slug}.md") or ""
            parts = [f"# {name} (`{slug}`)\n\n{ref}"]
            if include_example:
                suffix = VARIANT_SUFFIX.get((variant or "light").lower(), "")
                example_name = f"example-{slug}{suffix}.html"
                example = await self._read(f"assets/{example_name}")
                if example is None:
                    example_name = f"template{suffix}.html"
                    example = await self._read(f"assets/{example_name}") or ""
                limit = max(2000, self.valves.MAX_EXAMPLE_CHARS)
                if len(example) > limit:
                    example = example[:limit] + "\n<!-- [example truncated] -->"
                parts.append(
                    f"## Example to adapt: `{example_name}`\n\n"
                    "Keep its structure, CSS, markers, accessibility markup and "
                    "legend strip; replace the content. Note: the example may "
                    "predate the current style tokens — the tokens from "
                    "diagram_guide win.\n\n"
                    f"```html\n{example}\n```"
                )
        except httpx.HTTPError as e:
            return f"Could not fetch the skill from GitHub: {e}. {self._unavailable()}"
        await self._status(__event_emitter__, f"{name} reference loaded", done=True)
        return "\n\n".join(parts)

    async def render_diagram(
        self,
        html: str,
        title: str = "",
        __event_emitter__=None,
    ):
        """
        Render a finished diagram inline in the chat and save it as a .html
        file. Pass the COMPLETE self-contained HTML document (doctype, head,
        inline CSS, inline SVG). Do not also paste the HTML into your reply.
        :param html: The full HTML document for the diagram.
        :param title: Short name for the diagram, used for the saved file name.
        """
        doc = _strip_fences(html)
        if "<svg" not in doc.lower():
            return (
                _fault_panel(
                    "No <svg> found in the diagram.",
                    "Pass the complete HTML document with the inline SVG.",
                ),
                {"status": "error", "error": "no <svg> element in html"},
            )
        if not re.match(r"\s*(<!doctype|<html)", doc, re.IGNORECASE):
            doc = (
                "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                f"<title>{_html.escape(title or 'Diagram')}</title></head>"
                f"<body>{doc}</body></html>"
            )
        if not title:
            m = re.search(r"<title>(.*?)</title>", doc, re.IGNORECASE | re.DOTALL)
            title = m.group(1).strip() if m else "diagram"
        slug = _slugify(title)[:60] or "diagram"

        await self._status(__event_emitter__, "Saving diagram…")
        out_dir = Path(self.valves.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}.html"
        path.write_text(doc, encoding="utf-8")

        check = "skipped"
        local = self._local_dir()
        script = local / "scripts" / "self_check.py" if local else None
        if self.valves.RUN_SELF_CHECK and script and script.is_file():
            await self._status(__event_emitter__, "Running self-check…")
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-I", str(script), str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                report = (proc.stdout + proc.stderr).strip()
                check = "passed" if proc.returncode == 0 else report[-4000:]
            except (subprocess.TimeoutExpired, OSError) as e:
                check = f"self-check could not run: {e}"
        elif self.valves.RUN_SELF_CHECK:
            check = "skipped (needs a local SKILL_DIR checkout)"

        await self._status(__event_emitter__, "Diagram rendered", done=True)
        failed = check not in ("passed",) and not check.startswith("skipped")
        return (
            HTMLResponse(content=doc, headers={"Content-Disposition": "inline"}),
            {
                "status": "rendered",
                "saved_to": str(path),
                "self_check": check,
                "instruction": (
                    "The diagram is already displayed above. Fix the self-check "
                    "problems listed and call render_diagram again."
                    if failed
                    else "The diagram is already displayed above. Reply in one "
                    "or two short sentences; do not repeat the HTML."
                ),
            },
        )
