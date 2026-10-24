"""
title: Diagram Design
author: jtalbott22
version: 0.2.0
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
#
# 0.2.0 — inline viewer: zoom and pan like an image, full screen, and
#         PNG / SVG / HTML downloads. Only the chat copy gets the viewer.
# ---------------------------------------------------------------------------

import asyncio
import difflib
import html as _html
import json
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


# ---------------------------------------------------------------------------
# Inline viewer. Injected only into the copy shown in chat; the saved file
# stays exactly what the model wrote (and what the self-check passed).
#
#   - each diagram <svg role="img"> fits the panel width, and zooms and pans
#     like an image by rewriting its viewBox (vector-sharp at any zoom):
#     pinch or Ctrl/⌘+scroll, or click the diagram and then scroll;
#     drag to pan; double-click to zoom in, again to fit; + − 0 keys
#   - fullscreen, plus downloads: PNG (2×), standalone SVG (page CSS and
#     Google Fonts inlined), and the original self-contained HTML
#   - Open WebUI's embed iframe is not same-origin, so the page reports its
#     own height with postMessage({type: "iframe:height"})
# ---------------------------------------------------------------------------

VIEWER_CSS = """
html, body { min-height: 0 !important; height: auto !important; }
.ddv-wrap { position: relative; overflow: hidden; outline: none;
  touch-action: none; cursor: grab; border-radius: 6px; }
.ddv-wrap.ddv-drag { cursor: grabbing; }
.ddv-wrap.ddv-active { box-shadow: inset 0 0 0 1px rgba(127,127,127,.35); }
.ddv-wrap > svg.ddv-svg { width: 100% !important; min-width: 0 !important;
  height: auto !important; display: block; user-select: none; -webkit-user-select: none; }
.ddv-bar { position: absolute; top: 8px; right: 8px; z-index: 5; display: flex;
  gap: 2px; padding: 3px; border-radius: 6px; cursor: default;
  font: 500 11px/1 system-ui, -apple-system, sans-serif;
  background: rgba(250,250,250,.92); color: #2d3142;
  border: 1px solid rgba(45,49,66,.16); opacity: .35; transition: opacity .15s; }
.ddv-wrap:hover .ddv-bar, .ddv-wrap:focus-within .ddv-bar,
.ddv-wrap:fullscreen .ddv-bar { opacity: 1; }
@media (hover: none) { .ddv-bar { opacity: 1; } }
.ddv-bar button { all: unset; cursor: pointer; padding: 5px 7px; border-radius: 4px;
  font: inherit; color: inherit; white-space: nowrap; }
.ddv-bar button:hover, .ddv-bar button:focus-visible { background: rgba(45,49,66,.09); }
.ddv-bar .ddv-zoom { min-width: 34px; text-align: center; font-variant-numeric: tabular-nums; }
.ddv-bar .ddv-sep { width: 1px; margin: 3px 2px; background: rgba(45,49,66,.16); }
.ddv-hint { position: absolute; left: 50%; bottom: 8px; transform: translateX(-50%);
  z-index: 5; pointer-events: none; padding: 4px 8px; border-radius: 4px;
  font: 11px/1.3 system-ui, sans-serif; background: rgba(45,49,66,.82); color: #fff;
  opacity: 0; transition: opacity .2s; white-space: nowrap; }
.ddv-hint.ddv-show { opacity: 1; }
.ddv-wrap:fullscreen { display: flex; align-items: center; justify-content: center;
  background: var(--ddv-bg, #f5f5f5); border-radius: 0; }
.ddv-wrap:fullscreen > svg.ddv-svg { width: 100% !important; height: 100% !important; }
@media print { .ddv-bar, .ddv-hint { display: none !important; } }
"""

VIEWER_JS = r"""
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var MAX = 8;
  var SLUG = document.getElementById("ddv-source").dataset.slug || "diagram";
  var fontCache = null;

  function postHeight() {
    var h = Math.ceil(Math.max(document.documentElement.scrollHeight,
                               document.body ? document.body.scrollHeight : 0));
    parent.postMessage({ type: "iframe:height", height: h }, "*");
  }

  function save(blob, name) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  }

  function blobToDataURL(blob) {
    return new Promise(function (ok, fail) {
      var r = new FileReader();
      r.onload = function () { ok(r.result); };
      r.onerror = fail;
      r.readAsDataURL(blob);
    });
  }

  // Google Fonts CSS with the Latin font files inlined as data: URIs, so the
  // exported SVG/PNG keeps the diagram's typography outside the page.
  async function fontCSS() {
    if (fontCache !== null) return fontCache;
    var css = "";
    var links = document.querySelectorAll('link[rel="stylesheet"][href*="fonts.googleapis.com"]');
    for (var i = 0; i < links.length; i++) {
      try { css += (await (await fetch(links[i].href)).text()) + "\n"; } catch (e) {}
    }
    var latin = css.match(/\/\*\s*latin\s*\*\/\s*@font-face\s*\{[^}]*\}/g);
    if (latin) css = latin.map(function (b) { return b.replace(/^\/\*[^*]*\*\/\s*/, ""); }).join("\n");
    var urls = Array.from(new Set((css.match(/url\((https:[^)]+)\)/g) || [])
      .map(function (u) { return u.slice(4, -1); })));
    for (var j = 0; j < urls.length; j++) {
      try {
        var data = await blobToDataURL(await (await fetch(urls[j])).blob());
        css = css.split(urls[j]).join(data);
      } catch (e) {}
    }
    return (fontCache = css);
  }

  async function standaloneSVG(v) {
    var c = v.svg.cloneNode(true);
    c.removeAttribute("style");
    c.removeAttribute("class");
    c.setAttribute("xmlns", NS);
    c.setAttribute("viewBox", v.base.join(" "));
    c.setAttribute("width", v.base[2]);
    c.setAttribute("height", v.base[3]);
    var pageCSS = Array.from(document.querySelectorAll("style:not(.ddv-style)"))
      .map(function (s) { return s.textContent; }).join("\n");
    var st = document.createElementNS(NS, "style");
    st.textContent = (await fontCSS()) + "\n" + pageCSS +
      "\n:root{width:" + v.base[2] + "px!important;height:" + v.base[3] +
      "px!important;min-width:0!important;max-width:none!important}";
    var first = c.firstElementChild;
    if (first && first.tagName.toLowerCase() === "title") first.after(st);
    else c.prepend(st);
    return new XMLSerializer().serializeToString(c);
  }

  function pageBackground() {
    var bg = getComputedStyle(document.body).backgroundColor;
    if (!bg || bg === "transparent" || /rgba\(0, 0, 0, 0\)/.test(bg))
      bg = getComputedStyle(document.documentElement).backgroundColor;
    return (!bg || /rgba\(0, 0, 0, 0\)/.test(bg)) ? "#ffffff" : bg;
  }

  function setup(svg, idx) {
    var vb = (svg.getAttribute("viewBox") || "").trim().split(/[\s,]+/).map(Number);
    if (vb.length !== 4 || vb.some(isNaN) || !vb[2] || !vb[3]) {
      var bb = svg.getBBox ? svg.getBBox() : null;
      if (!bb || !bb.width) return;
      vb = [bb.x, bb.y, bb.width, bb.height];
      svg.setAttribute("viewBox", vb.join(" "));
    }
    var v = { svg: svg, base: vb, s: 1, x: vb[0], y: vb[1] };
    // Percentages resolve against the *current* viewBox, so a full-bleed
    // background rect (width="100%") would shrink on zoom. Pin them to the
    // unzoomed size first. Paint servers, masks etc. keep their own units.
    var dims = { x: [0, 2], y: [1, 3], width: [-1, 2], height: [-1, 3] };
    svg.querySelectorAll('[x$="%"],[y$="%"],[width$="%"],[height$="%"]').forEach(function (el) {
      if (el.parentElement.closest("pattern,mask,clipPath,filter,marker,symbol,svg") !== svg) return;
      Object.keys(dims).forEach(function (a) {
        var val = el.getAttribute(a);
        if (!val || val.slice(-1) !== "%") return;
        var d = dims[a], n = parseFloat(val) / 100 * vb[d[1]] + (d[0] >= 0 ? vb[d[0]] : 0);
        if (!isNaN(n)) el.setAttribute(a, n);
      });
    });
    var name = SLUG + (idx ? "-" + (idx + 1) : "");

    var wrap = document.createElement("div");
    wrap.className = "ddv-wrap";
    wrap.tabIndex = 0;
    wrap.setAttribute("aria-label", "Diagram viewer. Plus and minus keys zoom, 0 fits.");
    wrap.style.setProperty("--ddv-bg", pageBackground());
    svg.parentNode.insertBefore(wrap, svg);
    wrap.appendChild(svg);
    svg.classList.add("ddv-svg");
    for (var p = wrap.parentElement; p && p !== document.body; p = p.parentElement) {
      var cs = getComputedStyle(p);
      if (cs.overflowX === "auto" || cs.overflowX === "scroll") p.style.overflowX = "visible";
    }

    var bar = document.createElement("div");
    bar.className = "ddv-bar";
    var hint = document.createElement("div");
    hint.className = "ddv-hint";
    hint.textContent = "Scroll to zoom · drag to pan · double-click to zoom";
    wrap.appendChild(bar);
    wrap.appendChild(hint);

    function btn(label, title, fn) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = label; b.title = title;
      b.setAttribute("aria-label", title);
      b.addEventListener("click", function (e) { e.stopPropagation(); fn(); });
      b.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      b.addEventListener("dblclick", function (e) { e.stopPropagation(); });
      bar.appendChild(b);
      return b;
    }
    function sep() { var s = document.createElement("span"); s.className = "ddv-sep"; bar.appendChild(s); }

    function apply() {
      var w = v.base[2] / v.s, h = v.base[3] / v.s;
      v.x = Math.min(Math.max(v.x, v.base[0]), v.base[0] + v.base[2] - w);
      v.y = Math.min(Math.max(v.y, v.base[1]), v.base[1] + v.base[3] - h);
      svg.setAttribute("viewBox", [v.x, v.y, w, h].join(" "));
      zoomBtn.textContent = Math.round(v.s * 100) + "%";
    }
    // Zoom to scale s keeping the point at fraction (fx, fy) of the box fixed.
    function zoomTo(s, fx, fy) {
      s = Math.min(Math.max(s, 1), MAX);
      if (fx === undefined) { fx = 0.5; fy = 0.5; }
      var w = v.base[2] / v.s, h = v.base[3] / v.s;
      var px = v.x + fx * w, py = v.y + fy * h;
      v.s = s;
      v.x = px - fx * (v.base[2] / s);
      v.y = py - fy * (v.base[3] / s);
      apply();
    }
    // Map a client point to fractions of the drawn diagram area, allowing
    // for the letterboxing preserveAspectRatio adds in fullscreen.
    function frac(cx, cy) {
      var r = svg.getBoundingClientRect();
      var k = Math.min(r.width / v.base[2], r.height / v.base[3]);
      var dw = v.base[2] * k, dh = v.base[3] * k;
      var ox = r.left + (r.width - dw) / 2, oy = r.top + (r.height - dh) / 2;
      return [Math.min(Math.max((cx - ox) / dw, 0), 1),
              Math.min(Math.max((cy - oy) / dh, 0), 1), dw, dh];
    }

    btn("−", "Zoom out", function () { zoomTo(v.s / 1.25); });
    var zoomBtn = btn("100%", "Fit to panel", function () { zoomTo(1); });
    zoomBtn.classList.add("ddv-zoom");
    btn("+", "Zoom in", function () { zoomTo(v.s * 1.25); });
    if (document.fullscreenEnabled && wrap.requestFullscreen) {
      sep();
      btn("⛶", "Full screen", function () {
        if (document.fullscreenElement) document.exitFullscreen();
        else wrap.requestFullscreen().catch(function () {});
      });
    }
    sep();
    btn("PNG", "Download PNG (2×)", async function () {
      try {
        var str = await standaloneSVG(v);
        var img = new Image();
        img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(str);
        await img.decode();
        var scale = 2, cv = document.createElement("canvas");
        cv.width = Math.round(v.base[2] * scale);
        cv.height = Math.round(v.base[3] * scale);
        var ctx = cv.getContext("2d");
        ctx.fillStyle = pageBackground();
        ctx.fillRect(0, 0, cv.width, cv.height);
        ctx.drawImage(img, 0, 0, cv.width, cv.height);
        cv.toBlob(function (b) {
          if (b) save(b, name + ".png");
          else flash("PNG export failed — try SVG");
        }, "image/png");
      } catch (e) { flash("PNG export failed — try SVG"); }
    });
    btn("SVG", "Download SVG", async function () {
      save(new Blob([await standaloneSVG(v)], { type: "image/svg+xml" }), name + ".svg");
    });
    btn("HTML", "Download the self-contained HTML file", function () {
      var src = JSON.parse(document.getElementById("ddv-source").textContent);
      save(new Blob([src], { type: "text/html" }), SLUG + ".html");
    });

    var hintTimer;
    function flash(text) {
      hint.textContent = text;
      hint.classList.add("ddv-show");
      clearTimeout(hintTimer);
      hintTimer = setTimeout(function () { hint.classList.remove("ddv-show"); }, 1800);
    }

    // Plain scrolling keeps scrolling the chat until the diagram is clicked;
    // pinch (ctrlKey) and Ctrl/⌘+scroll always zoom.
    var active = false;
    function setActive(on) { active = on; wrap.classList.toggle("ddv-active", on); }
    wrap.addEventListener("pointerleave", function () { setActive(false); });
    wrap.addEventListener("blur", function () { setActive(false); });
    wrap.addEventListener("wheel", function (e) {
      if (!(active || e.ctrlKey || e.metaKey)) {
        if (Math.abs(e.deltaY) > 2) flash("Click the diagram, then scroll to zoom");
        return;
      }
      e.preventDefault();
      var f = frac(e.clientX, e.clientY);
      var dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      zoomTo(v.s * Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0025)), f[0], f[1]);
    }, { passive: false });

    var pts = new Map(), last = null, pinch = null, moved = 0;
    wrap.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      wrap.focus({ preventScroll: true });
      setActive(true);
      wrap.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      moved = 0;
      if (pts.size === 1) { last = [e.clientX, e.clientY]; wrap.classList.add("ddv-drag"); }
      if (pts.size === 2) {
        var a = Array.from(pts.values());
        var mid = [(a[0][0] + a[1][0]) / 2, (a[0][1] + a[1][1]) / 2];
        var f = frac(mid[0], mid[1]);
        pinch = { d: Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), s: v.s, f: f };
      }
    });
    wrap.addEventListener("pointermove", function (e) {
      if (!pts.has(e.pointerId)) return;
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 2 && pinch) {
        var a = Array.from(pts.values());
        var d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]);
        zoomTo(pinch.s * d / pinch.d, pinch.f[0], pinch.f[1]);
      } else if (pts.size === 1 && last) {
        var f = frac(e.clientX, e.clientY);
        var dx = e.clientX - last[0], dy = e.clientY - last[1];
        moved += Math.abs(dx) + Math.abs(dy);
        v.x -= dx / f[2] * (v.base[2] / v.s);
        v.y -= dy / f[3] * (v.base[3] / v.s);
        last = [e.clientX, e.clientY];
        apply();
      }
    });
    function end(e) {
      pts.delete(e.pointerId);
      if (pts.size < 2) pinch = null;
      if (pts.size === 1) last = Array.from(pts.values())[0];
      if (pts.size === 0) { last = null; wrap.classList.remove("ddv-drag"); }
    }
    wrap.addEventListener("pointerup", end);
    wrap.addEventListener("pointercancel", end);
    wrap.addEventListener("dblclick", function (e) {
      var f = frac(e.clientX, e.clientY);
      if (v.s > 1.01) zoomTo(1);
      else zoomTo(2.5, f[0], f[1]);
    });
    wrap.addEventListener("keydown", function (e) {
      if (e.key === "+" || e.key === "=") { zoomTo(v.s * 1.25); e.preventDefault(); }
      else if (e.key === "-" || e.key === "_") { zoomTo(v.s / 1.25); e.preventDefault(); }
      else if (e.key === "0") { zoomTo(1); e.preventDefault(); }
    });
    document.addEventListener("fullscreenchange", postHeight);
    apply();
  }

  function init() {
    var svgs = Array.from(document.querySelectorAll('svg[role="img"]'))
      .filter(function (s) { return !s.parentElement.closest("svg"); });
    if (!svgs.length) {
      svgs = Array.from(document.querySelectorAll("svg"))
        .filter(function (s) { return !s.parentElement.closest("svg"); })
        .sort(function (a, b) {
          var ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
          return rb.width * rb.height - ra.width * ra.height;
        }).slice(0, 1);
    }
    svgs.forEach(setup);
    postHeight();
    if (window.ResizeObserver) new ResizeObserver(postHeight).observe(document.body);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(postHeight);
    window.addEventListener("load", postHeight);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
"""


def _with_viewer(doc: str, slug: str) -> str:
    """Return the display copy of `doc` with the zoom/download viewer added.
    The original document rides along as JSON for the HTML download."""
    source = json.dumps(doc).replace("<", "\\u003c")
    tail = (
        f'<style class="ddv-style">{VIEWER_CSS}</style>'
        f'<script type="application/json" id="ddv-source" '
        f'data-slug="{_html.escape(slug)}">{source}</script>'
        f"<script>{VIEWER_JS}</script>"
    )
    i = doc.lower().rfind("</body>")
    return doc[:i] + tail + doc[i:] if i != -1 else doc + tail


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
        VIEWER: bool = Field(
            default=True,
            description="Show diagrams in chat with zoom, pan, full screen "
            "and PNG/SVG/HTML download controls. The saved file is "
            "never modified.",
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
            HTMLResponse(
                content=_with_viewer(doc, slug) if self.valves.VIEWER else doc,
                headers={"Content-Disposition": "inline"},
            ),
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
