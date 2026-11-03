"""Builds the two README architecture diagrams to the diagram-design rules."""
import html, sys
PAPER, INK, MUTED, SOFT, ACCENT, LINK = '#f5f5f5', '#2d3142', '#4f5d75', '#5a6580', '#bf4520', '#2e5aa8'
SANS, MONO = "'Geist', system-ui, sans-serif", "'Geist Mono', ui-monospace, monospace"
STYLE = {  # node type -> fill, stroke, dash, tag stroke, tag text
  'focal':    ('rgba(191,69,32,0.08)', ACCENT, None, 'rgba(191,69,32,0.50)', ACCENT),
  'backend':  ('#ffffff', INK, None, 'rgba(45,49,66,0.40)', INK),
  'store':    ('rgba(45,49,66,0.05)', MUTED, None, 'rgba(79,93,117,0.50)', MUTED),
  'external': ('rgba(45,49,66,0.03)', 'rgba(45,49,66,0.30)', None, 'rgba(45,49,66,0.22)', SOFT),
  'input':    ('rgba(79,93,117,0.10)', SOFT, None, 'rgba(90,101,128,0.40)', SOFT),
}
e = html.escape
def tw(s, px, em, track=0):  # text width budget
    return len(s) * px * (em + track)

class D:
    def __init__(s, slug, w, h): s.slug, s.w, s.h, s.zones, s.arrows, s.labels, s.nodes = slug, w, h, [], [], [], []
    def zone(s, x, y, w, h, label):
        lw = round((tw(label, 7, 0.62, 0.14) + 12) / 4) * 4
        s.zones.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>'
            f'<rect x="{x + 16}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>'
            f'<text x="{x + 16 + lw / 2}" y="{y + 13}" fill="{SOFT}" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.14em">{e(label)}</text>')
    def arrow(s, d, kind='muted', dashed=False):
        col, mk, sw = {'muted': (MUTED, 'arrow', 1.2), 'accent': (ACCENT, 'arrow-accent', 1.4), 'link': (LINK, 'arrow-link', 1.2)}[kind]
        da = ' stroke-dasharray="4,3"' if dashed else ''
        if dashed: sw = 1
        s.arrows.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}"{da} marker-end="url(#{mk})"/>')
    def label(s, cx, cy, text, kind='muted'):
        """A label centred at (cx, cy) on an opaque mask, 12px tall."""
        col = {'muted': MUTED, 'accent': ACCENT, 'link': LINK}[kind]
        w = round((tw(text, 8, 0.62, 0.06) + 10) / 4) * 4
        s.labels.append(f'<rect x="{cx - w / 2}" y="{cy - 6}" width="{w}" height="12" rx="2" fill="{PAPER}"/>'
            f'<text x="{cx}" y="{cy + 3}" fill="{col}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{e(text)}</text>')
    def node(s, x, y, w, h, kind, tag, name, subs, num=None):
        fill, stroke, dash, tstroke, tcol = STYLE[kind]
        assert tw(name, 12, 0.6) <= w - 16, ('name too wide', name)
        for sub in subs: assert tw(sub, 9, 0.62) <= w - 16, ('sublabel too wide', sub, tw(sub, 9, .62), w - 16)
        tagw = round((tw(tag, 7, 0.62, 0.08) + 10) / 4) * 4
        lines = 1 + len(subs); top = y + h / 2 - (lines * 16 - 4) / 2 + 6     # centre the text block, below the tag
        top = max(top, y + 34)
        out = (f'<g data-node="{e(name)}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>'
               f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
               f'<rect x="{x + 8}" y="{y + 8}" width="{tagw}" height="12" rx="2" fill="transparent" stroke="{tstroke}" stroke-width="0.8"/>'
               f'<text x="{x + 8 + tagw / 2}" y="{y + 17}" fill="{tcol}" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{e(tag)}</text>')
        if num: out += f'<text x="{x + w - 8}" y="{y + h - 6}" fill="{"rgba(191,69,32,0.10)" if kind == "focal" else "rgba(45,49,66,0.06)"}" font-size="28" font-weight="600" font-family="{MONO}" text-anchor="end">{num}</text>'
        cx = x + w / 2
        out += f'<text x="{cx}" y="{top}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="middle">{e(name)}</text>'
        for i, sub in enumerate(subs):
            out += f'<text x="{cx}" y="{top + 16 + i * 13}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="middle">{e(sub)}</text>'
        s.nodes.append(out + '</g>')
    def legend(s, y, items):
        out = (f'<line x1="40" y1="{y}" x2="{s.w - 40}" y2="{y}" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>'
               f'<text x="40" y="{y + 16}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>')
        x = 40
        for kind, text in items:
            if kind in STYLE:
                fill, stroke, *_ = STYLE[kind]
                out += f'<rect x="{x}" y="{y + 32}" width="14" height="10" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            else:
                col, mk, sw = {'muted': (MUTED, 'arrow', 1.2), 'accent': (ACCENT, 'arrow-accent', 1.4), 'link': (LINK, 'arrow-link', 1.2)}[kind]
                out += f'<line x1="{x}" y1="{y + 37}" x2="{x + 28}" y2="{y + 37}" stroke="{col}" stroke-width="{sw}" marker-end="url(#{mk})"/>'
            tx = x + (20 if kind in STYLE else 36)
            out += f'<text x="{tx}" y="{y + 40}" fill="{MUTED}" font-size="8.5" font-family="{SANS}">{e(text)}</text>'
            x = tx + round(tw(text, 8.5, 0.55) / 4) * 4 + 28
        assert x < s.w, 'legend too wide'
        s.leg = out
    def svg(s, title, desc):
        markers = ''.join(f'<marker id="{i}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{c}"/></marker>'
                          for i, c in (('arrow', MUTED), ('arrow-accent', ACCENT), ('arrow-link', LINK)))
        return (f'<svg viewBox="0 0 {s.w} {s.h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{s.slug}-title {s.slug}-desc">'
                f'<title id="{s.slug}-title">{e(title)}</title><desc id="{s.slug}-desc">{e(desc)}</desc>'
                f'<defs><pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.9" fill="rgba(45,49,66,0.10)"/></pattern>{markers}</defs>'
                f'<rect width="100%" height="100%" fill="{PAPER}"/><rect width="100%" height="100%" fill="url(#dots)" opacity="0.55"/>'
                + '\n'.join(s.zones + s.arrows + s.labels + s.nodes) + s.leg + '</svg>')

def page(eyebrow, h1, svg, caption, minw):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{e(h1)}</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: {SANS}; background: {PAPER}; color: {INK}; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 3rem 2rem; }}
  .frame {{ max-width: 1240px; width: 100%; }}
  .eyebrow {{ font-family: {MONO}; font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: {MUTED}; margin-bottom: 0.5rem; }}
  h1 {{ font-family: 'Instrument Serif', serif; font-size: clamp(1.5rem, 2.4vw + 0.75rem, 2rem); font-weight: 400; letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 1.5rem; }}
  svg {{ width: 100%; min-width: {minw}px; display: block; }}
  .cite {{ font-family: {MONO}; font-size: 0.62rem; letter-spacing: 0.06em; color: {SOFT}; margin-top: 0.9rem; }}
</style>
</head>
<body>
<div class="frame">
<p class="eyebrow">{e(eyebrow)}</p>
<h1>{e(h1)}</h1>
{svg}
<p class="cite">{e(caption)}</p>
</div>
</body>
</html>
'''

CITE = 'Drawn to the diagram-design rules and rendered by render_diagram on this stack (Open WebUI tool via the Lens bridge).'

# ---------------- 1. the whole stack ----------------
a = D('stack', 1176, 560)
a.zone(216, 216, 176, 112, 'INTERNET')
a.zone(424, 216, 192, 112, 'WINDOWS HOST')
a.zone(696, 24, 456, 424, 'WSL2 · DOCKER COMPOSE')
a.arrow('M 168,280 H 232', 'link');                     a.label(200, 268, 'HTTPS', 'link')
a.arrow('M 376,280 H 440', 'link');                     a.label(408, 268, 'ORIGIN', 'link')
a.arrow('M 104,312 V 344 Q 104,352 112,352 H 512 Q 520,352 520,344 V 312');  a.label(312, 340, 'LAN')
a.arrow('M 600,268 H 648 Q 656,268 656,260 V 192 Q 656,184 664,184 H 712', 'accent')
a.arrow('M 600,292 H 648 Q 656,292 656,300 V 368 Q 656,376 664,376 H 712');  a.label(680, 334, 'LAN')
a.arrow('M 792,152 V 120');                             a.label(820, 136, 'HOME')
a.arrow('M 872,172 H 968');                             a.label(920, 160, 'SPEECH')
a.arrow('M 872,196 H 904 Q 912,196 912,204 V 260 Q 912,268 920,268 H 968');  a.label(940, 228, 'CHAT')
a.arrow('M 872,364 H 916 Q 924,364 924,356 V 300 Q 924,292 932,292 H 968');  a.label(900, 326, 'CHAT')
a.arrow('M 872,388 H 968');                             a.label(920, 400, 'RAG · WEB')
a.node(40, 248, 128, 64, 'input', 'YOU', 'You', ['browser · phone'])
a.node(232, 248, 144, 64, 'external', 'CDN', 'CloudFront', ['your-domain.com'])
a.node(440, 248, 160, 64, 'backend', 'EDGE', 'nginx on Windows', ['Google sign-in · bans'])
a.node(712, 56, 160, 64, 'backend', 'IOT', 'Home Assistant', ['Hue · Lutron · TVs :8123'])
a.node(712, 152, 160, 64, 'focal', 'APP', 'Avatar', [':8702 · talking 3D head'])
a.node(712, 344, 160, 64, 'backend', 'APP', 'Open WebUI', [':3000 · chat, RAG, tools'])
a.node(968, 152, 168, 64, 'backend', 'AUDIO', 'Voice', ['Kokoro · Speaches · A2E'])
a.node(968, 248, 168, 64, 'backend', 'LLM', 'llama.cpp', [':8000 · local GGUF model'])
a.node(968, 344, 168, 88, 'store', 'TOOLS', 'Search, RAG & tools', ['Postgres · Tika · SearXNG', 'Playwright · Jupyter', 'ComfyUI · Terminal · MCP'])
a.legend(472, [('focal', 'Focal'), ('backend', 'Service'), ('store', 'Data & tools'), ('external', 'Cloud'), ('input', 'You'),
               ('link', 'Internet (HTTPS)'), ('accent', 'Signed-in path'), ('muted', 'Calls')])
stack = page('Architecture · Self-hosted AI stack', 'The whole stack, and how you reach it',
             a.svg('Self-hosted AI stack', 'Architecture of the stack: you reach the avatar through CloudFront and nginx with Google sign-in, or on the LAN; the avatar and Open WebUI share one local llama.cpp model, voice services, search and RAG tools, and the avatar controls Home Assistant.'),
             CITE, 1000)

# ---------------- 2. the avatar ----------------
b = D('avatar', 1000, 540)
b.zone(24, 40, 256, 304, 'VOICE & FACE')
b.zone(720, 232, 256, 208, 'PER GOOGLE ACCOUNT')
for y, t in ((104, 'SPEAK'), (192, 'TRANSCRIBE'), (296, 'LIP SYNC'), (392, 'PLAY')):
    b.arrow(f'M 420,{y} H 264'); b.label(342, y - 12, t)
for y, t, k in ((104, 'CHAT + TOOLS', 'accent'), (192, 'RUN TOOL', 'muted'), (296, 'SAVE', 'muted'), (392, 'CONTROL', 'muted')):
    b.arrow(f'M 580,{y} H 736', k); b.label(658, y - 12, t, k)
b.node(40, 72, 224, 64, 'backend', 'TTS', 'Kokoro', ['kokoro :8880 · your voices'])
b.node(40, 160, 224, 64, 'backend', 'STT', 'Speaches', ['speech-to-text :8090'])
b.node(40, 264, 224, 64, 'backend', 'LAM', 'LAM on the host GPU', ['A2E :8701 · maker :8703'])
b.node(40, 360, 224, 64, 'backend', 'MEDIA', 'Music', ['music :8704 · YouTube, radio'])
b.node(420, 72, 160, 352, 'focal', 'APP', 'Avatar page', ['avatar.html', 'nginx :8702', 'phone or browser'])
b.node(736, 72, 224, 64, 'backend', 'LLM', 'llama.cpp', ['llama :8000 · shared model'])
b.node(736, 160, 224, 64, 'backend', 'TOOLS', 'Lens bridge', ['lens-bridge :8095 · OWUI tools'])
b.node(736, 264, 224, 64, 'store', 'STORE', 'Saved chats', ['avatar-chats :8706'])
b.node(736, 360, 224, 64, 'backend', 'IOT', 'Home control', ['avatar-home :8707 → HA'])
b.legend(456, [('focal', 'The page'), ('backend', 'Service'), ('store', 'Store'), ('accent', 'Main path'), ('muted', 'Calls (relative paths)')])
avatar = page('Architecture · Avatar', 'What the avatar page talks to',
              b.svg('Avatar architecture', 'The avatar page calls, through its own nginx, the shared llama.cpp model, the Lens bridge for tools, Kokoro, Speaches and the LAM lip-sync and maker services, the music service, and per-account saved chats and home control.'),
              CITE, 900)
out = sys.argv[1]
open(out + '/stack.html', 'w').write(stack); open(out + '/avatar.html', 'w').write(avatar)
print('ok')
