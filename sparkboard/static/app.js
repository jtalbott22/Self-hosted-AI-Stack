/*
 * Sparkboard dashboard controller.
 *
 * Every request path below is relative ('./api/...'). That is what lets the
 * same files work at the domain root and behind an nginx subpath such as
 * /gpu/ -- with `proxy_pass http://127.0.0.1:9101/;` nginx strips the prefix
 * before the app ever sees it, so an absolute '/api/now' would escape the
 * mount point and hit whatever else is at the site root.
 */

import {
  PALETTE, StripChart, drawSparkline,
  fmtBytes, fmtRate, fmtNum, fmtDuration,
} from './charts.js';

const $ = (id) => document.getElementById(id);

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ESC[c]);

// Ink for each recognised service, so the same thing reads the same colour
// wherever it appears.
const TAG_CHIP = {
  'vLLM': 'gpu', 'Ollama': 'gpu',
  'Open WebUI': 'mem', 'postgres': 'mem', 'redis': 'mem',
  'uvicorn': 'cpu', 'gunicorn': 'cpu', 'hypercorn': 'cpu',
  'docker': 'power', 'containerd': 'power',
  'nginx': 'therm',
};
const chip = (label, kind) =>
  `<span class="chip chip--${kind || TAG_CHIP[label] || 'dim'}">${esc(label)}</span>`;

/*
 * The server marks these no-store, but a stale intermediate cache is exactly
 * the failure that leaves a table frozen while the SSE charts keep moving,
 * and it is invisible when it happens. A unique URL per request costs nothing
 * and removes the whole class of problem.
 */
async function jget(path) {
  const sep = path.includes('?') ? '&' : '?';
  const res = await fetch(`${path}${sep}_=${Date.now()}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${path} responded ${res.status}`);
  return res.json();
}

// What the per-row sparklines plot. Rates, not the cumulative counters docker
// reports, so a busy hour reads as a bump rather than a steeper ramp.
const CTREND = [
  { id: 'cpu', label: 'CPU', key: 'cpu', pen: PALETTE.cpu,
    fmt: (v) => `${v.toFixed(1)}%` },
  { id: 'mem', label: 'Memory', key: 'mem_used', pen: PALETTE.mem,
    fmt: (v) => fmtBytes(v, 1) },
  { id: 'net', label: 'Net in', key: 'net_rx_rate', pen: PALETTE.gpu,
    fmt: (v) => fmtRate(v) },
];

const LIVE_POINTS = 150;   // ~5 minutes at the default 2s cadence
const MIB = 1024 * 1024;

/* The sampler's in-memory ring uses short keys; the SQL history uses
   table-prefixed ones. Normalise the ring into the history's vocabulary so
   the charts never need to care which source they are reading. */
const RING_TO_SERIES = {
  gpu_util: 'gpu_util',
  gpu_temp: 'gpu_temp',
  gpu_power: 'gpu_power',
  gpu_mem_used: 'gpu_mem_used',
  gpu_sm_clock: 'gpu_sm_clock',
  gpu_mem_clock: 'gpu_mem_clock',
  cpu: 'host_cpu',
  cpu_temp: 'host_cpu_temp',
  mem_used: 'host_mem_used',
  net_rx: 'host_net_rx',
  net_tx: 'host_net_tx',
  disk_read: 'host_disk_read',
  disk_write: 'host_disk_write',
};

/*
 * Threshold bands. Return 'ok' / 'warn' / 'crit' so the readout value can be
 * tinted. Only the metrics where a level genuinely means something get one --
 * memory and clock don't have universal "danger" points, so they stay neutral.
 *
 * Temperature and power are scaled to the part rather than fixed, because the
 * fixed numbers that suit a 140 W passively-cooled module are wrong on a
 * discrete card in a tower: a 4070 at 65 degrees and 180 W is a card doing its
 * job, and painting that red would train the reader to ignore the colour. Power
 * is read against the card's own enforced cap when the driver reports one;
 * temperature bands shift up for a part with a fan.
 */
const BANDS = {
  temp: { warn: 50, crit: 70 },     // replaced at boot once the part is known
  power: { warn: 20, crit: 50 },
};

const th = {
  util: (v) => (v == null ? null : v < 50 ? 'ok' : v < 80 ? 'warn' : 'crit'),
  temp: (v) => (v == null ? null
    : v < BANDS.temp.warn ? 'ok' : v < BANDS.temp.crit ? 'warn' : 'crit'),
  power: (v) => (v == null ? null
    : v < BANDS.power.warn ? 'ok' : v < BANDS.power.crit ? 'warn' : 'crit'),
  cpu: (v) => (v == null ? null : v < 50 ? 'ok' : v < 85 ? 'warn' : 'crit'),
};

/* Called once /api/info has told us what card this is. */
function calibrateBands(info) {
  const cap = info.gpu_power_limit;
  if (cap) {
    // Fractions of the card's own cap: sustained work sits in the amber band,
    // and red means it is genuinely pinned against its limit.
    BANDS.power.warn = Math.round(cap * 0.35);
    BANDS.power.crit = Math.round(cap * 0.8);
  }
  if (!info.unified_memory) {
    // A discrete card with a fan runs hot by design. These are the numbers at
    // which a consumer GeForce part starts throttling, not the ones at which
    // a fanless module is already in trouble.
    BANDS.temp.warn = 65;
    BANDS.temp.crit = 83;
  }
}

const READOUTS = [
  { id: 'util', label: 'GPU util', unit: '%', pen: PALETTE.gpu,
    pick: (s) => s.gpu?.util, fmt: (v) => fmtNum(v, 0), spark: 'gpu_util',
    level: th.util,
    tip: 'Percentage of time the GPU compute units were active.' },
  { id: 'gmem', label: 'GPU memory', unit: 'GB', pen: PALETTE.mem,
    pick: (s) => (s.gpu?.mem_used != null ? s.gpu.mem_used / 1024 : null),
    fmt: (v) => fmtNum(v, 1), spark: 'gpu_mem_used',
    tip: 'GPU memory in use.',
    tipUnified: 'GPU memory in use. On this unified-memory part it is drawn '
      + 'from the same pool as system RAM.' },
  { id: 'temp', label: 'GPU temp', unit: '\u00B0C', pen: PALETTE.therm,
    pick: (s) => s.gpu?.temp, fmt: (v) => fmtNum(v, 0), spark: 'gpu_temp',
    level: th.temp,
    tip: 'GPU die temperature. Green below 50\u00B0C, amber 50\u201370\u00B0C, red above 70\u00B0C.' },
  { id: 'power', label: 'Power', unit: 'W', pen: PALETTE.power,
    pick: (s) => s.gpu?.power, fmt: (v) => fmtNum(v, 0), spark: 'gpu_power',
    level: th.power,
    tip: 'GPU module power draw in watts.' },
  { id: 'cpu', label: 'CPU', unit: '%', pen: PALETTE.cpu,
    pick: (s) => s.host?.cpu, fmt: (v) => fmtNum(v, 0), spark: 'host_cpu',
    level: th.cpu,
    tip: 'Average CPU utilization across all cores.' },
  { id: 'clock', label: 'SM clock', unit: 'GHz', pen: PALETTE.gpu,
    pick: (s) => (s.gpu?.sm_clock != null ? s.gpu.sm_clock / 1000 : null),
    fmt: (v) => fmtNum(v, 2), spark: 'gpu_sm_clock',
    tip: 'Streaming-multiprocessor clock \u2014 the GPU core frequency, in GHz.' },
];

const PANELS = [
  {
    id: 'util', title: 'GPU utilization', unit: '%', yMin: 0, yMax: 100,
    format: (v) => fmtNum(v, 0),
    series: [{ key: 'gpu_util', maxKey: 'gpu_util_max', label: 'GPU', color: PALETTE.gpu }],
  },
  {
    id: 'temp', title: 'Temperature', unit: '\u00B0C',
    format: (v) => fmtNum(v, 0),
    series: [
      { key: 'gpu_temp', maxKey: 'gpu_temp_max', label: 'GPU', color: PALETTE.therm },
      { key: 'host_cpu_temp', maxKey: 'host_cpu_temp_max', label: 'CPU', color: PALETTE.alert, fill: false },
    ],
  },
  {
    id: 'gmem', title: 'GPU memory allocated', unit: ' GB',
    format: (v) => fmtNum(v / 1024, 0),
    series: [{ key: 'gpu_mem_used', label: 'Allocated', color: PALETTE.mem }],
  },
  {
    id: 'power', title: 'Power draw', unit: ' W',
    format: (v) => fmtNum(v, 0),
    series: [{ key: 'gpu_power', maxKey: 'gpu_power_max', label: 'GPU module', color: PALETTE.power }],
  },
  {
    id: 'clocks', title: 'Clocks', unit: ' MHz',
    format: (v) => fmtNum(v, 0),
    series: [
      { key: 'gpu_sm_clock', label: 'SM', color: PALETTE.gpu, fill: false },
      { key: 'gpu_mem_clock', label: 'Memory', color: PALETTE.mem, fill: false },
    ],
  },
  {
    id: 'cpu', title: 'CPU utilization', unit: '%', yMin: 0, yMax: 100,
    format: (v) => fmtNum(v, 0),
    series: [{ key: 'host_cpu', maxKey: 'host_cpu_max', label: 'CPU', color: PALETTE.cpu }],
  },
  {
    id: 'net', title: 'Network', unit: '',
    format: (v) => fmtBytes(v, 0),
    tooltipFormat: fmtRate,
    series: [
      { key: 'host_net_rx', label: 'Receive', color: PALETTE.cpu },
      { key: 'host_net_tx', label: 'Transmit', color: PALETTE.power, fill: false },
    ],
  },
  {
    id: 'disk', title: 'Disk', unit: '',
    format: (v) => fmtBytes(v, 0),
    tooltipFormat: fmtRate,
    series: [
      { key: 'host_disk_read', label: 'Read', color: PALETTE.mem },
      { key: 'host_disk_write', label: 'Write', color: PALETTE.therm, fill: false },
    ],
  },
];

const RANGES = ['LIVE', '5m', '1h', '6h', '24h', '7d', '30d'];
const RANGE_SECONDS = { '5m': 300, '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800, '30d': 2592000 };

const state = {
  info: {},
  range: 'LIVE',
  paused: false,
  live: { t: [], series: {} },
  charts: new Map(),
  sparks: new Map(),
  es: null,
  refreshTimer: null,
  latest: null,
  ctrend: 'cpu',
  lastDocker: null,
  containerHistory: null,
  servicesTs: 0,
};

/* ------------------------------------------------------------------ build */

/* Some readouts describe themselves differently on a unified-memory part. The
   rail is rebuilt once /api/info lands, so this resolves correctly then even
   though the first build runs before the host has been identified. */
function readoutTip(r) {
  return (state.info?.unified_memory && r.tipUnified) ? r.tipUnified : r.tip;
}

function buildReadouts() {
  const host = $('readout');
  host.innerHTML = '';
  for (const r of READOUTS) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.innerHTML = `
      <div class="cell__label">
        <span class="cell__pen" style="background:${r.pen}"></span>${r.label}
        ${readoutTip(r) ? `<span class="hint" data-tip="${esc(readoutTip(r))}">?</span>` : ''}
      </div>
      <div class="cell__read">
        <span class="cell__value" id="rv-${r.id}">&mdash;</span>
        <span class="cell__unit">${r.unit}</span>
      </div>
      <div class="cell__sub" id="rs-${r.id}"></div>
      <div class="cell__spark"><canvas id="rc-${r.id}"></canvas></div>`;
    host.appendChild(cell);
  }
  // Newly built hints need to be focusable too (boot's pass ran before this).
  for (const h of host.querySelectorAll('.hint')) {
    h.setAttribute('tabindex', '0');
    h.setAttribute('role', 'button');
  }
}

function buildPanels() {
  const grid = $('grid');
  grid.innerHTML = '';
  for (const p of PANELS) {
    const panel = document.createElement('section');
    panel.className = 'panel';
    const legend = p.series.map((s) => `
      <span class="legend__item">
        <span class="legend__pen" style="background:${s.color}"></span>${s.label}
      </span>`).join('');
    panel.innerHTML = `
      <div class="panel__head">
        <div>
          <h2 class="eyebrow">${p.title}</h2>
          <div class="legend">${legend}</div>
        </div>
        <span class="panel__now" id="pn-${p.id}"></span>
      </div>
      <div class="panel__plot"><canvas id="pc-${p.id}"></canvas></div>`;
    grid.appendChild(panel);

    const chart = new StripChart($(`pc-${p.id}`), {
      series: p.series,
      yMin: p.yMin,
      yMax: p.yMax,
      format: p.format,
      unit: p.unit,
    });
    chart.onHover = (idx, ev) => showTip(chart, p, idx, ev);
    state.charts.set(p.id, chart);
  }
}

function buildRanges() {
  const host = $('ranges');
  host.innerHTML = '';
  for (const r of RANGES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = r;
    b.setAttribute('aria-pressed', String(r === state.range));
    b.addEventListener('click', () => selectRange(r));
    host.appendChild(b);
  }
}

/* ---------------------------------------------------------------- tooltip */

function showTip(chart, panel, idx, ev) {
  const tip = $('tip');
  if (idx === null || !ev) { tip.hidden = true; return; }
  const info = chart.tooltipFor(idx);
  if (!info) { tip.hidden = true; return; }

  const when = new Date(info.t * 1000).toLocaleString([], {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const rows = info.rows.map((r, i) => {
    const raw = chart.data[panel.series[i].key]?.[idx];
    const value = (panel.tooltipFormat && raw != null)
      ? panel.tooltipFormat(raw)
      : r.value;
    return `<div class="tip__row">
        <span class="tip__pen" style="background:${r.color}"></span>
        <span class="tip__label">${r.label}</span>
        <span class="tip__value">${value}</span>
      </div>`;
  }).join('');

  tip.innerHTML = `<div class="tip__time">${when}</div>${rows}`;
  tip.hidden = false;

  const box = tip.getBoundingClientRect();
  let x = ev.clientX + 14;
  let y = ev.clientY - box.height / 2;
  if (x + box.width > window.innerWidth - 8) x = ev.clientX - box.width - 14;
  y = Math.max(8, Math.min(window.innerHeight - box.height - 8, y));
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

/* ------------------------------------------------------------------- data */

function ringToSeries(samples) {
  const t = [];
  const series = {};
  for (const key of Object.values(RING_TO_SERIES)) series[key] = [];
  for (const s of samples) {
    t.push(s.ts);
    for (const [from, to] of Object.entries(RING_TO_SERIES)) {
      series[to].push(s[from] ?? null);
    }
  }
  return { t, series };
}

function pushLivePoint(point) {
  if (!point) return;
  const live = state.live;
  live.t.push(point.ts);
  for (const [from, to] of Object.entries(RING_TO_SERIES)) {
    if (!live.series[to]) live.series[to] = [];
    live.series[to].push(point[from] ?? null);
  }
  while (live.t.length > LIVE_POINTS) {
    live.t.shift();
    for (const key of Object.keys(live.series)) live.series[key].shift();
  }
}

async function loadLive() {
  const data = await jget('./api/live');
  state.live = ringToSeries(data.samples || []);
  renderCharts(state.live);
  $('range-detail').textContent =
    `streaming \u00b7 ${state.info.interval ?? '?'}s cadence`;
}

async function loadHistory(range) {
  let data;
  try {
    data = await jget(`./api/history?range=${encodeURIComponent(range)}`);
  } catch {
    $('range-detail').textContent = 'range unavailable';
    return;
  }
  renderCharts(data);
  const bucket = data.bucket >= 3600
    ? `${(data.bucket / 3600).toFixed(0)}h`
    : (data.bucket >= 60 ? `${(data.bucket / 60).toFixed(0)}m` : `${data.bucket}s`);
  $('range-detail').textContent =
    `${data.points} points \u00b7 ${bucket} buckets \u00b7 ${data.tier} tier`;
}

function renderCharts(data) {
  for (const p of PANELS) {
    const chart = state.charts.get(p.id);
    if (!chart) continue;
    chart.setData(data.t, data.series);

    const nowEl = $(`pn-${p.id}`);
    if (!nowEl) continue;
    const key = p.series[0].key;
    const arr = data.series[key] || [];
    let last = null;
    for (let i = arr.length - 1; i >= 0; i -= 1) {
      if (arr[i] !== null && arr[i] !== undefined) { last = arr[i]; break; }
    }
    if (last === null) { nowEl.textContent = '\u2014'; continue; }
    nowEl.textContent = p.tooltipFormat
      ? p.tooltipFormat(last)
      : `${p.format(last)}${p.unit}`;
  }
}

/* ---------------------------------------------------------------- render */

function renderReadouts(sample) {
  const flat = { gpu: (sample.gpus || [])[0] || {}, host: sample.host || {} };
  for (const r of READOUTS) {
    const v = r.pick(flat);
    const el = $(`rv-${r.id}`);
    if (el) {
      el.textContent = r.fmt(v);
      // Tint by threshold band where the metric has one.
      const lvl = r.level ? r.level(v) : null;
      el.classList.remove('lvl-ok', 'lvl-warn', 'lvl-crit');
      if (lvl) el.classList.add(`lvl-${lvl}`);
    }
    const spark = $(`rc-${r.id}`);
    if (spark) drawSparkline(spark, state.live.series[r.spark] || [], r.pen, 34);
  }

  const g = flat.gpu;
  const h = flat.host;
  const sub = (id, text) => { const e = $(`rs-${id}`); if (e) e.textContent = text; };

  sub('util', g.pstate ? `perf state ${g.pstate}` : '');
  // Say where the figure came from when nvidia-smi didn't supply it.
  sub('gmem', g.mem_total
    ? `of ${(g.mem_total / 1024).toFixed(0)} GB${g.mem_used_inferred ? ' \u00b7 from processes' : ''}`
    : '');
  // Power limit is [N/A] on GB10 -- show nothing rather than a fabricated cap.
  sub('power', g.power_limit ? `cap ${fmtNum(g.power_limit, 0)} W` : '');
  sub('temp', g.fan != null ? `fan ${fmtNum(g.fan, 0)}%` : '');
  sub('cpu', h.load1 != null ? `load ${fmtNum(h.load1, 2)}` : '');
  sub('clock', g.mem_clock ? `mem ${fmtNum(g.mem_clock, 0)} MHz` : '');
}

function renderPool(sample) {
  const pool = $('pool');
  const g = (sample.gpus || [])[0];
  const h = sample.host || {};
  if (!g || !h.mem_total) { pool.hidden = true; return; }
  pool.hidden = false;

  const unified = state.info.unified_memory;
  const gpuBytes = g.mem_used != null ? g.mem_used * MIB : null;
  const gpuTotal = g.mem_total != null ? g.mem_total * MIB : null;

  if (unified && gpuTotal) {
    const total = Math.max(gpuTotal, h.mem_total);
    $('pool-title').textContent = 'Unified memory pool';
    $('pool-cap').textContent = `${(total / 1024 ** 3).toFixed(0)} GB shared`;
    setRuler(total / 1024 ** 3, 'GB');
    setFill('pool-gpu-fill', gpuBytes, total);
    setFill('pool-host-fill', h.mem_used, total);
    $('pool-gpu-value').textContent = fmtBytes(gpuBytes, 1);
    $('pool-host-value').textContent = fmtBytes(h.mem_used, 1);
    $('pool-note').textContent = g.mem_used_inferred
      ? 'This part reports no framebuffer of its own, so GPU allocation is '
        + 'summed from the processes holding it and the pool size is host RAM. '
        + 'The two readings are not added together \u2014 they describe the '
        + 'same physical memory from two sides.'
      : 'CPU and GPU allocate from the same physical memory. The two readings '
        + 'are not added together \u2014 on a coherent pool they can describe '
        + 'some of the same pages.';
  } else {
    $('pool-title').textContent = 'Memory';
    $('pool-cap').textContent = gpuTotal
      ? `GPU ${(gpuTotal / 1024 ** 3).toFixed(0)} GB \u00b7 host ${(h.mem_total / 1024 ** 3).toFixed(0)} GB`
      : `host ${(h.mem_total / 1024 ** 3).toFixed(0)} GB`;
    setRuler(100, '%');
    setFill('pool-gpu-fill', gpuBytes, gpuTotal);
    setFill('pool-host-fill', h.mem_used, h.mem_total);
    $('pool-gpu-value').textContent = gpuTotal
      ? `${fmtBytes(gpuBytes, 1)}` : '\u2014';
    $('pool-host-value').textContent = fmtBytes(h.mem_used, 1);
    $('pool-note').textContent = '';
  }
}

let rulerKey = null;
function setRuler(max, unit) {
  const key = `${max}${unit}`;
  if (rulerKey === key) return;   // static once drawn; avoids per-sample churn
  rulerKey = key;
  const el = $('pool-ruler');
  el.innerHTML = '';
  const steps = 4;
  for (let i = 0; i <= steps; i += 1) {
    const v = (max / steps) * i;
    const span = document.createElement('span');
    span.className = 'pool__tick';
    span.style.left = `${(i / steps) * 100}%`;
    span.textContent = i === steps ? `${v.toFixed(0)} ${unit}` : v.toFixed(0);
    el.appendChild(span);
  }
}

function setFill(id, used, total) {
  const el = $(id);
  if (!el) return;
  const pct = (used != null && total) ? Math.max(0, Math.min(100, (used / total) * 100)) : 0;
  el.style.width = `${pct}%`;
}

/*
 * Who is using the machine's memory.
 *
 * First choice is the GPU's own answer, which names the processes actually
 * holding video memory. Some platforms can't give it: the WSL2 driver does no
 * per-process accounting, so the query returns an empty list on a card that is
 * fully committed. An empty table there would read as "nothing is running",
 * which is the opposite of the truth, so the panel switches to the heaviest
 * processes by resident memory and says so in its own heading. A weaker
 * question, answered honestly, beats a strong one answered wrongly.
 */
function renderProcs(sample) {
  const body = $('procs-body');
  const title = $('procs-title');
  const procs = sample.procs || [];
  const g = (sample.gpus || [])[0] || {};
  const total = g.mem_total || null;
  const caps = state.info.capabilities || {};

  if (procs.length) {
    title.textContent = 'Processes holding GPU memory';
    $('procs-col-mem').textContent = 'Memory';
    $('procs-col-share').textContent = 'Share of pool';
    $('procs-count').textContent =
      `${procs.length} process${procs.length === 1 ? '' : 'es'}`;
    body.innerHTML = procs.map((p) => {
      const pct = (total && p.mem != null) ? (p.mem / total) * 100 : 0;
      const label = esc(p.cmd || p.name || '?');
      return `<tr>
        <td>${p.pid}</td>
        <td class="key trunc" title="${label}">${label}</td>
        <td class="num">${p.mem != null ? `${(p.mem / 1024).toFixed(1)} GB` : '\u2014'}</td>
        <td class="hide-sm"><div class="dtable__bar"><i style="width:${pct.toFixed(1)}%"></i></div></td>
      </tr>`;
    }).join('');
    return;
  }

  // No GPU process list. Either the platform can't produce one, or the card
  // genuinely has nothing on it -- and those two get different wording.
  if (caps.gpu_process_accounting !== false) {
    title.textContent = 'Processes holding GPU memory';
    $('procs-count').textContent = '';
    body.innerHTML = '<tr class="dtable__empty"><td colspan="4">'
      + 'No processes are holding GPU memory.</td></tr>';
    return;
  }

  const host = sample.host_procs || [];
  const hasGpu = state.info.gpu_present;
  title.textContent = 'Heaviest processes';
  $('procs-col-mem').textContent = 'Resident';
  $('procs-col-share').textContent = 'Share of RAM';
  $('procs-count').textContent = hasGpu
    ? 'GPU cannot name its own processes here' : 'by resident memory';

  if (!host.length) {
    body.innerHTML = '<tr class="dtable__empty"><td colspan="4">'
      + 'Gathering the process table\u2026</td></tr>';
    return;
  }

  const ram = sample.host?.mem_total || null;
  const rows = host.map((p) => {
    const pct = (ram && p.rss != null) ? (p.rss / ram) * 100 : 0;
    const label = esc(p.name || '?');
    return `<tr>
      <td>${p.pid}</td>
      <td class="key trunc" title="${label}">${label}</td>
      <td class="num">${fmtBytes(p.rss, 1)}</td>
      <td class="hide-sm"><div class="dtable__bar"><i style="width:${pct.toFixed(1)}%"></i></div></td>
    </tr>`;
  }).join('');
  const note = hasGpu
    ? 'This driver does not report which processes hold video memory, so these '
      + 'are the heaviest processes by system memory instead. The GPU memory '
      + 'figure above is still the real one \u2014 it just cannot be attributed.'
    : 'No GPU on this host, so this is the heaviest processes by resident '
      + 'memory.';
  body.innerHTML = rows
    + `<tr class="dtable__empty"><td colspan="4">${note}</td></tr>`;
}

/* Filesystems, from the paths the service was told to watch. */
function renderStorage(info) {
  const section = $('storage');
  const disks = (info.disks || []).filter((d) => d.total || d.error);
  if (!section || !disks.length) {
    if (section) section.hidden = true;
    return;
  }
  section.hidden = false;
  $('storage-meta').textContent =
    `${disks.length} filesystem${disks.length === 1 ? '' : 's'}`;
  $('storage-body').innerHTML = disks.map((d) => {
    if (d.error || !d.total) {
      return `<tr>
        <td class="key">${esc(d.label || d.path)}</td>
        <td colspan="4" class="warn">${esc(d.error || 'unreadable')}</td>
      </tr>`;
    }
    const pct = d.pct != null ? d.pct : (d.used / d.total) * 100;
    return `<tr>
      <td class="key trunc" title="${esc(d.path)}">${esc(d.label || d.path)}</td>
      <td class="num">${fmtBytes(d.used, 1)}</td>
      <td class="num">${fmtBytes(d.free, 1)}</td>
      <td class="num hide-sm">${fmtBytes(d.total, 0)}</td>
      <td class="hide-sm"><div class="dtable__bar"><i style="width:${pct.toFixed(1)}%"></i></div></td>
    </tr>`;
  }).join('');
}

function renderDocker(d) {
  const body = $('docker-body');
  const meta = $('docker-meta');

  if (!d || !d.available) {
    const why = d?.error || 'unavailable';
    // Docker access is opt-in, so say what to do rather than just reporting.
    const hint = /permission|socket/i.test(why)
      ? 'No access to the docker socket. Re-run install.sh, or add the service '
        + 'user to the docker group \u2014 see the README on what that grants.'
      : `Docker unavailable: ${esc(why)}`;
    meta.textContent = '';
    body.innerHTML = `<tr class="dtable__empty"><td colspan="7" class="warn">${hint}</td></tr>`;
    return;
  }

  const cs = d.containers || [];
  tickServicesAge();

  if (!cs.length) {
    body.innerHTML = '<tr class="dtable__empty"><td colspan="7">'
      + 'No containers are running.</td></tr>';
    return;
  }

  body.innerHTML = cs.map((c, i) => {
    const ports = (c.ports || []).map((p) => (p.host
      ? chip(`${p.host}\u2192${p.container}`, 'power')
      : chip(`${p.container}`, 'dim'))).join('') || '<span class="scope--local">none</span>';
    const mem = c.mem_used != null
      ? `${fmtBytes(c.mem_used, 1)}${c.mem_pct != null ? ` <span class="scope--local">${c.mem_pct.toFixed(1)}%</span>` : ''}`
      : '\u2014';
    const net = (c.net_rx != null || c.net_tx != null)
      ? `${fmtBytes(c.net_rx, 1)} / ${fmtBytes(c.net_tx, 1)}`
      : '\u2014';
    return `<tr>
      <td class="key">${esc(c.name)}</td>
      <td class="trunc hide-sm" title="${esc(c.image)}">${esc(c.image)}</td>
      <td class="num">${c.cpu != null ? `${c.cpu.toFixed(1)}%` : '\u2014'}</td>
      <td class="num">${mem}</td>
      <td>${ports}</td>
      <td class="num hide-sm">${net}</td>
      <td class="trend hide-sm"><canvas id="ctr-${i}"></canvas></td>
    </tr>`;
  }).join('');

  // Defer a frame: the canvases size themselves from their parent cell, and
  // the table hasn't been laid out yet at this point in the same tick.
  requestAnimationFrame(() => drawRowTrends(cs));
}

function tickServicesAge() {
  const el = $('docker-meta');
  if (!el || !state.servicesTs || !state.lastDocker?.available) return;
  const cs = state.lastDocker.containers || [];
  const age = Math.max(0, Math.round(Date.now() / 1000 - state.servicesTs));
  const bits = [`${cs.length} running`];
  if (state.lastDocker.stats_error) bits.push('usage unavailable');
  // Freshness on the face of it: a frozen table should look frozen instead of
  // looking like nothing happened to change.
  bits.push(age < 3 ? 'just now' : `${age}s ago`);
  el.textContent = bits.join(' \u00b7 ');
}

function renderPorts(l) {
  const body = $('ports-body');
  const meta = $('ports-meta');

  if (!l || l.error) {
    meta.textContent = '';
    body.innerHTML = `<tr class="dtable__empty"><td colspan="5" class="warn">${
      esc(l?.error || 'Port inventory unavailable.')}</td></tr>`;
    return;
  }

  const ports = l.ports || [];
  const parts = [`${ports.length} listening`];
  if (l.host_scope === false) {
    // Worth stating plainly: a containerised install with no host /proc
    // mounted is reporting its own namespace, which is a true answer about
    // the wrong machine.
    parts.push('container namespace only');
  }
  if (l.unattributed) {
    // Being able to see the socket but not its owner is a capability problem,
    // not an empty result -- say which, so it's actionable.
    parts.push(`${l.unattributed} unattributed`);
  }
  meta.textContent = parts.join(' \u00b7 ');

  if (!ports.length) {
    body.innerHTML = '<tr class="dtable__empty"><td colspan="5">'
      + 'Nothing is listening on a TCP port.</td></tr>';
    return;
  }

  const rows = ports.map((p) => {
    const tags = [];
    if (p.tag) tags.push(chip(p.tag));
    if (p.container) tags.push(chip(p.container, 'power'));
    const scopeClass = p.scope === 'localhost' ? 'scope--local' : 'scope--open';
    const cmd = p.cmdline || p.process;
    return `<tr>
      <td class="num key">${p.port}</td>
      <td>${tags.join('') || '<span class="scope--local">\u2014</span>'}</td>
      <td class="trunc" title="${esc(cmd || '')}">${
        cmd ? esc(cmd) : '<span class="scope--local">not attributed</span>'}</td>
      <td class="num">${p.pid ?? '\u2014'}</td>
      <td class="hide-sm"><span class="scope ${scopeClass}">${esc(p.scope)}</span></td>
    </tr>`;
  }).join('');

  const note = l.unattributed
    ? `<tr class="dtable__empty"><td colspan="5" class="warn">${l.unattributed} port${
        l.unattributed === 1 ? '' : 's'} could not be matched to a process. Reading `
      + 'another user\u2019s /proc entries needs CAP_SYS_PTRACE and '
      + 'CAP_DAC_READ_SEARCH \u2014 re-run install.sh for the service, or add '
      + '<code>cap_add: [SYS_PTRACE]</code> and <code>pid: host</code> to the '
      + 'container.</td></tr>'
    : '';
  body.innerHTML = rows + note;
}

async function loadServices() {
  let d;
  try {
    d = await jget('./api/services');
  } catch (err) {
    $('docker-meta').textContent = 'inventory unreachable';
    return;
  }
  state.lastDocker = d.docker;
  state.servicesTs = d.ts || Date.now() / 1000;
  // Render outside the fetch try: swallowing an exception thrown in here
  // would leave the table silently frozen on its last good result, which is
  // indistinguishable from the data simply not changing.
  renderDocker(d.docker);
  renderPorts(d.listeners);
}

async function loadContainerHistory(range) {
  // Containers sample every 10s, so LIVE has no separate meaning for them;
  // the shortest window that holds enough points is the 5m one.
  const r = (range === 'LIVE') ? '5m' : range;
  try {
    state.containerHistory = await jget(
      `./api/containers/history?range=${encodeURIComponent(r)}`);
  } catch {
    state.containerHistory = null;
  }
  const head = $('ctrend-head');
  if (head) head.textContent = `Trend \u00b7 ${range === 'LIVE' ? '5m' : range}`;
  if (state.lastDocker) renderDocker(state.lastDocker);
}

function buildContainerToggle() {
  const host = $('cmetric');
  if (!host) return;
  host.innerHTML = '';
  for (const m of CTREND) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = m.label;
    b.setAttribute('aria-pressed', String(m.id === state.ctrend));
    b.addEventListener('click', () => {
      state.ctrend = m.id;
      for (const other of host.children) {
        other.setAttribute('aria-pressed', String(other.textContent === m.label));
      }
      if (state.lastDocker) renderDocker(state.lastDocker);
    });
    host.appendChild(b);
  }
}

function drawRowTrends(containers) {
  const metric = CTREND.find((m) => m.id === state.ctrend) || CTREND[0];
  const hist = state.containerHistory;
  containers.forEach((c, i) => {
    const canvas = $(`ctr-${i}`);
    if (!canvas) return;
    const series = hist?.containers?.[c.name]?.[metric.key];
    if (!series || !series.some((v) => v !== null && v !== undefined)) return;
    drawSparkline(canvas, series, metric.pen, 22);
    const real = series.filter((v) => v !== null && v !== undefined);
    if (real.length) {
      canvas.parentElement.title =
        `${metric.label} \u00b7 peak ${metric.fmt(Math.max(...real))}`;
    }
  });
}

/* --------------------------------------------------------------- cores */

let coresBuilt = 0;
// CPU load tiers. Thresholds chosen so "idle" is genuinely idle (background
// noise), and "high" is a core doing real work. A stacked bar of these counts
// reads the whole CPU at a glance; the individual cores are still available
// underneath for anyone who wants them.
const CPU_TIERS = [
  { id: 'idle', label: 'Idle', min: 0, max: 10, color: 'var(--ink-faint)' },
  { id: 'low', label: 'Low', min: 10, max: 40, color: PALETTE.cpu },
  { id: 'med', label: 'Medium', min: 40, max: 75, color: PALETTE.gpu },
  { id: 'high', label: 'High', min: 75, max: 101, color: PALETTE.therm },
];

function tierOf(v) {
  for (const t of CPU_TIERS) {
    if (v >= t.min && v < t.max) return t;
  }
  return CPU_TIERS[CPU_TIERS.length - 1];
}

let coresExpanded = false;

function renderCores(sample) {
  const cores = sample?.host?.cpu_cores;
  const section = $('cores');
  if (!Array.isArray(cores) || cores.length < 2) {
    if (section) section.hidden = true;
    return;
  }
  section.hidden = false;

  // Bucket cores into tiers.
  const counts = { idle: 0, low: 0, med: 0, high: 0 };
  let peak = 0;
  let sum = 0;
  for (const raw of cores) {
    const v = raw ?? 0;
    sum += v;
    if (v > peak) peak = v;
    counts[tierOf(v).id] += 1;
  }
  const n = cores.length;
  const avg = sum / n;

  // Stacked tier bar -- one segment per tier, width proportional to core count.
  const bar = $('cores-bar');
  if (bar) {
    bar.innerHTML = CPU_TIERS.map((t) => {
      const c = counts[t.id];
      if (!c) return '';
      const pct = (c / n) * 100;
      return `<span class="tierbar__seg" style="width:${pct}%;background:${t.color}"
                    title="${t.label}: ${c} core${c === 1 ? '' : 's'}"></span>`;
    }).join('');
  }

  // Tier legend with live counts.
  const legend = $('cores-legend');
  if (legend) {
    legend.innerHTML = CPU_TIERS.map((t) => `
      <span class="tierkey">
        <span class="tierkey__dot" style="background:${t.color}"></span>
        ${t.label} <b>${counts[t.id]}</b>
      </span>`).join('');
  }

  $('cores-meta').textContent =
    `${n} cores \u00b7 avg ${avg.toFixed(0)}% \u00b7 peak ${peak.toFixed(0)}%`;

  // Expandable per-core detail -- built once, updated each sample when open.
  const detail = $('cores-detail');
  if (detail) {
    detail.hidden = !coresExpanded;
    if (coresExpanded) {
      if (coresBuilt !== n) {
        detail.innerHTML = cores.map((_, i) => `
          <div class="core">
            <div class="core__top">
              <span class="core__id">c${i}</span>
              <span class="core__val" id="core-v-${i}">&mdash;</span>
            </div>
            <div class="core__bar"><i class="core__fill" id="core-f-${i}"></i></div>
          </div>`).join('');
        coresBuilt = n;
      }
      for (let i = 0; i < n; i += 1) {
        const v = cores[i] ?? 0;
        const fill = $(`core-f-${i}`);
        const val = $(`core-v-${i}`);
        if (fill) {
          fill.style.width = `${Math.max(0, Math.min(100, v))}%`;
          fill.style.background = tierOf(v).color;
        }
        if (val) val.textContent = `${v.toFixed(0)}`;
      }
    }
  }
}

/* --------------------------------------------------------- activity feed */

const FEED_SPEEDS = [
  { id: 'slow', label: 'Slow', pxps: 28 },
  { id: 'med', label: 'Medium', pxps: 60 },
  { id: 'fast', label: 'Fast', pxps: 110 },
];

// Ink per category family, so a glance at the colours reads the mix even
// before the words.
const CAT_COLOR = {
  'writing code': PALETTE.gpu,
  'debugging code': '#4FD6A0',
  'reviewing code': '#8CE0C0',
  'writing or editing text': PALETTE.mem,
  'drafting an email or message': '#8FB4FF',
  'writing a story or creative piece': PALETTE.power,
  'summarizing or extracting': '#B4A0E8',
  'translating': '#7DD3E0',
  'answering a question': PALETTE.cpu,
  'explaining a concept': '#6FC7D4',
  'planning or organizing': PALETTE.therm,
  'data analysis or math': '#FFC070',
  'role-play or conversation': '#E89BC0',
  'other': PALETTE.dim,
};
const catColor = (label) => CAT_COLOR[label] || PALETTE.dim;

const feedState = {
  enabled: false,
  seen: 0,
  pending: [],
  onscreen: [],
  speed: 'med',
  paused: false,
  x: 0,
  lastFrame: 0,
  raf: 0,
  empty: true,
};

function buildFeedControls() {
  const host = $('feed-speed');
  if (!host) return;
  host.innerHTML = '';
  for (const s of FEED_SPEEDS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = s.label;
    b.setAttribute('aria-pressed', String(s.id === feedState.speed));
    b.addEventListener('click', () => {
      feedState.speed = s.id;
      for (const o of host.children) o.setAttribute('aria-pressed', String(o.textContent === s.label));
    });
    host.appendChild(b);
  }
  $('feed-pause').addEventListener('click', () => {
    feedState.paused = !feedState.paused;
    $('feed-pause').setAttribute('aria-pressed', String(feedState.paused));
    $('feed-pause').textContent = feedState.paused ? 'Resume' : 'Pause';
  });
}

function feedItemEl(ev) {
  const el = document.createElement('div');
  el.className = 'feed__item';
  const dot = document.createElement('span');
  dot.className = 'feed__dot';
  dot.style.background = catColor(ev.label);
  const label = document.createElement('span');
  label.textContent = ev.label;
  el.appendChild(dot);
  el.appendChild(label);
  if (ev.tokens != null) {
    const age = document.createElement('span');
    age.className = 'feed__age';
    age.textContent = `${ev.tokens} tok`;
    el.appendChild(age);
  }
  return el;
}

// Driven by rAF rather than a CSS animation: items arrive continuously and the
// speed is adjustable live, which a keyframe animation can't do without a jump.
function feedTick(now) {
  feedState.raf = requestAnimationFrame(feedTick);
  const track = $('feed-track');
  if (!track) return;

  const dt = feedState.lastFrame ? (now - feedState.lastFrame) / 1000 : 0;
  feedState.lastFrame = now;
  if (feedState.paused) return;

  const speed = FEED_SPEEDS.find((s) => s.id === feedState.speed) || FEED_SPEEDS[1];
  const marquee = $('feed-marquee');
  const width = marquee ? marquee.clientWidth : 800;

  // Feed new items in as the tail clears the right edge.
  if (feedState.pending.length) {
    const last = feedState.onscreen[feedState.onscreen.length - 1];
    const lastRight = last ? last.el.offsetLeft + last.el.offsetWidth + feedState.x : 0;
    if (!last || lastRight < width) {
      const ev = feedState.pending.shift();
      const el = feedItemEl(ev);
      track.appendChild(el);
      feedState.onscreen.push({ el });
      if (feedState.empty) { feedState.empty = false; clearFeedEmpty(); }
    }
  }

  feedState.x -= speed.pxps * dt;
  track.style.transform = `translateX(${feedState.x}px)`;

  while (feedState.onscreen.length) {
    const first = feedState.onscreen[0];
    const right = first.el.offsetLeft + first.el.offsetWidth + feedState.x;
    if (right < -20) { first.el.remove(); feedState.onscreen.shift(); } else break;
  }

  // Once drained, reset the offset so x can't grow unbounded over long uptime.
  if (!feedState.onscreen.length && !feedState.pending.length) {
    feedState.x = 0;
    track.style.transform = 'translateX(0)';
  }
}

function clearFeedEmpty() {
  const track = $('feed-track');
  const ph = track && track.querySelector('.feed__empty');
  if (ph) ph.remove();
}

function showFeedEmpty() {
  const track = $('feed-track');
  if (!track || track.querySelector('.feed__empty')) return;
  const ph = document.createElement('div');
  ph.className = 'feed__empty';
  ph.textContent = 'Waiting for inference activity\u2026';
  track.appendChild(ph);
}

function renderFeedSummary(summary) {
  const host = $('feed-summary');
  if (!host || !summary?.categories?.length) { if (host) host.innerHTML = ''; return; }
  host.innerHTML = summary.categories.map((c) => `
    <span class="feed__cat">
      <span class="feed__dot" style="background:${catColor(c.label)}"></span>
      ${esc(c.label)} <b>${c.count}</b>
    </span>`).join('');
}

async function loadFeed() {
  let d;
  try {
    d = await jget(`./api/prompts/feed?after=${feedState.seen}`);
  } catch {
    return;
  }
  const section = $('feed');
  feedState.enabled = !!d.enabled;
  if (!d.enabled) { if (section) section.hidden = true; return; }
  section.hidden = false;

  const pulse = $('feed-pulse');
  if (pulse) pulse.dataset.live = d.summary?.active ? '1' : '0';

  const fresh = (d.recent || []).filter((e) => e.seq > feedState.seen);
  if (fresh.length) {
    feedState.seen = fresh[fresh.length - 1].seq;
    feedState.pending.push(...fresh);
  }
  renderFeedSummary(d.summary);
}

function startFeed() {
  buildFeedControls();
  showFeedEmpty();
  loadFeed();
  setInterval(loadFeed, 2000);
  feedState.raf = requestAnimationFrame(feedTick);
}

/* --------------------------------------------------------------- header */

function renderHeader() {
  const i = state.info;
  $('hostname').textContent = i.hostname || 'unknown host';
  $('gpu-name').textContent = i.gpu_present
    ? `${i.gpu_name}${i.gpu_count > 1 ? ` \u00d7${i.gpu_count}` : ''}`
    : 'no GPU detected';
  $('fact-driver').textContent = i.driver_version || '\u2014';
  $('fact-cuda').textContent = i.cuda_version || '\u2014';
  const caps = i.capabilities || {};
  $('fact-arch').textContent = caps.wsl
    ? `${i.arch || '\u2014'} \u00b7 WSL2` : (i.arch || '\u2014');
  document.title = `${i.hostname || 'Sparkboard'} \u00b7 Sparkboard`;
}

function tickUptime() {
  const boot = state.info.boot_time;
  $('fact-uptime').textContent = boot ? fmtDuration(Date.now() / 1000 - boot) : '\u2014';
}

/* ------------------------------------------------------------ live stream */

function setDot(stateName) { $('live-dot').dataset.state = stateName; }

function openStream() {
  if (state.es) state.es.close();
  const es = new EventSource('./api/stream');
  state.es = es;

  es.onopen = () => setDot(state.paused ? 'paused' : 'live');
  es.onerror = () => setDot('down');
  es.onmessage = (ev) => {
    if (state.paused) return;
    let payload;
    try { payload = JSON.parse(ev.data); } catch { return; }
    setDot('live');
    state.latest = payload.sample;

    pushLivePoint(payload.point);
    renderReadouts(payload.sample);
    renderPool(payload.sample);
    renderProcs(payload.sample);
    renderCores(payload.sample);

    if (state.range === 'LIVE') renderCharts(state.live);
  };
}

/* --------------------------------------------------------------- ranges */

function selectRange(range) {
  state.range = range;
  for (const b of $('ranges').children) {
    b.setAttribute('aria-pressed', String(b.textContent === range));
  }
  if (state.refreshTimer) { clearInterval(state.refreshTimer); state.refreshTimer = null; }

  loadContainerHistory(range);

  if (range === 'LIVE') {
    loadLive();
    return;
  }
  loadHistory(range);
  // Historical windows still move: refresh often enough to stay current
  // without re-querying a month of buckets every few seconds.
  const secs = RANGE_SECONDS[range] || 3600;
  const every = Math.min(120, Math.max(10, Math.round(secs / 240)));
  state.refreshTimer = setInterval(() => loadHistory(range), every * 1000);
}

/* ----------------------------------------------------------------- boot */

async function refreshFooter() {
  try {
    const h = await jget('./api/health');
    const s = h.store || {};
    $('foot-store').textContent = s.db_bytes != null
      ? `store ${fmtBytes(s.db_bytes, 1)} \u00b7 ${s.raw_rows ?? 0} raw \u00b7 ${s.rollup_rows ?? 0} rolled up`
      : 'store \u2014';
    $('foot-samples').textContent =
      `${h.samples ?? 0} samples${h.errors ? ` \u00b7 ${h.errors} errors` : ''}`;
  } catch {
    $('foot-store').textContent = 'store unreachable';
  }
}

async function boot() {
  buildReadouts();
  buildPanels();
  buildRanges();
  buildContainerToggle();

  $('pause').addEventListener('click', () => {
    state.paused = !state.paused;
    $('pause').setAttribute('aria-pressed', String(state.paused));
    $('pause').textContent = state.paused ? 'Resume' : 'Pause';
    setDot(state.paused ? 'paused' : 'live');
  });

  const coresToggle = $('cores-toggle');
  if (coresToggle) {
    coresToggle.addEventListener('click', () => {
      coresExpanded = !coresExpanded;
      coresToggle.setAttribute('aria-expanded', String(coresExpanded));
      coresToggle.textContent = coresExpanded ? 'Hide cores' : 'Show cores';
      coresBuilt = 0;  // force rebuild of detail cells
      if (state.latest) renderCores(state.latest);
    });
  }

  // Theme toggle. Artifacts can't use localStorage, so this is per-session --
  // the default follows the OS preference on load.
  const themeBtn = $('theme-toggle');
  if (themeBtn) {
    const prefersLight = window.matchMedia
      && window.matchMedia('(prefers-color-scheme: light)').matches;
    let theme = prefersLight ? 'light' : 'dark';
    const apply = () => {
      document.documentElement.setAttribute('data-theme', theme);
      themeBtn.textContent = theme === 'light' ? '☀' : '◐';
      // Charts read CSS vars at draw time, so repaint after a theme flip.
      if (state.latest) { renderReadouts(state.latest); renderCores(state.latest); }
      if (state.range === 'LIVE') renderCharts(state.live);
      else loadHistory(state.range);
    };
    apply();
    themeBtn.addEventListener('click', () => {
      theme = theme === 'light' ? 'dark' : 'light';
      apply();
    });
  }

  // Make every tooltip hint keyboard- and touch-focusable.
  for (const h of document.querySelectorAll('.hint')) {
    h.setAttribute('tabindex', '0');
    h.setAttribute('role', 'button');
  }

  window.addEventListener('resize', () => {
    if (state.latest) renderReadouts(state.latest);
    if (state.lastDocker?.containers) drawRowTrends(state.lastDocker.containers);
  });

  try {
    state.info = await jget('./api/info');
  } catch {
    $('hostname').textContent = 'backend unreachable';
    setDot('down');
    return;
  }
  calibrateBands(state.info);
  // Rebuild the rail now that the part is known: the first build ran before
  // /api/info answered, so its tooltips and bands were the generic ones.
  buildReadouts();
  renderHeader();
  renderStorage(state.info);
  tickUptime();
  setInterval(tickUptime, 30000);

  try {
    const sample = await jget('./api/now');
    state.latest = sample;
    renderPool(sample);
    renderProcs(sample);
    renderReadouts(sample);
    renderCores(sample);
  } catch { /* the stream will fill this in shortly */ }

  await loadLive();
  // loadLive fills the ring the sparklines read from, so repaint the rail
  // now rather than leaving it blank until the first stream message.
  if (state.latest) renderReadouts(state.latest);
  openStream();
  refreshFooter();
  setInterval(refreshFooter, 30000);
  // Disk usage moves slowly, but a model download or an image batch can eat a
  // lot of it in an hour, so it is re-read rather than fixed at boot.
  setInterval(async () => {
    try { renderStorage(await jget('./api/info')); } catch { /* keep last */ }
  }, 60000);

  loadServices();
  setInterval(loadServices, 10000);
  setInterval(tickServicesAge, 1000);

  loadContainerHistory(state.range);
  setInterval(() => loadContainerHistory(state.range), 30000);

  startFeed();
}

boot();
