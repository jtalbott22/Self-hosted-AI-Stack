/*
 * Strip-chart rendering for Sparkboard.
 *
 * Deliberately dependency-free: the machine this runs on may sit on an
 * isolated network, so there is no CDN fetch and no build step. Everything
 * draws to a 2D canvas.
 *
 * Nulls are drawn as gaps rather than interpolated across. A break in the
 * trace means "no data here", which on a monitoring dashboard is information
 * -- it usually means the service was down.
 */

const PALETTE = {
  gpu: '#6FE3A8',
  mem: '#7AA2F7',
  therm: '#FFA657',
  power: '#D2A8FF',
  cpu: '#56D4DD',
  alert: '#FF7B72',
  dim: '#8B97A6',
};

/* ---------------------------------------------------------------- format */

function fmtNum(v, digits = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return '\u2014';
  return v.toFixed(digits);
}

function fmtBytes(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '\u2014';
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  let i = 0;
  let n = Math.abs(v);
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i === 0 ? 0 : digits)} ${units[i]}`;
}

function fmtRate(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '\u2014';
  return `${fmtBytes(v, 1)}/s`;
}



function fmtDuration(sec) {
  if (sec === null || sec === undefined || Number.isNaN(sec)) return '\u2014';
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

/* Axis tick selection: 1 / 2 / 2.5 / 5 x 10^n, so labels land on round numbers. */
function niceTicks(min, max, target = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { ticks: [0, 1], min: 0, max: 1 };
  if (max === min) { max = min + 1; }
  const raw = (max - min) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  let step;
  if (norm <= 1) step = 1;
  else if (norm <= 2) step = 2;
  else if (norm <= 2.5) step = 2.5;
  else if (norm <= 5) step = 5;
  else step = 10;
  step *= mag;

  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) {
    ticks.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  }
  return { ticks, min: lo, max: hi };
}

function timeFormatter(spanSeconds) {
  const pad = (n) => String(n).padStart(2, '0');
  if (spanSeconds <= 900) {
    return (t) => {
      const d = new Date(t * 1000);
      return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    };
  }
  if (spanSeconds <= 86400 * 1.2) {
    return (t) => {
      const d = new Date(t * 1000);
      return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    };
  }
  if (spanSeconds <= 86400 * 8) {
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    return (t) => {
      const d = new Date(t * 1000);
      return `${days[d.getDay()]} ${pad(d.getHours())}:00`;
    };
  }
  return (t) => {
    const d = new Date(t * 1000);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  };
}

/* ----------------------------------------------------------------- chart */

class StripChart {
  /*
   * opts:
   *   series   [{ key, label, color, fill, maxKey }]
   *   yMin/yMax  fixed axis bounds (omit to autoscale)
   *   format   value -> label, used on axis and tooltip
   *   unit     appended in the tooltip
   */
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.opts = Object.assign({ series: [], format: (v) => fmtNum(v, 0), unit: '' }, opts);
    this.t = [];
    this.data = {};
    this.hover = null;
    this.dpr = window.devicePixelRatio || 1;

    this.pad = { top: 10, right: 10, bottom: 20, left: 44 };

    this._resize = this._resize.bind(this);
    this._onMove = this._onMove.bind(this);
    this._onLeave = this._onLeave.bind(this);

    this.ro = new ResizeObserver(this._resize);
    this.ro.observe(canvas.parentElement);
    canvas.addEventListener('pointermove', this._onMove);
    canvas.addEventListener('pointerleave', this._onLeave);

    this._resize();
  }

  destroy() {
    this.ro.disconnect();
    this.canvas.removeEventListener('pointermove', this._onMove);
    this.canvas.removeEventListener('pointerleave', this._onLeave);
  }

  _resize() {
    const box = this.canvas.parentElement.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;
    this.dpr = window.devicePixelRatio || 1;
    this.w = box.width;
    this.h = box.height;
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
    this.canvas.style.width = `${this.w}px`;
    this.canvas.style.height = `${this.h}px`;
    this.draw();
  }

  setData(t, data) {
    this.t = t || [];
    this.data = data || {};
    this.draw();
  }

  _onMove(ev) {
    const rect = this.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    if (!this.t.length) return;
    const { left, right } = this._plotBox();
    const frac = (x - left) / Math.max(1, right - left);
    const idx = Math.round(frac * (this.t.length - 1));
    this.hover = Math.max(0, Math.min(this.t.length - 1, idx));
    this.draw();
    if (this.onHover) this.onHover(this.hover, ev);
  }

  _onLeave() {
    this.hover = null;
    this.draw();
    if (this.onHover) this.onHover(null);
  }

  _plotBox() {
    return {
      left: this.pad.left,
      right: this.w - this.pad.right,
      top: this.pad.top,
      bottom: this.h - this.pad.bottom,
    };
  }

  _bounds() {
    const { yMin, yMax } = this.opts;
    if (yMin !== undefined && yMax !== undefined) {
      return niceTicks(yMin, yMax, 4);
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of this.opts.series) {
      for (const key of [s.key, s.maxKey]) {
        const arr = key && this.data[key];
        if (!arr) continue;
        for (const v of arr) {
          if (v === null || v === undefined || Number.isNaN(v)) continue;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    if (!Number.isFinite(lo)) return niceTicks(0, 1, 4);
    if (yMin !== undefined) lo = yMin;
    // Anchor to zero unless the data sits well away from it -- a clock chart
    // reading 1.60-1.62 GHz is unreadable if forced down to a zero baseline.
    if (lo > 0 && lo < hi * 0.55) lo = 0;
    const padding = (hi - lo) * 0.08;
    return niceTicks(lo, hi + padding, 4);
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.w) return;
    ctx.save();
    ctx.scale(this.dpr, this.dpr);
    ctx.clearRect(0, 0, this.w, this.h);

    const { left, right, top, bottom } = this._plotBox();
    const plotW = right - left;
    const plotH = bottom - top;
    if (plotW <= 0 || plotH <= 0) { ctx.restore(); return; }

    const css = getComputedStyle(document.documentElement);
    const ruleColor = css.getPropertyValue('--rule').trim() || '#242D39';
    const dimColor = css.getPropertyValue('--ink-faint').trim() || '#5A6675';

    // A timeline with no finite values still has length, so length alone is
    // not enough to decide there is something to plot. Drawing an axis for an
    // all-null series yields a scale of repeated zeros, which reads like a
    // real measurement of nothing -- exactly the wrong thing to show for a
    // sensor the hardware never reported.
    let hasData = false;
    outer: for (const s of this.opts.series) {
      for (const key of [s.key, s.maxKey]) {
        const arr = key && this.data[key];
        if (!arr) continue;
        for (const v of arr) {
          if (v !== null && v !== undefined && !Number.isNaN(v)) { hasData = true; break outer; }
        }
      }
    }

    if (!this.t.length || !hasData) {
      ctx.fillStyle = dimColor;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(this.t.length ? 'not reported for this range' : 'no data for this range',
                   left + plotW / 2, top + plotH / 2);
      ctx.restore();
      return;
    }

    const b = this._bounds();
    const yScale = (v) => bottom - ((v - b.min) / (b.max - b.min)) * plotH;
    const n = this.t.length;
    const xScale = (i) => (n <= 1 ? left : left + (i / (n - 1)) * plotW);

    /* horizontal grid + y labels */
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const tick of b.ticks) {
      const y = Math.round(yScale(tick)) + 0.5;
      if (y < top - 1 || y > bottom + 1) continue;
      ctx.strokeStyle = ruleColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
      ctx.fillStyle = dimColor;
      ctx.fillText(this.opts.format(tick), left - 6, y);
    }

    /* x labels */
    const span = this.t[n - 1] - this.t[0];
    const fmtT = timeFormatter(span);
    const xCount = Math.max(2, Math.min(6, Math.floor(plotW / 82)));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (let k = 0; k <= xCount; k += 1) {
      const i = Math.round((k / xCount) * (n - 1));
      const x = Math.round(xScale(i)) + 0.5;
      ctx.strokeStyle = ruleColor;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = dimColor;
      ctx.fillText(fmtT(this.t[i]), x, bottom + 5);
    }

    /* peak band: at wide buckets an averaged spike vanishes, so the bucket
       maximum is drawn behind the mean as a faint envelope */
    for (const s of this.opts.series) {
      if (!s.maxKey) continue;
      const maxArr = this.data[s.maxKey];
      const avgArr = this.data[s.key];
      if (!maxArr || !avgArr) continue;
      let differs = false;
      for (let i = 0; i < n; i += 1) {
        if (maxArr[i] !== null && avgArr[i] !== null
            && Math.abs(maxArr[i] - avgArr[i]) > 1e-6) { differs = true; break; }
      }
      if (!differs) continue;

      ctx.fillStyle = s.color;
      ctx.globalAlpha = 0.13;
      let open = false;
      ctx.beginPath();
      for (let i = 0; i < n; i += 1) {
        const v = maxArr[i];
        if (v === null || v === undefined) { open = false; continue; }
        if (!open) { ctx.moveTo(xScale(i), yScale(v)); open = true; }
        else ctx.lineTo(xScale(i), yScale(v));
      }
      for (let i = n - 1; i >= 0; i -= 1) {
        const v = avgArr[i];
        if (v === null || v === undefined) continue;
        ctx.lineTo(xScale(i), yScale(v));
      }
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    /* traces */
    for (const s of this.opts.series) {
      const arr = this.data[s.key];
      if (!arr) continue;

      if (s.fill !== false) {
        const grad = ctx.createLinearGradient(0, top, 0, bottom);
        grad.addColorStop(0, `${s.color}38`);
        grad.addColorStop(1, `${s.color}00`);
        ctx.fillStyle = grad;
        let open = false;
        let startI = 0;
        for (let i = 0; i < n; i += 1) {
          const v = arr[i];
          const missing = v === null || v === undefined || Number.isNaN(v);
          if (missing) {
            if (open) {
              ctx.lineTo(xScale(i - 1), bottom);
              ctx.lineTo(xScale(startI), bottom);
              ctx.closePath();
              ctx.fill();
              open = false;
            }
            continue;
          }
          if (!open) {
            ctx.beginPath();
            ctx.moveTo(xScale(i), yScale(v));
            startI = i;
            open = true;
          } else {
            ctx.lineTo(xScale(i), yScale(v));
          }
        }
        if (open) {
          ctx.lineTo(xScale(n - 1), bottom);
          ctx.lineTo(xScale(startI), bottom);
          ctx.closePath();
          ctx.fill();
        }
      }

      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.6;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      let open = false;
      ctx.beginPath();
      for (let i = 0; i < n; i += 1) {
        const v = arr[i];
        if (v === null || v === undefined || Number.isNaN(v)) { open = false; continue; }
        if (!open) { ctx.moveTo(xScale(i), yScale(v)); open = true; }
        else ctx.lineTo(xScale(i), yScale(v));
      }
      ctx.stroke();
    }

    /* crosshair */
    if (this.hover !== null && this.hover < n) {
      const x = Math.round(xScale(this.hover)) + 0.5;
      ctx.strokeStyle = css.getPropertyValue('--ink-dim').trim() || '#8B97A6';
      ctx.globalAlpha = 0.55;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      for (const s of this.opts.series) {
        const v = this.data[s.key] && this.data[s.key][this.hover];
        if (v === null || v === undefined || Number.isNaN(v)) continue;
        const y = yScale(v);
        ctx.fillStyle = css.getPropertyValue('--housing').trim() || '#0E1116';
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.8;
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  tooltipFor(idx) {
    if (idx === null || idx >= this.t.length) return null;
    const rows = [];
    for (const s of this.opts.series) {
      const v = this.data[s.key] && this.data[s.key][idx];
      rows.push({
        label: s.label,
        color: s.color,
        value: (v === null || v === undefined || Number.isNaN(v))
          ? '\u2014'
          : `${this.opts.format(v)}${this.opts.unit}`,
      });
    }
    return { t: this.t[idx], rows };
  }
}

/* Inline sparkline for the readout rail: no axes, no interaction. */
function drawSparkline(canvas, values, color, height = 26) {
  const dpr = window.devicePixelRatio || 1;
  const box = canvas.parentElement.getBoundingClientRect();
  const w = box.width;
  const h = height;
  if (w <= 0) return;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const pts = values.filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  if (pts.length < 2) { ctx.restore(); return; }

  let lo = Math.min(...pts);
  let hi = Math.max(...pts);
  if (hi - lo < 1e-6) { hi = lo + 1; }
  const pad = (hi - lo) * 0.15;
  lo -= pad; hi += pad;

  const n = values.length;
  const x = (i) => (i / (n - 1)) * w;
  const y = (v) => h - 2 - ((v - lo) / (hi - lo)) * (h - 4);

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, `${color}30`);
  grad.addColorStop(1, `${color}00`);
  ctx.fillStyle = grad;
  ctx.beginPath();
  let started = false;
  let startI = 0;
  for (let i = 0; i < n; i += 1) {
    const v = values[i];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (!started) { ctx.moveTo(x(i), y(v)); startI = i; started = true; }
    else ctx.lineTo(x(i), y(v));
  }
  if (started) {
    ctx.lineTo(x(n - 1), h);
    ctx.lineTo(x(startI), h);
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.3;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  started = false;
  for (let i = 0; i < n; i += 1) {
    const v = values[i];
    if (v === null || v === undefined || Number.isNaN(v)) { started = false; continue; }
    if (!started) { ctx.moveTo(x(i), y(v)); started = true; }
    else ctx.lineTo(x(i), y(v));
  }
  ctx.stroke();
  ctx.restore();
}

export {
  PALETTE, StripChart, drawSparkline, niceTicks, timeFormatter,
  fmtNum, fmtBytes, fmtRate, fmtDuration,
};
