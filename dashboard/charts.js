/* CampusPulse chart helpers: plain SVG, no libraries.
   Every chart: thin marks, recessive grid, one axis, hover tooltip, and a table view next to it. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tip = () => document.getElementById("tooltip");

  function el(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function svg(container, w, h) {
    container.innerHTML = "";
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
    container.appendChild(s);
    return s;
  }
  function hover(node, html) {
    node.style.cursor = "default";
    node.addEventListener("mousemove", (e) => {
      const t = tip();
      t.innerHTML = html;
      t.style.opacity = 1;
      const x = Math.min(e.clientX + 14, window.innerWidth - t.offsetWidth - 8);
      t.style.left = x + "px";
      t.style.top = (e.clientY + 14) + "px";
    });
    node.addEventListener("mouseleave", () => { tip().style.opacity = 0; });
  }
  function niceTicks(min, max, n = 5) {
    const span = max - min || 1;
    const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || step0;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  /* bar path: rounded 4px at the data end, square at the baseline */
  function hbarPath(x0, x1, y, h) {
    const r = Math.min(4, Math.abs(x1 - x0) / 2, h / 2);
    if (x1 >= x0) return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
    return `M${x0},${y}H${x1 + r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 + r},${y + h}H${x0}Z`;
  }
  function vbarPath(x, w, y0, y1) {
    const r = Math.min(4, w / 2, Math.abs(y0 - y1) / 2);
    return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
  }
  function legend(container, items) {
    const d = document.createElement("div");
    d.className = "legend";
    d.innerHTML = items.map(([c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join("");
    container.prepend(d);
  }
  const width = (c) => Math.max(300, Math.round(c.clientWidth || 600));

  /* Horizontal bars. rows: [{label, value, color?, tip?}], opts: {fmt, min, max, refs:[{value,label}]} */
  function hbars(container, rows, opts = {}) {
    const fmt = opts.fmt || ((v) => v.toFixed(2));
    const W = width(container), labelW = Math.min(opts.labelWidth || 190, W * 0.42), rowH = 34, top = (opts.refs || []).length ? 22 : 6, bottom = 26;
    const H = top + rows.length * rowH + bottom;
    const s = svg(container, W, H);
    const vals = rows.map((r) => r.value).concat((opts.refs || []).map((r) => r.value));
    let min = opts.min ?? Math.min(0, ...vals), max = opts.max ?? Math.max(0, ...vals);
    if (max === min) max = min + 1;
    const x0 = labelW + 8, x1 = W - 56;
    const sx = (v) => x0 + ((v - min) / (max - min)) * (x1 - x0);
    niceTicks(min, max, 4).forEach((t) => {
      el("line", { x1: sx(t), x2: sx(t), y1: top, y2: H - bottom, class: "gridline" }, s);
      const g = el("g", { class: "tick" }, s);
      el("text", { x: sx(t), y: H - 8, "text-anchor": "middle" }, g).textContent = opts.tickFmt ? opts.tickFmt(t) : fmt(t);
    });
    const base = Math.min(Math.max(0, min), max);   // bars start at 0, or at the axis minimum if the axis starts above 0
    el("line", { x1: sx(base), x2: sx(base), y1: top, y2: H - bottom, class: "baseline" }, s);
    rows.forEach((r, i) => {
      const y = top + i * rowH, bh = Math.min(20, rowH - 12), by = y + (rowH - bh) / 2;
      el("text", { x: labelW, y: y + rowH / 2 + 4, "text-anchor": "end" }, s).textContent = r.label;
      if (r.value == null) {
        el("text", { x: x0 + 4, y: y + rowH / 2 + 4, class: "muted" }, s).textContent = "n/a";
        return;
      }
      const p = el("path", { d: hbarPath(sx(base), sx(r.value), by, bh), fill: r.color || "var(--series-1)" }, s);
      // negative bars: print the value just right of the zero line, so it never collides with the row label
      const lx = r.value >= 0 ? sx(r.value) + 6 : sx(base) + 6;
      el("text", { x: lx, y: y + rowH / 2 + 4, "text-anchor": "start", class: "value-label" }, s).textContent = fmt(r.value);
      const hit = el("rect", { x: x0, y, width: x1 - x0 + 50, height: rowH, fill: "transparent" }, s);
      hover(hit, r.tip || `<b>${r.label}</b><br>${fmt(r.value)}`);
      hover(p, r.tip || `<b>${r.label}</b><br>${fmt(r.value)}`);
    });
    (opts.refs || []).forEach((ref) => {
      el("line", { x1: sx(ref.value), x2: sx(ref.value), y1: top - 4, y2: H - bottom, class: "ref" }, s);
      el("text", { x: sx(ref.value), y: top - 9, "text-anchor": "middle", class: "small" }, s).textContent = ref.label;
    });
  }

  /* Grouped columns. groups: ["Students","Staff"], series: [{key,label,color}], value(g, key) */
  function columns(container, groups, series, value, opts = {}) {
    const fmt = opts.fmt || ((v) => v.toFixed(2));
    const W = width(container), H = opts.height || 240, left = 44, right = 10, top = 14, bottom = 30;
    const s = svg(container, W, H);
    const all = groups.flatMap((g) => series.map((se) => value(g, se.key))).filter((v) => v != null);
    const max = opts.max ?? Math.max(...all) * 1.15;
    const sy = (v) => H - bottom - (v / max) * (H - top - bottom);
    niceTicks(0, max, 4).forEach((t) => {
      el("line", { x1: left, x2: W - right, y1: sy(t), y2: sy(t), class: "gridline" }, s);
      const g = el("g", { class: "tick" }, s);
      el("text", { x: left - 6, y: sy(t) + 4, "text-anchor": "end" }, g).textContent = (opts.tickFmt || fmt)(t);
    });
    el("line", { x1: left, x2: W - right, y1: sy(0), y2: sy(0), class: "baseline" }, s);
    const band = (W - left - right) / groups.length, bw = Math.min(24, band / (series.length + 1.5));
    groups.forEach((g, gi) => {
      const cx = left + band * gi + band / 2, start = cx - (series.length * bw + (series.length - 1) * 2) / 2;
      series.forEach((se, si) => {
        const v = value(g, se.key);
        if (v == null) return;
        const x = start + si * (bw + 2);
        const p = el("path", { d: vbarPath(x, bw, sy(0), sy(v)), fill: se.color }, s);
        hover(p, `<b>${g}</b> · ${se.label}<br>${fmt(v)}`);
        el("text", { x: x + bw / 2, y: sy(v) - 5, "text-anchor": "middle", class: "value-label" }, s).textContent = fmt(v);
      });
      el("text", { x: cx, y: H - 10, "text-anchor": "middle" }, s).textContent = g;
    });
    if (series.length > 1) legend(container, series.map((se) => [se.color, se.label]));
  }

  /* Dots over time coloured by a category, optional 7-point raw line. points: [{x: Date, y, cat, tip}] */
  function dots(container, points, cats, opts = {}) {
    const fmt = opts.fmt || ((v) => v.toFixed(0));
    const W = width(container), H = opts.height || 230, left = 46, right = 12, top = 12, bottom = 28;
    const s = svg(container, W, H);
    const xs = points.map((p) => +p.x), ys = points.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    let ymin = opts.ymin ?? Math.min(...ys), ymax = opts.ymax ?? Math.max(...ys);
    const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
    const sx = (v) => left + ((v - xmin) / (xmax - xmin || 1)) * (W - left - right);
    const sy = (v) => H - bottom - ((v - ymin) / (ymax - ymin)) * (H - top - bottom);
    niceTicks(ymin, ymax, 4).forEach((t) => {
      el("line", { x1: left, x2: W - right, y1: sy(t), y2: sy(t), class: "gridline" }, s);
      const g = el("g", { class: "tick" }, s);
      el("text", { x: left - 6, y: sy(t) + 4, "text-anchor": "end" }, g).textContent = fmt(t);
    });
    // month ticks
    const d0 = new Date(xmin);
    for (let d = new Date(d0.getFullYear(), d0.getMonth() + 1, 1); +d <= xmax; d = new Date(d.getFullYear(), d.getMonth() + 1, 1)) {
      const g = el("g", { class: "tick" }, s);
      el("text", { x: sx(+d), y: H - 8, "text-anchor": "middle" }, g).textContent = d.toLocaleString("en", { month: "short" });
    }
    el("line", { x1: left, x2: W - right, y1: H - bottom, y2: H - bottom, class: "baseline" }, s);
    (opts.refs || []).forEach((ref) => {
      el("line", { x1: left, x2: W - right, y1: sy(ref.value), y2: sy(ref.value), class: "ref" }, s);
      el("text", { x: W - right, y: sy(ref.value) - 5, "text-anchor": "end", class: "small" }, s).textContent = ref.label;
    });
    points.forEach((p) => {
      const c = el("circle", { cx: sx(+p.x), cy: sy(p.y), r: 4.5, fill: cats[p.cat].color, stroke: "var(--surface)", "stroke-width": 2 }, s);
      hover(c, p.tip);
    });
    legend(container, Object.values(cats).map((c) => [c.color, c.label]));
  }

  /* Confidence-interval rows: [{label, lower, est, ok}] with reference lines (e.g. -margin and 0) */
  function intervals(container, rows, opts = {}) {
    const fmt = opts.fmt || ((v) => v.toFixed(1));
    const W = width(container), labelW = Math.min(170, W * 0.36), rowH = 40, top = 22, bottom = 26;
    const H = top + rows.length * rowH + bottom;
    const s = svg(container, W, H);
    const vals = rows.flatMap((r) => [r.lower, r.est]).concat((opts.refs || []).map((r) => r.value));
    const min = Math.min(...vals) * 1.15, max = Math.max(...vals, 0) + Math.abs(min) * 0.25;
    const x0 = labelW + 10, x1 = W - 16;
    const sx = (v) => x0 + ((v - min) / (max - min)) * (x1 - x0);
    niceTicks(min, max, 5).forEach((t) => {
      el("line", { x1: sx(t), x2: sx(t), y1: top, y2: H - bottom, class: "gridline" }, s);
      const g = el("g", { class: "tick" }, s);
      el("text", { x: sx(t), y: H - 8, "text-anchor": "middle" }, g).textContent = fmt(t);
    });
    (opts.refs || []).forEach((ref) => {
      el("line", { x1: sx(ref.value), x2: sx(ref.value), y1: top - 6, y2: H - bottom, class: "ref" }, s);
      el("text", { x: sx(ref.value), y: top - 10, "text-anchor": "middle", class: "small" }, s).textContent = ref.label;
    });
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      el("text", { x: labelW, y: y + 4, "text-anchor": "end" }, s).textContent = r.label;
      const colour = r.ok ? "var(--series-1)" : "var(--critical)";
      el("line", { x1: sx(r.lower), x2: sx(r.est), y1: y, y2: y, stroke: colour, "stroke-width": 2, "stroke-linecap": "round" }, s);
      el("rect", { x: sx(r.lower) - 1, y: y - 7, width: 2, height: 14, fill: colour }, s);
      const c = el("circle", { cx: sx(r.est), cy: y, r: 5, fill: colour, stroke: "var(--surface)", "stroke-width": 2 }, s);
      const html = `<b>${r.label}</b><br>difference ${fmt(r.est)}<br>lower 95% bound ${fmt(r.lower)}<br>${r.ok ? "non-inferior" : "not proven non-inferior"}`;
      hover(c, html);
      hover(el("rect", { x: x0, y: y - rowH / 2, width: x1 - x0, height: rowH, fill: "transparent" }, s), html);
    });
  }

  /* One stacked bar per row: rows [{label, parts:{key:n}}], keys [{key,label,color}] */
  function stacked(container, rows, keys) {
    const W = width(container), labelW = Math.min(150, W * 0.3), rowH = 32, top = 4, bottom = 6;
    const H = top + rows.length * rowH + bottom;
    const s = svg(container, W, H);
    const x0 = labelW + 10, x1 = W - 70;
    rows.forEach((r, i) => {
      const y = top + i * rowH, bh = 18, by = y + (rowH - bh) / 2;
      el("text", { x: labelW, y: y + rowH / 2 + 4, "text-anchor": "end" }, s).textContent = r.label;
      const total = keys.reduce((a, k) => a + (r.parts[k.key] || 0), 0) || 1;
      let x = x0;
      keys.forEach((k) => {
        const n = r.parts[k.key] || 0;
        if (!n) return;
        const w = (n / total) * (x1 - x0) - 2;
        const rect = el("rect", { x, y: by, width: Math.max(w, 1), height: bh, rx: 3, fill: k.color }, s);
        hover(rect, `<b>${r.label}</b><br>${k.label}: ${n} of ${total}`);
        x += w + 2;
      });
      el("text", { x: x1 + 8, y: y + rowH / 2 + 4, class: "value-label" }, s).textContent = `${r.parts.holds || 0}/${total} hold`;
    });
    legend(container, keys.map((k) => [k.color, k.label]));
  }

  /* Scatter of 2-D points (PCA / t-SNE / cluster maps). points [{x,y,c}], cats [{key,label,color,ring?}] */
  function scatter(container, points, cats, opts = {}) {
    const W = width(container), H = opts.height || Math.min(360, Math.round(W * 0.72)), m = { l: 36, r: 10, t: 10, b: 30 };
    const s = svg(container, W, H);
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const pad = (a) => { const lo = Math.min(...a), hi = Math.max(...a), d = (hi - lo) * 0.05 || 1; return [lo - d, hi + d]; };
    const [x0, x1] = pad(xs), [y0, y1] = pad(ys);
    const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    if (opts.axes !== false) {
      niceTicks(x0, x1, 5).forEach((t) => {
        el("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, class: "gridline" }, s);
        el("text", { x: sx(t), y: H - m.b + 14, "text-anchor": "middle", class: "tick" }, el("g", { class: "tick" }, s)).textContent = +t.toFixed(1);
      });
      niceTicks(y0, y1, 4).forEach((t) => {
        el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "gridline" }, s);
        el("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end" }, el("g", { class: "tick" }, s)).textContent = +t.toFixed(1);
      });
    }
    el("text", { x: W - m.r, y: H - 4, "text-anchor": "end", class: "small" }, s).textContent = opts.xLabel || "";
    el("text", { x: m.l + 4, y: m.t + 10, class: "small" }, s).textContent = opts.yLabel || "";
    const byKey = Object.fromEntries(cats.map((c) => [c.key, c]));
    // draw the largest group first so smaller groups stay visible on top
    const order = cats.map((c) => c.key).sort((a, b) =>
      points.filter((p) => p.c === b).reduce((sum, p) => sum + (p.n || 1), 0)
      - points.filter((p) => p.c === a).reduce((sum, p) => sum + (p.n || 1), 0));
    const coordinateGroups = new Map();
    points.forEach((p) => {
      const key = `${p.x},${p.y}`;
      if (!coordinateGroups.has(key)) coordinateGroups.set(key, []);
      coordinateGroups.get(key).push(p);
    });
    const radiusFor = (p) => p.n ? Math.min(9, 3 + 0.18 * Math.sqrt(p.n)) : 4;
    order.forEach((key) => points.filter((p) => p.c === key).forEach((p) => {
      const c = byKey[key] || { color: "var(--muted)", label: key };
      const count = p.n || 1;
      const coordinatePeers = coordinateGroups.get(`${p.x},${p.y}`);
      const peers = [...new Set(coordinatePeers.map((point) => point.c))];
      if (opts.offsetCategoryToRight && peers.includes(opts.offsetCategoryToRight)) {
        peers.splice(peers.indexOf(opts.offsetCategoryToRight), 1);
        peers.push(opts.offsetCategoryToRight);
      }
      const peerRadius = Math.max(...coordinatePeers.map(radiusFor));
      const offsetX = peers.length > 1 ? (peers.indexOf(p.c) - (peers.length - 1) / 2) * (peerRadius * 2 + 1) : 0;
      const radius = radiusFor(p);
      const cx = sx(p.x) + offsetX;
      const tipText = p.n
        ? `<b>${c.label}</b><br>${count.toLocaleString()} students<br>x ${p.x.toFixed(2)} · y ${p.y.toFixed(2)}`
        : `<b>${c.label}</b><br>x ${p.x.toFixed(2)} · y ${p.y.toFixed(2)}`;
      const dot = el("circle", c.ring
        ? { cx, cy: sy(p.y), r: radius, fill: "none", stroke: c.color, "stroke-width": 1.5, opacity: 0.85 }
        : { cx, cy: sy(p.y), r: radius, fill: c.color, stroke: "var(--surface)", "stroke-width": 1, "fill-opacity": 0.75 }, s);
      hover(dot, tipText);
    }));
    legend(container, cats.map((c) => [c.color, `${c.label} (${points.filter((p) => p.c === c.key).reduce((sum, p) => sum + (p.n || 1), 0)})`]));
  }

  /* Histogram, single series. bins [{lo,hi,n}] */
  function histogram(container, bins, opts = {}) {
    const W = width(container), H = opts.height || 200, m = { l: 40, r: 8, t: 12, b: 28 };
    const s = svg(container, W, H);
    const max = Math.max(...bins.map((b) => b.n)) * 1.1 || 1;
    const lo = bins[0].lo, hi = bins[bins.length - 1].hi;
    const sx = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v / max) * (H - m.t - m.b);
    niceTicks(0, max, 4).forEach((t) => {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "gridline" }, s);
      el("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end" }, el("g", { class: "tick" }, s)).textContent = t.toLocaleString();
    });
    el("line", { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0), class: "baseline" }, s);
    const fmt = opts.fmt || ((v) => +v.toFixed(2));
    const every = Math.ceil(bins.length / 8);
    bins.forEach((b, i) => {
      const x = sx(b.lo) + 1, w = Math.max(1, sx(b.hi) - sx(b.lo) - 2);
      const p = el("path", { d: vbarPath(x, w, sy(0), sy(b.n)), fill: (opts.highlight && opts.highlight(b)) ? "var(--series-2)" : "var(--series-1)" }, s);
      hover(p, `<b>${fmt(b.lo)} – ${fmt(b.hi)}</b><br>${b.n.toLocaleString()} ${opts.unit || "rows"}`);
      if (i % every === 0) el("text", { x: sx(b.lo), y: H - 10, "text-anchor": "middle" }, el("g", { class: "tick" }, s)).textContent = fmt(b.lo);
    });
    el("text", { x: sx(hi), y: H - 10, "text-anchor": "end" }, el("g", { class: "tick" }, s)).textContent = fmt(hi);
  }

  /* Lines over a numeric x axis. series: [{label, color, points:[{x, y}]}] */
  function lines(container, series, opts = {}) {
    const W = width(container), H = opts.height || 240, m = { l: 44, r: 12, t: 12, b: 30 };
    const s = svg(container, W, H);
    const all = series.flatMap((se) => se.points);
    const x0 = Math.min(...all.map((p) => p.x)), x1 = Math.max(...all.map((p) => p.x));
    const y1 = Math.max(...all.map((p) => p.y)) * 1.1, y0 = 0;
    const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    const fmt = opts.fmt || ((v) => v.toFixed(0));
    niceTicks(y0, y1, 4).forEach((t) => {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "gridline" }, s);
      el("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end" }, el("g", { class: "tick" }, s)).textContent = fmt(t);
    });
    (opts.xTicks || niceTicks(x0, x1, 6)).forEach((t) => {
      el("text", { x: sx(t), y: H - 10, "text-anchor": "middle" }, el("g", { class: "tick" }, s)).textContent = opts.xFmt ? opts.xFmt(t) : t;
    });
    el("line", { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0), class: "baseline" }, s);
    series.forEach((se) => {
      const d = se.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join("");
      el("path", { d, fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, s);
      se.points.forEach((p) => {
        const c = el("circle", { cx: sx(p.x), cy: sy(p.y), r: 3, fill: se.color, stroke: "var(--surface)", "stroke-width": 1 }, s);
        hover(c, `<b>${se.label}</b><br>${opts.xFmt ? opts.xFmt(p.x) : p.x}: ${fmt(p.y)}${opts.unit || ""}`);
      });
    });
    legend(container, series.map((se) => [se.color, se.label]));
  }

  window.Charts = { hbars, columns, dots, intervals, stacked, scatter, histogram, lines };
})();
