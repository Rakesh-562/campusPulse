/* CampusPulse dashboard.
 *
 * How it works
 *   1. scripts/export_dashboard_data.py runs the pipelines and writes every number into data.js
 *      as one object: window.CAMPUSPULSE.
 *   2. This file only reads that object and shows it. It never computes a result.
 *
 * Pages (chosen in the top bar, kept in the URL):
 *   #/                     overview
 *   #/<pulse>/<panel>      one pulse, one deliverable panel (e.g. #/academic/statistics)
 *   #/robustness           every robustness matrix
 *   #/method               method, limitations, why we changed dataset
 */

const DATA = window.CAMPUSPULSE;
const PULSES = ["academic", "engagement", "wellbeing", "food", "energy"];

// The deliverables the course asks for, shown as buttons on every pulse page.
const PANELS = [
  { id: "findings", label: "Findings" },
  { id: "processing", label: "Data processing" },
  { id: "statistics", label: "Statistics" },
  { id: "reduction", label: "Dimensionality reduction" },
  { id: "rules", label: "Association rules" },
  { id: "clustering", label: "Clustering" },
];

// Short names for findings (the full claim is shown underneath).
const SHORT = {
  A1: "Debtors drop out more", A2: "Scholarship holders drop out less", A3: "Dropouts have lower semester-1 grades",
  A4: "Debt matters within each grade band",
  E1: "Least active learners withdraw more", E2: "Persisters score higher early", E3: "Withdrawers' activity falls faster",
  W1: "Sleep-deprived students report more sadness", W2: "Bullied students report more sadness",
  W3: "Inactive students report more sadness", W4: "Course is associated with depression (exploratory)",
  F1: "More vegetarian meals did not cost sales", F2: "Vegetarian share rises in intervention weeks",
  F3: "Students choose vegetarian more than staff",
  EN1: "Academic building uses more on teaching days", EN2: "Hostels use more at night in term than vacation",
  EN3: "Term-vs-vacation difference within the same month",
};

/* ------------------------------------------------------------------ small helpers */

function $(selector) { return document.querySelector(selector); }

function esc(text) {
  return String(text ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function isMissing(v) { return v === null || v === undefined || Number.isNaN(v); }

function number(v, digits = 2) {
  if (isMissing(v)) return "n/a";
  return Number(v).toLocaleString("en", { maximumFractionDigits: digits, minimumFractionDigits: Math.min(digits, 2) });
}

function percent(v, digits = 1) { return isMissing(v) ? "n/a" : (v * 100).toFixed(digits) + "%"; }

function signed(v, digits = 2, unit = "") { return isMissing(v) ? "n/a" : (v >= 0 ? "+" : "") + number(v, digits) + unit; }

function points(v) { return isMissing(v) ? "n/a" : signed(v * 100, 1, " pp"); }   // percentage points

function pValue(p) { return isMissing(p) ? "n/a" : p < 0.0001 ? "< 0.0001" : p.toFixed(4); }

// How to print each finding's value.
function formatFinding(id, v) {
  if (["A1", "A2", "A4", "E1", "W1", "W2", "W3", "F2", "F3"].includes(id)) return points(v);
  if (id === "A3") return signed(v, 2, " grade pts");
  if (id === "E2") return signed(v, 1, " score pts");
  if (id === "W4") return isMissing(v) ? "n/a" : "V = " + number(v, 3);
  if (id === "F1") return signed(v, 1, " meals/day");
  if (id.startsWith("EN")) return signed(v, 0, " W");
  return signed(v, 3);
}

function shortName(id, claim) { return SHORT[id] || claim; }

function niceVariant(v) { return v === "DEFAULT" ? "our default" : v.replace("=", " = ").replace(/_/g, " "); }

function niceItem(item) {
  const names = { dropout_1: "dropout", choice_veg: "vegetarian/vegan meal", withdrawn_withdrawn: "withdrawal",
                  sadness: "persistent sadness", power_demand_hi_power: "high campus demand" };
  return item.split(", ").map((t) => names[t] || t.replace(/_/g, " ")).join(", ");
}

// A status chip: coloured dot + word, so colour is never the only signal.
function chip(status) {
  const cls = status === "n/a" ? "na" : status;
  return `<span class="chip ${cls}"><i></i>${esc(status)}</span>`;
}

/* Build an HTML table.
 * columns: [{ label, value: row => text, numeric: true/false, html: true/false }] */
function table(rows, columns) {
  if (!rows || rows.length === 0) return `<p class="muted small">Nothing to show.</p>`;
  let html = `<div class="table-wrap"><table><thead><tr>`;
  for (const c of columns) html += `<th class="${c.numeric ? "num" : ""}">${esc(c.label)}</th>`;
  html += `</tr></thead><tbody>`;
  for (const row of rows) {
    html += "<tr>";
    for (const c of columns) {
      const v = c.value(row);
      html += `<td class="${c.numeric ? "num" : ""}">${c.html ? v : esc(v)}</td>`;
    }
    html += "</tr>";
  }
  return html + "</tbody></table></div>";
}

function card(title, subtitle, body) {
  return `<div class="card"><h3>${title}</h3>${subtitle ? `<div class="sub">${subtitle}</div>` : ""}${body}</div>`;
}

function chartBox(id) { return `<div class="chart" id="${id}"></div>`; }

/* Charts are drawn after the HTML is on the page, and again when the window is resized. */
let pendingCharts = [];
function addChart(drawFunction) { pendingCharts.push(drawFunction); }
function drawCharts() { for (const draw of pendingCharts) draw(); }

/* ------------------------------------------------------------------ overview page */

function overviewPage() {
  // The three findings that one cleaning choice moves the most, across all pulses with data.
  let fragile = [];
  for (const p of PULSES) {
    const pulse = DATA.pulses[p];
    if (!pulse.available) continue;
    for (const f of pulse.robustness.fragility) if (f.verdict === "fragile") fragile.push({ ...f, pulse: p });
  }
  const size = (f) => (f.change === "sign flips" ? 1e9 : parseFloat(f.change.slice(1)));
  fragile.sort((a, b) => size(b) - size(a));
  fragile = fragile.slice(0, 3);

  let fragileCards = "";
  for (const f of fragile) {
    const meta = DATA.pulses[f.pulse].meta;
    fragileCards += `<div class="card"><div class="sub">${meta.num} ${meta.title} · ${f.finding}</div>
      <h3>${esc(shortName(f.finding, f.claim))}</h3>
      <p class="small">Default <b>${formatFinding(f.finding, f.default)}</b> → with <code>${esc(niceVariant(f.worst_variant))}</code>
      <b>${formatFinding(f.finding, f.worst_value)}</b> (${esc(f.change)}).</p></div>`;
  }

  let pulseCards = "";
  for (const p of PULSES) {
    const pulse = DATA.pulses[p];
    const firstId = Object.keys(pulse.findings)[0];
    const body = pulse.available
      ? `<div class="tile"><div class="label">${firstId} · ${esc(shortName(firstId, pulse.findings[firstId]))}</div>
         <div class="value compact">${formatFinding(firstId, pulse.robustness.values[firstId].DEFAULT)}</div>
         <div class="note">${pulse.robustness.summary.holds}/${pulse.robustness.summary.tested} checks hold</div></div>`
      : `<p class="small muted">Data not exported on this machine.</p>`;
    pulseCards += `<a class="card link" href="#/${p}/findings"><div class="sub">${pulse.meta.num} ${pulse.meta.title}</div>
      <h3>${esc(pulse.meta.question)}</h3>${body}</a>`;
  }

  $("#app").innerHTML = `
    <h1>CampusPulse</h1>
    <p class="lede">Five real datasets, each a "vital sign" of a university. Every cleaning decision is logged and every
    finding is re-computed under each reasonable alternative, so we know which conclusions are solid.</p>
    <div class="question"><b>Umbrella question.</b> Can routine campus data warn a university early enough to act, and how
    much do those warnings depend on our data-cleaning decisions?</div>
    <h2>The five vital signs</h2><div class="grid g5">${pulseCards}</div>
    <h2>Cleaning choices that change the answer</h2><div class="grid g3">${fragileCards}</div>`;
}

/* ------------------------------------------------------------------ pulse page */

function pulsePage(p, panelId) {
  const pulse = DATA.pulses[p];
  const meta = pulse.meta;
  const header = `
    <div class="pulse-head">
      <div><div class="sub">${meta.num} ${meta.title} · <span class="badge">REAL DATA</span></div>
      <h1>${esc(meta.question)}</h1>
      <p class="muted small">${esc(meta.dataset)} · ${esc(meta.source)} · ${esc(meta.licence)}</p></div>
    </div>`;

  if (!pulse.available) {
    $("#app").innerHTML = header + `<div class="callout warn">Data not exported on this machine: ${esc(pulse.reason || "")}<br>
      Run the download and preprocess steps in README.md, then <code>python scripts/export_dashboard_data.py</code>.</div>`;
    return;
  }

  let buttons = "";
  for (const panel of PANELS) {
    buttons += `<a class="pill ${panel.id === panelId ? "active" : ""}" href="#/${p}/${panel.id}">${panel.label}</a>`;
  }
  const panelFunctions = {
    findings: findingsPanel, processing: processingPanel, statistics: statisticsPanel,
    reduction: reductionPanel, rules: rulesPanel, clustering: clusteringPanel,
  };
  const show = panelFunctions[panelId] || findingsPanel;

  $("#app").innerHTML = header + `<nav class="pills" aria-label="Deliverables">${buttons}</nav><div id="panel"></div>`;
  $("#panel").innerHTML = show(p, pulse, pulse.extras);
  if (show === findingsPanel) connectExplorer(pulse);
}

/* --- Findings: headline numbers, the decision explorer and the pulse's key chart */

function findingsPanel(p, pulse, X) {
  let selects = "";
  for (const [name, options] of Object.entries(pulse.decisions)) {
    let opts = "";
    options.forEach((o, i) => { opts += `<option value="${esc(o)}">${esc(o.replace(/_/g, " "))}${i === 0 ? " (default)" : ""}</option>`; });
    selects += `<label>${esc(name.replace(/_/g, " "))}<select data-decision="${esc(name)}">${opts}</select></label>`;
  }
  const explorer = card("Try a different cleaning choice",
    "Each finding is recomputed with one decision changed (the others stay at our default).",
    `<div class="controls" id="explorer">${selects}</div><div id="explorer-table"></div>`);
  return `<div class="question small">${esc(pulse.meta.challenge)}</div>` + explorer + keyChart(p, X);
}

function connectExplorer(pulse) {
  const R = pulse.robustness;
  const selects = Array.from(document.querySelectorAll("#explorer select"));
  function show(variant) {
    const rows = Object.keys(pulse.findings);
    $("#explorer-table").innerHTML = table(rows, [
      { label: "Finding", html: true, value: (id) => `<b>${id} · ${esc(shortName(id, pulse.findings[id]))}</b><div class="small muted">${esc(pulse.findings[id])}</div>` },
      { label: "Our default", numeric: true, value: (id) => formatFinding(id, R.values[id].DEFAULT) },
      { label: variant === "DEFAULT" ? "Selected" : niceVariant(variant), numeric: true, value: (id) => formatFinding(id, R.values[id][variant]) },
      { label: "Status", html: true, value: (id) => chip(variant === "DEFAULT" ? "holds" : R.status[id][variant]) },
    ]);
  }
  for (const s of selects) {
    s.addEventListener("change", () => {
      for (const other of selects) if (other !== s) other.selectedIndex = 0;     // one change at a time
      show(s.selectedIndex === 0 ? "DEFAULT" : `${s.dataset.decision}=${s.value}`);
    });
  }
  show("DEFAULT");
}

// The one chart that tells each pulse's main story.
function keyChart(p, X) {
  if (p === "academic") {
    addChart(() => Charts.hbars($("#c-key"), X.timeline.map((r) => ({ label: r.stage.replace(/^\d /, ""), value: r.capture_at_25pct,
      tip: `<b>${r.stage}</b><br>reaches ${percent(r.capture_at_25pct)} of dropouts<br>ROC AUC ${number(r.roc_auc, 3)}` })),
      { fmt: (v) => percent(v, 0), min: 0, max: 1, refs: [{ value: X.timeline[0].max_possible_capture, label: "max possible" }], labelWidth: 140 }));
    addChart(() => Charts.hbars($("#c-key2"), X.debt_by_band.map((r) => ({ label: r.band, value: r.gap })), { fmt: points, labelWidth: 110 }));
    return `<div class="grid g2" style="margin-top:16px">
      ${card("How early can the university warn?", "Share of eventual dropouts reached by contacting the 25% highest-risk students (cross-validated logistic regression)", chartBox("c-key"))}
      ${card("Does debt matter beyond grades? (A4)", "Dropout rate of debtors minus non-debtors, inside each semester-1 approval band", chartBox("c-key2"))}</div>`;
  }
  if (p === "engagement") {
    addChart(() => Charts.hbars($("#c-key"), X.timeline.map((r) => ({ label: r.stage.replace("_", " "), value: r.roc_auc,
      tip: `<b>${r.stage}</b><br>ROC AUC ${number(r.roc_auc, 3)}<br>reaches ${percent(r.capture_at_25pct)} of withdrawals` })),
      { fmt: (v) => number(v, 2), min: 0.5, max: 1, labelWidth: 70 }));
    return `<div class="grid g2" style="margin-top:16px">
      ${card("How well do early clicks predict withdrawal?", "ROC AUC of 'fewest clicks = highest risk' at each cutoff (0.5 = coin flip, 1 = perfect)", chartBox("c-key"))}
      ${card("Most of the obvious signal was leakage", "",
        `<div class="tile"><div class="value">${points(X.leakage.gap_including)} → ${points(X.leakage.gap_excluding)}</div>
         <div class="note">Withdrawal gap, least vs most active quarter, before and after removing the ${X.leakage.n_left_before_cutoff.toLocaleString()}
         students who had already left by week 5 (of ${X.leakage.n_withdrawn.toLocaleString()} withdrawals). Their low clicks were a result of leaving, not a warning.</div></div>`)}</div>`;
  }
  if (p === "wellbeing") {
    addChart(() => Charts.hbars($("#c-key"), X.sleep_gradient.map((r) => ({ label: (r.sleep_hours === 4 ? "≤4" : r.sleep_hours === 10 ? "10+" : r.sleep_hours) + " h", value: r.rate })),
      { fmt: (v) => percent(v, 0), min: 0, labelWidth: 50 }));
    let tiles = "";
    for (const [option, a] of Object.entries(X.course_association)) {
      tiles += `<div class="tile"><div class="label">${option === "raw" ? "Raw course text" : "Cleaned courses"} (${a.n_groups} groups)</div>
        <div class="value">V = ${number(a.cramers_v, 2)}</div><div class="note">p = ${pValue(a.p)}</div></div>`;
    }
    return `<div class="grid g2" style="margin-top:16px">
      ${card("Persistent sadness by hours of sleep", "Weighted share of students (CDC YRBS 2023). Association, not cause.", chartBox("c-key"))}
      ${card("Why the text cleaning matters (W4, Malaysia survey)", "Dozens of one-person 'courses' inflate the association by sparsity alone",
        `<div class="grid g2">${tiles}</div>`)}</div>`;
  }
  if (p === "food") {
    const margin = X.non_inferiority[0].margin;
    const names = { weekday_adjusted: "weekday adjusted", raw: "raw totals", rolling_5day: "5-day rolling" };
    addChart(() => Charts.intervals($("#c-key"), X.non_inferiority.map((r) => ({ label: names[r.option] || r.option, lower: r.lower_95, est: r.diff, ok: r.non_inferior })),
      { refs: [{ value: -margin, label: "−5% margin" }, { value: 0, label: "0" }], fmt: (v) => v.toFixed(0) }));
    addChart(() => Charts.columns($("#c-key2"), ["Students", "Staff"],
      [{ key: "Basis", label: "Baseline week", color: "var(--series-1)" }, { key: "Intervention", label: "Intervention week", color: "var(--series-2)" }],
      (g, k) => (X.veg_share_by_group.find((r) => r.group === g && r.condit === k) || {}).veg_share, { fmt: (v) => percent(v, 0), max: 0.6 }));
    return `<div class="grid g2" style="margin-top:16px">
      ${card("Is the intervention proven 'not worse' on sales? (F1)", "Difference in meals/day and its one-sided 95% lower bound. Blue = proven non-inferior, red = not proven.", chartBox("c-key"))}
      ${card("Share of meals that were vegetarian or vegan", "Randomised weeks, so this difference is causal", chartBox("c-key2"))}</div>`;
  }
  if (p === "energy") {
    addChart(() => Charts.hbars($("#c-key"), X.by_month.map((r) => ({ label: r.month, value: r.diff,
      tip: `<b>${r.month}</b><br>term ${number(r.term_mean, 0)} W · vacation ${number(r.vacation_mean, 0)} W` })),
      { fmt: (v) => signed(v / 1000, 1, " kW"), labelWidth: 70 }));
    return `<div class="grid g2" style="margin-top:16px">
      ${card("Is the term effect just the season? (EN3)", "Night-time hostel power, term minus vacation, inside months that contain both", chartBox("c-key"))}
      ${card("What the calendar can tell us", "", `<p class="small">The institute calendar marks teaching days and low-activity days (weekends,
        holidays, vacations). It does <b>not</b> mark exams, so this pulse makes no claim about exams. Delhi's long vacation falls in the
        hottest months, which is why EN3 compares term and vacation inside the same month.</p>`)}</div>`;
  }
  return "";
}

/* --- Data processing: the data challenge and the cleaning log */

function processingPanel(p, pulse, X) {
  let challenge = "";
  if (p === "academic") {
    const z = X.hidden_zeros;
    addChart(() => Charts.hbars($("#c-proc"), X.a3_by_zero_grade.map((r) => ({
      label: { nan_plus_flag: "0 = missing (default)", as_score: "0 = real score", drop: "drop those rows" }[r.option] || r.option,
      value: r.value, color: r.option === "as_score" ? "var(--critical)" : "var(--series-1)" })), { fmt: (v) => v.toFixed(2), labelWidth: 150 }));
    challenge = `<div class="grid g3">
      <div class="card tile"><div class="label">Enrolled in units but graded 0</div><div class="value">${z.n}</div>
        <div class="note">${percent(z.dropout_rate)} dropped out (others: ${percent(z.dropout_rate_others)}). A 0 here is a missing exam, not a score.</div></div>
      <div class="card tile"><div class="label">No enrolled units in semester 1</div><div class="value">${z.no_units}</div>
        <div class="note">${z.no_units_graduated} still graduated, so 0 is not failure.</div></div>
      ${card("Grade gap (A3) under each treatment of 0", "Persisters minus dropouts, out of 20", chartBox("c-proc"))}</div>`;
  }
  if (p === "engagement") {
    challenge = card("Outcome leakage", "", `<p class="small">${X.leakage.n_left_before_cutoff.toLocaleString()} of ${X.leakage.n_withdrawn.toLocaleString()}
      withdrawals happened before week 5. Keeping them makes early clicks look like a strong warning (gap ${points(X.leakage.gap_including)});
      removing them leaves ${points(X.leakage.gap_excluding)}. The default removes them (decision <code>already left = exclude</code>).</p>`);
  }
  if (p === "wellbeing") {
    const labels = { sad: "Sadness", sleep_code: "Sleep", active_code: "Physical activity", social_media_code: "Social media", bullied_school: "Bullied at school" };
    addChart(() => Charts.hbars($("#c-proc"), X.skipped.map((r) => ({ label: labels[r.question] || r.question, value: r.share })), { fmt: (v) => percent(v, 1), min: 0, labelWidth: 130 }));
    challenge = card("Share of students who skipped each question", "Skipping is far more common for some questions, so 'drop incomplete answers' is not neutral", chartBox("c-proc"));
  }
  if (p === "food") {
    const cats = { Basis: { label: "Baseline week", color: "var(--series-1)" }, Intervention: { label: "Intervention week", color: "var(--series-2)" } };
    const dots = (key) => X.daily.map((d) => ({ x: new Date(d.date), y: d[key], cat: d.condit, tip: `<b>${d.weekday} ${d.date}</b><br>${number(d[key], 0)} meals` }));
    const all = X.daily.flatMap((d) => [d.sales_raw, d.sales_adjusted]);
    const range = { ymin: Math.min(...all), ymax: Math.max(...all) };
    addChart(() => Charts.dots($("#c-proc"), dots("sales_raw"), cats, range));
    addChart(() => Charts.dots($("#c-proc2"), dots("sales_adjusted"), cats, range));
    challenge = `<div class="grid g2">
      ${card("Meals sold per day, raw", "Fridays are much quieter: noise for this question", chartBox("c-proc"))}
      ${card("Same days, weekday effect removed", "Each day minus its weekday average (the default)", chartBox("c-proc2"))}</div>`;
  }
  if (p === "energy") {
    addChart(() => Charts.hbars($("#c-proc"), X.gaps.map((r) => ({ label: r.meter.replace("_", " "), value: r.share })), { fmt: (v) => percent(v, 0), min: 0, labelWidth: 110 }));
    challenge = card("Hours with no reading, per meter", "The hostel meters were silent in about a quarter of all hours", chartBox("c-proc"));
  }
  const log = table(pulse.log, [
    { label: "Step", value: (r) => r.step },
    { label: "Choice", value: (r) => r.decision || "–" },
    { label: "Rows before", numeric: true, value: (r) => (r.rows_before ? r.rows_before.toLocaleString() : "–") },
    { label: "Rows after", numeric: true, value: (r) => r.rows_after.toLocaleString() },
    { label: "Why", value: (r) => r.why },
  ]);
  return challenge + `<div style="margin-top:16px">${card("Cleaning log (default choices)", "", log)}</div>`;
}

/* --- Statistics: descriptive table, distributions and hypothesis tests */

function statisticsPanel(p, pulse, X) {
  const describe = table(X.describe, [
    { label: "Variable", value: (r) => r.variable },
    { label: "n", numeric: true, value: (r) => r.n.toLocaleString() },
    { label: "Missing", numeric: true, value: (r) => r.missing.toLocaleString() },
    { label: "Mean", numeric: true, value: (r) => number(r.mean) },
    { label: "SD", numeric: true, value: (r) => number(r.sd) },
    { label: "Median", numeric: true, value: (r) => number(r.median) },
    { label: "IQR", numeric: true, value: (r) => `${number(r.q1)} – ${number(r.q3)}` },
    { label: "Min – max", numeric: true, value: (r) => `${number(r.min)} – ${number(r.max)}` },
    { label: "Skew", numeric: true, value: (r) => number(r.skew) },
  ]);
  let histograms = "";
  (X.histograms || []).forEach((h, i) => {
    addChart(() => Charts.histogram($(`#c-hist-${i}`), h.bins, { highlight: h.label.startsWith("Semester-1 grade") ? (b) => b.lo === 0 : null }));
    histograms += card(esc(h.label), esc(h.note || ""), chartBox(`c-hist-${i}`));
  });
  const tests = table(X.tests, [
    { label: "Test", value: (r) => r.test },
    { label: "Statistic", numeric: true, value: (r) => number(r.statistic) },
    { label: "p-value", numeric: true, value: (r) => pValue(r.p) },
    { label: "Effect size", numeric: true, value: (r) => (isMissing(r.effect) ? "–" : `${number(r.effect, 3)} ${r.effect_name || ""}`) },
  ]);
  return card("Descriptive statistics", "After the default cleaning choices", describe)
    + (histograms ? `<div class="grid g2" style="margin-top:16px">${histograms}</div>` : "")
    + `<div style="margin-top:16px">${card("Hypothesis tests", "With thousands of rows almost any difference is 'significant', so read the effect size", tests)}</div>`;
}

/* --- Dimensionality reduction: PCA, t-SNE, variance explained, loadings */

const GROUP_COLOURS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
function groupStyles(names) {
  // Only three colours stay distinguishable for every pair; a fourth group is drawn as grey rings.
  return names.map((n, i) => (i < 3 ? { key: n, label: n, color: GROUP_COLOURS[i] } : { key: n, label: n + " (rings)", color: "var(--muted)", ring: true }));
}

function reductionPanel(p, pulse, X) {
  const D = X.dr;
  if (!D) return `<p class="muted">Not available for this pulse.</p>`;
  const v = D.variance;
  const styles = groupStyles(D.categories);
  addChart(() => Charts.scatter($("#c-pca"), D.pca, styles, { xLabel: `PC1 (${percent(v[0], 0)})`, yLabel: `PC2 (${percent(v[1], 0)})` }));
  addChart(() => Charts.scatter($("#c-tsne"), D.tsne, styles, { axes: false, xLabel: "t-SNE 1", yLabel: "t-SNE 2" }));
  addChart(() => {
    let cumulative = 0;
    const rows = v.map((x, i) => { cumulative += x; return { label: `PC${i + 1}`, value: x, tip: `<b>PC${i + 1}</b><br>${percent(x)}; cumulative ${percent(cumulative)}` }; });
    Charts.hbars($("#c-scree"), rows, { fmt: (x) => percent(x, 1), min: 0, labelWidth: 50 });
  });
  const loadings = table(D.top_loadings, [
    { label: "Feature", value: (r) => r.feature },
    { label: "PC1", numeric: true, value: (r) => number(r.pc1) },
    { label: "PC2", numeric: true, value: (r) => number(r.pc2) },
  ]);
  const overlap = D.n_unique < 200
    ? ` Only ${D.n_unique} different answer combinations exist, so many students sit on exactly the same point.` : "";
  return `<p class="muted small">${D.n_rows.toLocaleString()} rows × ${D.n_features} standardised features. Colour shows the outcome,
      which the methods never see, so any separation is structure already in the data.${overlap}</p>
    <div class="grid g2">
      ${card("PCA: first two components", `Together they explain ${percent(v[0] + v[1], 0)} of the variance`, chartBox("c-pca"))}
      ${card("t-SNE", `Non-linear map of a ${D.tsne.length}-row sample; only local neighbourhoods are meaningful`, chartBox("c-tsne"))}
      ${card("Variance explained by each component", "", chartBox("c-scree"))}
      ${card("What PC1 and PC2 are made of", "Features with the largest loadings", loadings)}
    </div>`;
}

/* --- Association rules */

function rulesPanel(p, pulse, X) {
  const rules = table(X.rules, [
    { label: "If a record has…", value: (r) => niceItem(r.antecedents) },
    { label: "…then", value: (r) => niceItem(r.consequents) },
    { label: "Support", numeric: true, value: (r) => percent(r.support) },
    { label: "Confidence", numeric: true, value: (r) => percent(r.confidence) },
    { label: "Lift", numeric: true, value: (r) => number(r.lift) },
  ]);
  return card("Apriori rules with the outcome as the 'then' side",
    "Support = how common the whole combination is. Confidence = how often the 'then' follows. Lift above 1 = more often than chance. Co-occurrence, not cause.", rules);
}

/* --- Clustering */

function clusteringPanel(p, pulse, X) {
  const M = X.cluster_map;
  const C = X.clusters;
  const methodNames = { kmeans: "K-Means", agglo: "Ward (hierarchical)", dbscan: "DBSCAN", optics: "OPTICS" };
  const metrics = table((M && M.metrics) || (C && C.metrics) || [], [
    { label: "Method", value: (r) => methodNames[r.method] || r.method },
    { label: "Groups", numeric: true, value: (r) => r.n_clusters },
    { label: "Unassigned", numeric: true, value: (r) => percent(r.noise_share, 0) },
    { label: "Silhouette", numeric: true, value: (r) => number(r.silhouette, 3) },
    { label: "Davies–Bouldin", numeric: true, value: (r) => number(r.davies_bouldin) },
  ]);
  let profiles = "";
  if (C && C.profiles && C.profiles.length) {
    const keys = Object.keys(C.profiles[0]);
    profiles = table(C.profiles, keys.map((k) => ({
      label: k.replace(/_/g, " "),
      numeric: !/cluster|group/.test(k),
      value: (r) => {
        const v = r[k];
        if (/cluster|group/.test(k)) return typeof v === "number" ? "Group " + (v + 1) : v;   // e.g. "DBSCAN noise"
        if (typeof v !== "number") return v ?? "–";
        if (/rate|share/.test(k)) return percent(v);
        return Number.isInteger(v) ? v.toLocaleString() : number(v, 2);
      },
    })));
  }
  if (M) addChart(() => Charts.scatter($("#c-clu"), M.points, groupStyles(M.groups), { xLabel: "PC1", yLabel: "PC2" }));
  return `<div class="grid g2">
      ${M ? card("K-Means groups on the first two principal components", `Sample of ${M.points.length.toLocaleString()}; DBSCAN left ${percent(M.noise_share, 0)} outside every group`, chartBox("c-clu")) : ""}
      ${card("Four methods compared", "Higher silhouette and lower Davies–Bouldin = tighter, better-separated groups", metrics)}
    </div>
    ${profiles ? `<div style="margin-top:16px">${card("What each group looks like", "", profiles)}</div>` : ""}`;
}

/* ------------------------------------------------------------------ robustness page */

function robustnessPage() {
  let html = `<h1>How fragile are campus insights?</h1>
    <p class="lede">Each finding, recomputed after changing one cleaning decision. It <b>holds</b> if it keeps its sign and stays
    between half and double our default. <b>Inflates</b> means the choice creates most of the effect.</p>
    <div class="legend">${chip("holds")} ${chip("weakens")} ${chip("inflates")} ${chip("reverses")} ${chip("n/a")}</div>`;
  for (const p of PULSES) {
    const pulse = DATA.pulses[p];
    if (!pulse.available) continue;
    const R = pulse.robustness;
    const variants = R.variants.filter((v) => v !== "DEFAULT");
    const columns = [
      { label: "Finding", html: true, value: (id) => `<b>${id}</b> ${esc(shortName(id, pulse.findings[id]))}` },
      { label: "Default", numeric: true, value: (id) => formatFinding(id, R.values[id].DEFAULT) },
    ];
    for (const v of variants) {
      columns.push({ label: niceVariant(v), html: true, value: (id) => `${chip(R.status[id][v])}<div class="small muted">${formatFinding(id, R.values[id][v])}</div>` });
    }
    html += `<details class="card" style="margin-bottom:12px" ${p === "academic" ? "open" : ""}>
      <summary><b>${pulse.meta.num} ${pulse.meta.title}</b> · ${R.summary.holds}/${R.summary.tested} hold</summary>
      ${table(Object.keys(pulse.findings), columns)}</details>`;
  }
  $("#app").innerHTML = html;
}

/* ------------------------------------------------------------------ method page */

function methodPage() {
  const F = DATA.forensics;
  $("#app").innerHTML = `<h1>Method</h1>
    <ol class="findings">
      <li><b>Real data only.</b> Each dataset is downloaded and documented (source, licence, size).</li>
      <li><b>Clean with a log.</b> Each step records rows before and after, and why.</li>
      <li><b>Name the choices.</b> Where analysts could reasonably differ, the choice is a decision with options.</li>
      <li><b>Analyse:</b> descriptive statistics, hypothesis tests with effect sizes, PCA / t-SNE, association rules, clustering.</li>
      <li><b>Stress-test:</b> recompute every finding under each alternative choice.</li>
    </ol>
    <h2>Why we changed dataset</h2>
    ${F && F.available ? `<p>Our first dataset (student lifestyle) was generated, not collected: in ${F.rows_summing_to_24.toLocaleString()} of
      ${F.rows.toLocaleString()} rows the five "hours per day" add up to exactly 24, and a depth-3 decision tree reproduces its stress level
      with ${percent(F.tree_accuracy[2].accuracy, 0)} accuracy. Its "findings" were just the formula that generated it.</p>` : ""}
    <h2>Limitations</h2>
    <ul>
      <li>Five institutions in four countries, none of them ours: we compare conclusions, never join rows.</li>
      <li>Only the canteen experiment is randomised; everything else is association.</li>
      <li>YRBS is a high-school survey; the Malaysia survey has 101 respondents.</li>
      <li>The energy calendar has no exam dates, so we compare teaching days and vacations instead.</li>
      <li>Decisions are changed one at a time; combinations are not tested.</li>
    </ul>
    <p class="small muted">Data exported ${esc(DATA.generated)}.</p>`;
}

/* ------------------------------------------------------------------ navigation */

function topBar(current) {
  let links = `<a href="#/" class="${current === "" ? "active" : ""}">Overview</a>`;
  for (const p of PULSES) {
    const m = DATA.pulses[p].meta;
    links += `<a href="#/${p}/findings" class="${current === p ? "active" : ""}">${m.num} ${m.title}</a>`;
  }
  links += `<a href="#/robustness" class="${current === "robustness" ? "active" : ""}">Robustness</a>`;
  links += `<a href="#/method" class="${current === "method" ? "active" : ""}">Method</a>`;
  $("#tabs").innerHTML = links;
}

function showPage() {
  const [page = "", panel = "findings"] = location.hash.replace(/^#\/?/, "").split("/");
  pendingCharts = [];
  topBar(page);
  if (PULSES.includes(page)) pulsePage(page, panel);
  else if (page === "robustness") robustnessPage();
  else if (page === "method") methodPage();
  else overviewPage();
  drawCharts();
}

function setUpThemeButton() {
  try {
    const saved = localStorage.getItem("cp-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* storage blocked: follow the system setting */ }
  $("#theme").addEventListener("click", () => {
    const now = document.documentElement.getAttribute("data-theme")
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = now === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("cp-theme", next); } catch (e) { /* ignore */ }
    drawCharts();
  });
}

if (!DATA) {
  document.getElementById("app").innerHTML = "<p>data.js is missing. Run <code>python scripts/export_dashboard_data.py</code>.</p>";
} else {
  setUpThemeButton();
  window.addEventListener("hashchange", showPage);
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawCharts, 150); });
  showPage();
}
