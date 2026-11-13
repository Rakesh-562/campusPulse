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
 *   #/conclusions          conclusions and limitations
 */

const DATA = window.CAMPUSPULSE;
const PULSES = ["academic", "wellbeing", "energy"];

// The deliverables the course asks for, in the order we present them: data processing first, findings last.
const PANELS = [
  { id: "processing", label: "Data processing" },
  { id: "statistics", label: "Statistics" },
  { id: "reduction", label: "Dimensionality reduction" },
  { id: "rules", label: "Association rules" },
  { id: "clustering", label: "Clustering" },
  { id: "findings", label: "Findings" },
];

// Short names for findings (the full claim is shown underneath).
const SHORT = {
  A1: "Debtors drop out more", A2: "Scholarship holders drop out less", A3: "Dropouts have lower semester-1 grades",
  A4: "Debt matters within each grade band",
  W1: "Sleep-deprived students report more sadness", W2: "Bullied students report more sadness",
  W3: "Inactive students report more sadness",
  EN1: "Academic building uses more on teaching days", EN2: "Hostels use more at night in term than vacation",
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
  if (["A1", "A2", "A4", "W1", "W2", "W3"].includes(id)) return points(v);
  if (id === "A3") return signed(v, 2, " grade pts");
  if (id.startsWith("EN")) return signed(v / 1000, 1, " kW");
  return signed(v, 3);
}

function shortName(id, claim) { return SHORT[id] || claim; }

function displayedFindingIds(p, pulse) {
  return Object.keys(pulse.findings).filter((id) => !(p === "energy" && id === "EN3"));
}

function displayedRobustnessCounts(pulse, ids) {
  const variants = pulse.robustness.variants.filter((variant) => variant !== "DEFAULT");
  const statuses = ids.flatMap((id) => variants.map((variant) => pulse.robustness.status[id][variant]));
  return { holds: statuses.filter((status) => status === "holds").length, tested: statuses.length };
}

function niceVariant(v) { return v === "DEFAULT" ? "our default" : v.replace("=", " = ").replace(/_/g, " "); }

function niceItem(item) {
  const names = { dropout_1: "dropout", sadness: "persistent sadness", power_demand_hi_power: "high campus demand" };
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

// One line that states a simple hypothesis test in words.
function testLine(t) {
  if (!t) return "";
  const p = pValue(t.p);
  return `<p class="small"><b>${esc(t.test)}.</b> H0: ${esc(t.null)}. ${esc(t.statistic_name)} = ${number(t.statistic)},
    p ${p.startsWith("<") ? p : "= " + p} → <b>${esc(t.decision)}</b> at the 5% level.</p>`;
}

/* ------------------------------------------------------------------ overview page */

function overviewPage() {
  let cards = "";
  for (const p of PULSES) {
    const m = DATA.pulses[p].meta;
    cards += `<a class="card link" href="#/${p}/processing"><div class="sub">${m.num} ${m.title}</div>
      <h3>${esc(m.question)}</h3><p class="small muted">${esc(m.dataset)} · ${esc(m.source)}</p></a>`;
  }
  const pipeline = `
    <ol class="pipeline-flow">
      <li class="pipeline-step">
        <span class="pipeline-num">01</span>
        <div class="pipeline-title">Load the raw data</div>
        <div class="pipeline-desc">Open datasets go into a standard pipeline</div>
      </li>
      <li class="pipeline-step">
        <span class="pipeline-num">02</span>
        <div class="pipeline-title">Log the decisions</div>
        <div class="pipeline-desc">Rows in and out, the reason, and 2 to 3 options per choice</div>
      </li>
      <li class="pipeline-step">
        <span class="pipeline-num">03</span>
        <div class="pipeline-title">Clean with a setting</div>
        <div class="pipeline-desc">The same code runs with each option</div>
      </li>
      <li class="pipeline-step">
        <span class="pipeline-num">04</span>
        <div class="pipeline-title">Analyse</div>
        <div class="pipeline-desc">Hypothesis tests and machine learning</div>
      </li>
      <li class="pipeline-step">
        <span class="pipeline-num">05</span>
        <div class="pipeline-title">Check robustness</div>
        <div class="pipeline-desc">Check whether findings hold under different data-cleaning choices</div>
      </li>
    </ol>`;

  $("#app").innerHTML = `
    <h1>CampusPulse</h1>
    <div class="question"><b>Umbrella question.</b> Can routine campus data warn a university early enough to act, and how
    much do those warnings depend on our data-cleaning decisions?</div>
    <h2>Three questions, three real datasets</h2><div class="grid g3">${cards}</div>
    <h2>Standard data pipeline for all pulses</h2>${pipeline}`;
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
  PANELS.forEach((panel, i) => {
    buttons += `<a class="pill ${panel.id === panelId ? "active" : ""}" href="#/${p}/${panel.id}">${i + 1}. ${panel.label}</a>`;
  });
  const panelFunctions = {
    processing: processingPanel, statistics: statisticsPanel, reduction: reductionPanel,
    rules: rulesPanel, clustering: clusteringPanel, findings: findingsPanel,
  };
  const show = panelFunctions[panelId] || processingPanel;

  $("#app").innerHTML = header + `<nav class="pills" aria-label="Deliverables">${buttons}</nav><div id="panel"></div>`;
  $("#panel").innerHTML = show(p, pulse, pulse.extras);
  if (show === findingsPanel) connectExplorer(p, pulse);
}

/* --- 1. Data processing: the data challenge, its graphs, and the cleaning log */

function processingPanel(p, pulse, X) {
  let challenge = "";
  if (p === "academic") {
    const z = X.hidden_zeros;
    const grades = (X.histograms || []).find((h) => h.label.startsWith("Semester-1 grade"));
    if (grades) addChart(() => Charts.histogram($("#c-processing-grades"), grades.bins, { highlight: (b) => b.lo === 0 }));

    challenge = `<div class="grid g2">
      <div class="card tile"><div class="label">Enrolled in units but graded 0 (raw file)</div><div class="value">${z.n}</div>
        <div class="note">${percent(z.dropout_rate)} dropped out (others: ${percent(z.dropout_rate_others)}). A 0 here is a missing exam, not a score,
        so the grade is set to missing and flagged. ${z.n - z.n_after_enrolled} were still 'Enrolled' and are removed with them, leaving ${z.n_after_enrolled}.</div></div>
      <div class="card tile"><div class="label">No enrolled units in semester 1</div><div class="value">${z.no_units}</div>
        <div class="note">${z.no_units_graduated} still graduated, so 0 is not failure. Kept under our default choice.</div></div>
    </div>${grades ? `<div style="margin-top:16px">${card(esc(grades.label), esc(grades.note || ""), chartBox("c-processing-grades"))}</div>` : ""}`;
  }
  if (p === "wellbeing") {
    const labels = { sad: "Sadness", sleep_code: "Sleep", active_code: "Physical activity", social_media_code: "Social media",
      bullied_school: "Bullied at school", bullied_online: "Bullied online", sex: "Sex" };
    const skipped = (X.skipped || []).map((r) => ({
      label: labels[r.question] || r.question,
      value: r.share,
      tip: `<b>${esc(labels[r.question] || r.question)}</b><br>${percent(r.share, 1)} skipped`,
    })).sort((a, b) => b.value - a.value);
    addChart(() => Charts.hbars($("#c-wb-missing"), skipped, {
      fmt: (v) => percent(v, 1), min: 0, max: 0.25, tickFmt: (v) => percent(v, 0), labelWidth: 150,
    }));
    challenge = `<div class="card">
      <h3>Missing-answer patterns in survey data</h3>
      <div class="sub">Skipping is far more common for some questions, so dropping incomplete answers is not neutral</div>
      ${chartBox("c-wb-missing")}
      <p class="small muted" style="margin-top:10px">Survey answers are recoded into readable 0/1 indicators. Complete-case analysis removes records with missing fields, while missing-flags preserve sample size.</p>
    </div>`;
  }
  if (p === "energy") {
    const meterLabels = {
      Academic: "Academic building", Lecture: "Lecture building", Library: "Library", Mess: "Mess",
      Facilities: "Facilities", Boys_main: "Boys hostel (main)", Boys_backup: "Boys hostel (backup)",
      Girls_main: "Girls hostel (main)", Girls_backup: "Girls hostel (backup)",
    };
    const gaps = (X.gaps || []).map((r) => ({
      label: meterLabels[r.meter] || r.meter.replace(/_/g, " "),
      value: r.share,
      tip: `<b>${esc(meterLabels[r.meter] || r.meter.replace(/_/g, " "))}</b><br>${percent(r.share, 1)} of hourly readings missing`,
    })).sort((a, b) => b.value - a.value);
    addChart(() => Charts.hbars($("#c-energy-gaps"), gaps, {
      fmt: (v) => percent(v, 1), min: 0, max: 0.3, tickFmt: (v) => percent(v, 0), labelWidth: 170,
    }));
    challenge = `<div class="card">
      <h3>Missing hourly readings by meter</h3>
      <div class="sub">Hostel meters miss about a quarter of hourly readings; the default choice drops hours without readings</div>
      ${chartBox("c-energy-gaps")}
    </div>`;
  }
  const log = table(pulse.log, [
    { label: "Step", value: (r) => r.step },
    { label: "Choice", value: (r) => r.decision || "–" },
    { label: "Rows before", numeric: true, value: (r) => (r.rows_before ? r.rows_before.toLocaleString() : "–") },
    { label: "Rows after", numeric: true, value: (r) => r.rows_after.toLocaleString() },
    { label: "Why", value: (r) => r.why },
  ]);
  return `<div class="question small">${esc(pulse.meta.challenge)}</div>` + challenge
    + `<div style="margin-top:16px">${card("Cleaning log (default choices)", "", log)}</div>`;
}

/* --- 2. Statistics: descriptive table, distributions and one simple test per finding */

function statisticsPanel(p, pulse, X) {
  let statOverview = "";
  if (p === "energy") {
    addChart(() => Charts.lines($("#c-energy-load-curve"), [
      { label: "Teaching days", color: "var(--series-2)", points: X.load_curve.map((r) => ({ x: r.hour, y: r.teaching })) },
      { label: "Low-activity days", color: "var(--series-1)", points: X.load_curve.map((r) => ({ x: r.hour, y: r.low })) },
    ], { fmt: (v) => v.toFixed(0), unit: " kW", xTicks: [0, 3, 6, 9, 12, 15, 18, 21, 23], xFmt: (h) => `${h}:00` }));
    statOverview = `<div style="margin-bottom:16px">
      ${card("A day in the academic building",
        "Mean power at each hour (kW). Teaching days rise from about 8:00 and stay high until evening; nights are almost the same on both.",
        chartBox("c-energy-load-curve"))}
      <div class="callout" style="margin-top:16px">The institute calendar marks teaching days and low-activity days (weekends,
      holidays, vacations). It does <b>not</b> mark exams, so this pulse makes no claim about exams. Delhi's long vacation falls in the
      hottest months, so term-versus-vacation comparisons may also reflect seasonal differences.</div>
    </div>`;
  }
  if (p === "wellbeing") {
    const weightedBars = (rows) => rows.map((row) => ({
      label: row.label,
      value: row.share,
      tip: `<b>${esc(row.label)}</b><br>Weighted share: ${percent(row.share, 1)}<br>${row.n.toLocaleString()} respondents`,
    }));
    addChart(() => Charts.hbars($("#c-wb-sleep-dist"), weightedBars(X.sleep_distribution), {
      fmt: (v) => percent(v, 0), min: 0, labelWidth: 100,
    }));
    addChart(() => Charts.hbars($("#c-wb-indicator-prev"), weightedBars(X.indicator_prevalence), {
      fmt: (v) => percent(v, 0), min: 0, labelWidth: 180,
    }));
    statOverview = `<div class="grid g2" style="margin-bottom:16px">
      ${card("School-night sleep distribution", "Weighted share by reported sleep category", chartBox("c-wb-sleep-dist"))}
      ${card("Derived wellbeing indicators", "Weighted prevalence of the yes/no measures used in the analysis", chartBox("c-wb-indicator-prev"))}
    </div>`;
  }
  if (p === "academic" && X.outcomes && pulse.evidence.A1 && pulse.evidence.A2) {
    const outcomeColors = { Graduate: "var(--series-1)", Dropout: "var(--critical)", Enrolled: "var(--muted)" };
    const outcomeBars = X.outcomes.map((o) => ({
      label: o.label,
      value: o.share,
      color: outcomeColors[o.label] || "var(--series-1)",
      tip: `<b>${esc(o.label)}</b><br>${percent(o.share, 1)} (${o.n.toLocaleString()} of 4,424 students)`
    }));
    addChart(() => Charts.hbars($("#c-stat-outcomes"), outcomeBars, {
      fmt: (v) => percent(v, 1),
      min: 0,
      max: 0.6,
      labelWidth: 110
    }));

    const outcomeGroups = {
      debt: [
        { label: "Debtors", evidence: "A1", category: "Debtors" },
        { label: "Non-debtors", evidence: "A1", category: "Non-debtors" },
      ],
      scholarship: [
        { label: "No scholarship", evidence: "A2", category: "No scholarship" },
        { label: "Scholarship", evidence: "A2", category: "Scholarship" },
      ],
    };
    for (const [comparison, groups] of Object.entries(outcomeGroups)) {
      const rates = groups.map((group) => ({
        label: group.label,
        dropout: pulse.evidence[group.evidence].bars.find((bar) => bar.label === group.category).value,
      }));
      addChart(() => Charts.columns(
        $(`#c-stat-${comparison}-outcomes`),
        rates.map((group) => group.label),
        [
          { key: "dropout", label: "Dropout", color: "var(--critical)" },
          { key: "nonDropout", label: "Non-dropout", color: "var(--series-1)" },
        ],
        (label, key) => {
          const group = rates.find((row) => row.label === label);
          if (!group) return null;
          return key === "dropout" ? group.dropout : 1 - group.dropout;
        },
        { fmt: (value) => percent(value, 0), tickFmt: (value) => percent(value, 0), max: 1, height: 260 },
      ));
    }

    const ages = (X.histograms || []).find((h) => h.label.startsWith("Age"));
    if (ages) addChart(() => Charts.histogram($("#c-stat-ages"), ages.bins, { fmt: (v) => v.toFixed(0), unit: "students" }));

    statOverview = `<div class="grid g2" style="margin-bottom:16px">
      ${card("Cohort outcomes & percentages (Dropouts vs Graduates)", "Full cohort of 4,424 students: 32.1% drop out, 49.9% graduate, 17.9% enrolled", chartBox("c-stat-outcomes"))}
      ${card("Dropout and non-dropout by student group", "Share within each group; comparisons are separated for readability",
        `<div class="grid g2">
          <div><h4>Debt status</h4>${chartBox("c-stat-debt-outcomes")}</div>
          <div><h4>Scholarship status</h4>${chartBox("c-stat-scholarship-outcomes")}</div>
        </div>`)}
      ${ages ? card("Outliers: age at enrolment", esc(ages.note || ""), chartBox("c-stat-ages")) : ""}
    </div>`;
  }
  const describe = table(X.describe, [
    { label: "Variable", value: (r) => r.variable },
    { label: "n", numeric: true, value: (r) => r.n.toLocaleString() },
    { label: "Missing", numeric: true, value: (r) => r.missing.toLocaleString() },
    { label: "Mean", numeric: true, value: (r) => number(r.mean) },
    { label: "SD", numeric: true, value: (r) => number(r.sd) },
    { label: "Median", numeric: true, value: (r) => number(r.median) },
    { label: "IQR", numeric: true, value: (r) => `${number(r.q1)} – ${number(r.q3)}` },
    { label: "Min – max", numeric: true, value: (r) => `${number(r.min)} – ${number(r.max)}` },
  ]);
  const ids = displayedFindingIds(p, pulse);
  const tests = table(ids, [
    { label: "Finding", html: true, value: (id) => `<b>${id}</b> ${esc(shortName(id, pulse.findings[id]))}` },
    { label: "Test", value: (id) => pulse.evidence[id].test.test },
    { label: "Null hypothesis (H0)", value: (id) => pulse.evidence[id].test.null },
    { label: "Statistic", numeric: true, value: (id) => `${pulse.evidence[id].test.statistic_name} = ${number(pulse.evidence[id].test.statistic)}` },
    { label: "p-value", numeric: true, value: (id) => pValue(pulse.evidence[id].test.p) },
    { label: "Decision (5%)", value: (id) => pulse.evidence[id].test.decision },
  ]);
  let bandAnalysis = "";
  if (p === "academic" && X.debt_by_band && X.debt_by_band.length) {
    const bands = X.debt_by_band;
    const bandNames = {
      "low (<=50%)": "Low ≤50%",
      mid: "Mid >50–99%",
      "all passed": "All passed >99%",
    };
    const bandTests = (pulse.evidence.A4.tests || []);
    const strongest = bands.reduce((best, row) => Math.abs(row.gap) > Math.abs(best.gap) ? row : best);
    const bandTable = table(bands, [
      { label: "Semester-1 approval band", value: (r) => bandNames[r.band] || r.band },
      { label: "Debtors' dropout rate", numeric: true, value: (r) => percent(r.debtor_dropout_rate) },
      { label: "Non-debtors' dropout rate", numeric: true, value: (r) => percent(r.non_debtor_dropout_rate) },
      { label: "Difference (debtors − non-debtors)", numeric: true, value: (r) => points(r.gap) },
      { label: "Debtors / non-debtors (n)", numeric: true, value: (r) => `${r.n_debtors.toLocaleString()} / ${r.n_non_debtors.toLocaleString()}` },
      { label: "Within-band z-test p", numeric: true, value: (r) => pValue((bandTests.find((t) => t.band === r.band) || {}).p) },
    ]);
    addChart(() => Charts.columns(
      $("#c-debt-bands"),
      bands.map((r) => bandNames[r.band] || r.band),
      [
        { key: "debtor_dropout_rate", label: "Debtors", color: "var(--series-2)" },
        { key: "non_debtor_dropout_rate", label: "Non-debtors", color: "var(--series-1)" },
      ],
      (label, key) => {
        const row = bands.find((r) => (bandNames[r.band] || r.band) === label);
        return row ? row[key] : null;
      },
      { fmt: (v) => percent(v, 0), tickFmt: (v) => percent(v, 0), max: 1, height: 260 },
    ));
    bandAnalysis = `<div style="margin-top:16px">${card(
      "Debt and dropout within semester-1 approval bands",
      "Approval rate = approved units ÷ enrolled units. The chart compares observed dropout rates for debtors and non-debtors within each band.",
      `${chartBox("c-debt-bands")}${bandTable}
       <p class="small band-interpretation"><b>Interpretation:</b> the largest observed debtor/non-debtor gap is in the ${esc(bandNames[strongest.band] || strongest.band)} band
       (${points(strongest.gap)}; ${strongest.n_debtors.toLocaleString()} debtors and ${strongest.n_non_debtors.toLocaleString()} non-debtors).
       A4's average gap across the included bands is ${points(pulse.robustness.values.A4.DEFAULT)}. These are associations, not evidence that debt causes dropout;
       the per-band z-test p-values are unadjusted for multiple comparisons.</p>
       `,
    )}</div>`;
  }
  return statOverview + card("Descriptive statistics", "After the default cleaning choices", describe)
    + `<div style="margin-top:16px">${card("Hypothesis tests, one per finding",
        "Simple tests from the course: a z-test to compare two percentages, a t-test to compare two means, a chi-square test for two categories. With thousands of rows almost every test rejects H0, so the size of the difference (Findings) matters more than p.", tests)}</div>`
    + bandAnalysis;
}

/* --- 3. Dimensionality reduction: PCA, t-SNE, variance explained, loadings */

const GROUP_COLOURS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
function groupStyles(names) {
  // Only three colours stay distinguishable for every pair; a fourth group is drawn as grey rings.
  return names.map((n, i) => (i < 3 ? { key: n, label: n, color: GROUP_COLOURS[i] } : { key: n, label: n + " (rings)", color: "var(--muted)", ring: true }));
}

// A short "how to read this" list written by the export from the computed results.
function notesCard(notes) {
  if (!notes || !notes.length) return "";
  return `<div class="card" style="margin-bottom:16px"><h3>Interpretation</h3><ul class="notes">${notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>`;
}

function reductionPanel(p, pulse, X) {
  const D = X.dr;
  if (!D) return `<p class="muted">Not available for this pulse.</p>`;
  const v = D.variance;
  const styles = groupStyles(D.categories);
  addChart(() => Charts.scatter($("#c-pca"), D.pca, styles, {
    xLabel: `PC1 (${percent(v[0], 0)})`,
    yLabel: `PC2 (${percent(v[1], 0)})`,
    offsetCategoryToRight: p === "academic" ? "Graduated"
      : p === "wellbeing" ? "Persistent sadness" : undefined,
  }));
  const showTsne = D.n_unique >= 200;   // with few distinct rows t-SNE only scatters copies of the same points
  if (showTsne) addChart(() => Charts.scatter($("#c-tsne"), D.tsne, styles, { axes: false, xLabel: "t-SNE 1", yLabel: "t-SNE 2" }));
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
  const overlap = D.n_unique < 200 ? ` Only ${D.n_unique} different answer combinations exist, so many students sit on exactly the same point.` : "";
  const pcaPointNote = D.pca.some((point) => point.n > 1)
    ? `Markers are grouped at identical PCA coordinates; larger markers represent more students. When outcome groups share a coordinate, their markers are slightly offset horizontally for visibility${p === "wellbeing" ? ", with Persistent sadness shifted to the right" : p === "academic" ? ", with Graduated shifted to the right" : ""} (the PCA values themselves are unchanged).`
    : "";
  const pcaDescription = p === "wellbeing" ? "" : `<p class="muted small">${D.n_rows.toLocaleString()} rows × ${D.n_features} standardised features. Colour shows the outcome,
      which the methods never see, so any separation is structure already in the data.${overlap} ${pcaPointNote}</p>
    `;
  return notesCard(X.dr_notes) + pcaDescription + `
    <div class="grid g2">
      ${card("PCA: first two components", `Together they explain ${percent(v[0] + v[1], 0)} of the variance`, chartBox("c-pca"))}
      ${showTsne ? card("t-SNE", `Non-linear map of a ${D.tsne.length}-row sample; only local neighbourhoods are meaningful`, chartBox("c-tsne")) : ""}
      ${card("Variance explained by each component", "", chartBox("c-scree"))}
      ${card("What PC1 and PC2 are made of", "Features with the largest loadings", loadings)}
    </div>`;
}

/* --- 4. Association rules: common beliefs the data contradicts (lift below 1) */

function rulesPanel(p, pulse, X) {
  const outcome = { dropout_1: "drops out", sadness: "reports persistent sadness",
                    hostels_high: "hostels in their busiest third of hours", campus_high: "campus in its busiest third of hours" };
  const item = { displaced_yes: "lives away from home", international_yes: "international student", heavy_social_media: "social media several times a day",
                 enough_sleep: "8+ hours of sleep", inactive: "0 active days a week", male: "male", week_weekend: "weekend", month_Jun: "June" };
  let cards = "";
  for (const r of pulse.surprising_rules || []) {
    const ifSide = r.antecedents.split(", ").map((t) => item[t] || niceItem(t)).join(" + ");
    const rule = `IF ${ifSide} → THEN ${outcome[r.consequents] || niceItem(r.consequents)}`;
    cards += card(`Common belief: ${esc(r.belief)}`, esc(rule),
      `<div class="metrics">
         <div class="tile"><div class="label">Lift</div><div class="value compact">${number(r.lift, 2)}×</div><div class="note">below 1 = less likely than average</div></div>
         <div class="tile"><div class="label">Confidence</div><div class="value compact">${percent(r.confidence, 0)}</div><div class="note">vs ${percent(r.base_rate, 0)} overall</div></div>
         <div class="tile"><div class="label">Support</div><div class="value compact">${percent(r.support, 1)}</div><div class="note">of all records</div></div></div>
       <p class="small"><b>What the data says:</b> the opposite. When the "if" side is true, the outcome happens in ${percent(r.confidence, 0)} of
       records, against ${percent(r.base_rate, 0)} overall.</p>`);
  }
  return `    <div class="grid g2">${cards || `<p class="muted">No rules exported.</p>`}</div>`;
}

/* --- 5. Clustering */

function clusteringPanel(p, pulse, X) {
  const M = X.cluster_map;
  const C = X.clusters;
  const kmeansMetric = (C && C.metrics || []).find((metric) => metric.method === "kmeans");
  const kmeansNotes = (X.cluster_notes || []).filter((note) => !/\b(DBSCAN|OPTICS)\b/i.test(note));
  let profiles = "";
  if (C && C.profiles && C.profiles.length) {
    const kmeansProfiles = C.profiles.filter((r) => typeof r.cluster === "number" || typeof r.group === "number");
    const profileRows = kmeansProfiles.length ? kmeansProfiles : C.profiles;
    const keys = Object.keys(profileRows[0] || {});
    profiles = table(profileRows, keys.map((k) => ({
      label: k.replace(/_/g, " "),
      numeric: !/cluster|group/.test(k),
      value: (r) => {
        const v = r[k];
        if (/cluster|group/.test(k)) return typeof v === "number" ? "Group " + (v + 1) : v;
        if (typeof v !== "number") return v ?? "–";
        if (/rate|share/.test(k)) return percent(v);
        return Number.isInteger(v) ? v.toLocaleString() : number(v, 2);
      },
    })));
  }
  if (M) addChart(() => Charts.scatter($("#c-clu"), M.points, groupStyles(M.groups), { xLabel: "PC1", yLabel: "PC2" }));
  return notesCard(kmeansNotes)
    + (M ? card("K-Means groups on the first two principal components",
      `${p === "wellbeing" && M.n_rows ? `All ${M.n_rows.toLocaleString()} students; identical positions are grouped into markers` : `Sample of ${M.points.length.toLocaleString()}`}${p === "wellbeing" && kmeansMetric ? ` · Silhouette score: ${number(kmeansMetric.silhouette, 3)} (estimated from ${kmeansMetric.silhouette_sample_size.toLocaleString()} students; −1 to 1, higher indicates clearer separation)` : ""}`,
      chartBox("c-clu")) : "")
    + (profiles ? `<div style="margin-top:16px">${card("What each group looks like", "", profiles)}</div>` : "");
}

/* --- 6. Findings: one graph + one simple test per finding, then the cleaning-choice selector */

function findingChart(id, ev) {
  const isRate = ev.kind === "rate" || (ev.kind === "gap" && !id.startsWith("EN"));
  const fmt = isRate ? (ev.kind === "gap" ? points : (v) => percent(v, 1)) : (v) => number(v, 1);
  const twoGroups = ev.bars.length === 2 && ev.kind !== "gap";
  const bars = ev.bars.map((b, i) => ({
    label: b.label, value: b.value,
    color: twoGroups ? (i === 0 ? "var(--series-2)" : "var(--series-1)") : "var(--series-1)",
    tip: `<b>${esc(b.label)}</b><br>${fmt(b.value)}<br>${b.n.toLocaleString()} ${id.startsWith("EN") ? "hours" : "students"}`,
  }));
  addChart(() => Charts.hbars($(`#c-f-${id}`), bars, { fmt, min: ev.kind === "gap" ? undefined : 0, labelWidth: 150 }));
}

function findingsPanel(p, pulse, X) {
  const R = pulse.robustness;
  let cards = "";
  for (const id of displayedFindingIds(p, pulse)) {
    const ev = pulse.evidence[id];
    findingChart(id, ev);
    cards += card(`${id} · ${esc(shortName(id, pulse.findings[id]))}: ${formatFinding(id, R.values[id].DEFAULT)}`,
      `${esc(pulse.findings[id])}. Graph: ${esc(ev.unit)}.`, chartBox(`c-f-${id}`) + (ev.tests ? ev.tests.map(testLine).join("") : testLine(ev.test)));
  }
  let selects = "";
  for (const [name, options] of Object.entries(pulse.decisions)) {
    let opts = "";
    options.forEach((o, i) => { opts += `<option value="${esc(o)}">${esc(o.replace(/_/g, " "))}${i === 0 ? " (default)" : ""}</option>`; });
    selects += `<label>${esc(name.replace(/_/g, " "))}<select data-decision="${esc(name)}">${opts}</select></label>`;
  }
  const explorer = card("Does each finding survive a different cleaning choice?",
    "Each finding is recomputed with one decision changed (the others stay at our default).",
    `<div class="controls" id="explorer">${selects}</div><div id="explorer-table"></div>`);
  const answer = p === "academic" ? "" : extraAnswer(p, X);
  return `<div class="grid g2">${cards}</div>` + answer + `<div style="margin-top:16px">${explorer}</div>`;
}

// The chart that answers the research question as a whole (beyond the individual findings).
function extraAnswer(p, X) {
  if (p === "academic") {
    addChart(() => Charts.hbars($("#c-answer"), X.timeline.map((r) => ({ label: r.stage.replace(/^\d /, ""), value: r.capture_at_25pct,
      tip: `<b>${r.stage}</b><br>reaches ${percent(r.capture_at_25pct)} of dropouts<br>ROC AUC ${number(r.roc_auc, 3)}` })),
      { fmt: (v) => percent(v, 0), min: 0, max: 1, refs: [{ value: X.timeline[0].max_possible_capture, label: "max possible" }], labelWidth: 140 }));
    return `<div style="margin-top:16px">${card("The answer: how early can the university warn?",
      "Share of eventual dropouts reached by contacting the 25% highest-risk students (cross-validated logistic regression). After semester 1 the list is almost as good as it can get.", chartBox("c-answer"))}</div>`;
  }
  if (p === "wellbeing") {
    addChart(() => Charts.hbars($("#c-answer"), X.sleep_gradient.map((r) => ({ label: (r.sleep_hours === 4 ? "≤4" : r.sleep_hours === 10 ? "10+" : r.sleep_hours) + " h", value: r.rate })),
      { fmt: (v) => percent(v, 0), min: 0, labelWidth: 50 }));
    return `<div style="margin-top:16px">${card("Persistent sadness by hours of sleep",
      "Weighted share of students: lowest at 8 hours, higher with less and with much more sleep", chartBox("c-answer"))}</div>`;
  }
  return "";
}

function connectExplorer(p, pulse) {
  const R = pulse.robustness;
  const selects = Array.from(document.querySelectorAll("#explorer select"));
  function show(variant) {
    const rows = displayedFindingIds(p, pulse);
    $("#explorer-table").innerHTML = table(rows, [
      { label: "Finding", html: true, value: (id) => `<b>${id} · ${esc(shortName(id, pulse.findings[id]))}</b>` },
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
    const ids = displayedFindingIds(p, pulse);
    const counts = displayedRobustnessCounts(pulse, ids);
    const columns = [
      { label: "Finding", html: true, value: (id) => `<b>${id}</b> ${esc(shortName(id, pulse.findings[id]))}` },
      { label: "Default", numeric: true, value: (id) => formatFinding(id, R.values[id].DEFAULT) },
    ];
    for (const v of variants) {
      columns.push({ label: niceVariant(v), html: true, value: (id) => `${chip(R.status[id][v])}<div class="small muted">${formatFinding(id, R.values[id][v])}</div>` });
    }
    html += `<details class="card" style="margin-bottom:12px" ${p === "academic" ? "open" : ""}>
      <summary><b>${pulse.meta.num} ${pulse.meta.title}</b> · ${counts.holds}/${counts.tested} hold</summary>
      ${table(ids, columns)}</details>`;
  }
  $("#app").innerHTML = html;
}

/* ------------------------------------------------------------------ conclusions page */

function conclusionsPage() {
  const A = DATA.pulses.academic, W = DATA.pulses.wellbeing, E = DATA.pulses.energy;
  const v = (pulse, id) => pulse.robustness.values[id].DEFAULT;
  const a3 = A.robustness.values.A3["zero_grade=as_score"] / v(A, "A3");
  const bandLow = (A.evidence.A4.tests || [])[0];
  const academicRules = table([
    { when: "Week 1", rule: "Fees not paid or in debt", flagged: "18.5%", caught: "39%", precision: "82%" },
    { when: "End of semester 1", rule: "Passed half or fewer of semester-1 courses", flagged: "22.3%", caught: "56%", precision: "98%" },
    { when: "End of semester 1", rule: "Passed half or fewer of semester-1 courses or fees not paid", flagged: "27.8%", caught: "68%", precision: "95%" },
    { when: "End of semester 2", rule: "Passed half or fewer of semester-2 courses", flagged: "26.8%", caught: "66%", precision: "97%" },
  ], [
    { label: "When", value: (r) => r.when },
    { label: "Rule", value: (r) => r.rule },
    { label: "Students flagged", numeric: true, value: (r) => r.flagged },
    { label: "Future dropouts caught", numeric: true, value: (r) => r.caught },
    { label: "Flagged students who dropped out", numeric: true, value: (r) => r.precision },
  ]);
  let holds = 0, tested = 0;
  for (const p of PULSES) {
    const pulse = DATA.pulses[p];
    const counts = displayedRobustnessCounts(pulse, displayedFindingIds(p, pulse));
    holds += counts.holds;
    tested += counts.tested;
  }
  $("#app").innerHTML = `<h1>Conclusions</h1>
    <h2>① Academic: who drops out, and how early can we tell?</h2>
    <p class="small muted">Simple rule-based contact lists, among 3,630 students with resolved outcomes (1,421 dropouts). These are observed rates, not regression predictions.</p>
    ${academicRules}
    <p class="small">At the end of semester 1, the combined course-progress and fee rule flags 27.8% of students and catches 68% of future dropouts; 95% of those flagged later drop out. Waiting until semester 2 does not improve the rule's yield (66% caught). Without a scholarship is a much broader flag: it catches 91% of dropouts but includes 73% of all students, so it is less practical as a contact list.</p>
    <ul class="notes">
      <li>Debtors drop out ${formatFinding("A1", v(A, "A1"))} more often, and students without a scholarship ${formatFinding("A2", v(A, "A2"))} more often.</li>
      <li>Debt still matters among students with similar semester-1 results, except those already failing${bandLow ? ` (p = ${pValue(bandLow.p)} there)` : ""}.</li>
      <li>A grade of 0 is a missing exam, not a score: treating it as a score would make the grade gap ${number(a3, 1)}× too big.</li>
    </ul>
    <h2>② Wellbeing: which behaviours co-occur with persistent sadness?</h2>
    <ul class="notes">
      <li>Bullying has the strongest link (${formatFinding("W2", v(W, "W2"))}), then sleeping under 8 hours (${formatFinding("W1", v(W, "W1"))}),
        then no physical activity (${formatFinding("W3", v(W, "W3"))}). All three tests reject H0.</li>
      <li>All three hold under every cleaning choice, including weighted and unweighted rates.</li>
    </ul>
    <h2>③ Energy: when does a campus building behave differently, and why?</h2>
    <ul class="notes">
      <li>The academic building uses ${formatFinding("EN1", v(E, "EN1"))} more on teaching days; the hostels use ${formatFinding("EN2", v(E, "EN2"))} more at night in term.</li>
    </ul>
    <h2>The umbrella question</h2>
    <ul class="notes">
      <li>${holds} of ${tested} finding × cleaning-choice checks hold: most findings are solid.</li>
      <li>A few reasonable-looking choices change the answer (treating missing exams as 0, filling meter gaps), so cleaning
        decisions should be reported next to the numbers.</li>
    </ul>
    <h2>Limitations</h2>
    <ul class="notes">
      <li>Three institutions in three countries, none of them ours: we compare conclusions, never join rows.</li>
      <li>All findings are associations, not causes.</li>
      <li>The energy calendar has no exam dates, so we compare teaching days and vacations instead.</li>
      <li>Decisions are changed one at a time; combinations are not tested.</li>
    </ul>`;
}

/* ------------------------------------------------------------------ navigation */

function topBar(current) {
  let links = `<a href="#/" class="${current === "" ? "active" : ""}">Overview</a>`;
  for (const p of PULSES) {
    const m = DATA.pulses[p].meta;
    links += `<a href="#/${p}/processing" class="${current === p ? "active" : ""}">${m.num} ${m.title}</a>`;
  }
  links += `<a href="#/robustness" class="${current === "robustness" ? "active" : ""}">Robustness</a>`;
  links += `<a href="#/conclusions" class="${current === "conclusions" ? "active" : ""}">Conclusions</a>`;
  $("#tabs").innerHTML = links;
}

function showPage() {
  const [page = "", panel = "processing"] = location.hash.replace(/^#\/?/, "").split("/");
  pendingCharts = [];
  topBar(page);
  if (PULSES.includes(page)) pulsePage(page, panel);
  else if (page === "robustness") robustnessPage();
  else if (page === "conclusions") conclusionsPage();
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
