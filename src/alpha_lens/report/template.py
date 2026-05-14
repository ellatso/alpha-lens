"""HTML/CSS/JS shell for the autopsy report.

The template uses Python ``str.format`` semantics with named placeholders.
We deliberately avoid Jinja2 to keep the dependency surface small —
the report is a one-shot render with no logic inside the template.
"""

from __future__ import annotations

# Template uses dollar-sign placeholders to avoid escaping every CSS brace.
# Substitution is done with string.Template.safe_substitute.
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>$title</title>
<style>
:root {
  --bg-dark: #0e1116;
  --bg-panel: #161b22;
  --bg-panel-2: #1c2128;
  --ink: #e6edf3;
  --ink-muted: #8b949e;
  --ink-faint: #6e7681;
  --border: #30363d;
  --border-subtle: #21262d;
  --accent-primary: #f6c177;
  --accent-secondary: #9ccfd8;
  --accent-tertiary: #c4a7e7;
  --good: #a3be8c;
  --warn: #f6c177;
  --bad: #eb6f92;
  --neutral: #7aa2f7;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html {
  scroll-behavior: smooth;
}

body {
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Helvetica Neue",
               Helvetica, Arial, sans-serif;
  background: var(--bg-dark);
  color: var(--ink);
  line-height: 1.55;
  font-size: 14px;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--accent-secondary); }
code, pre, .mono {
  font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.93em;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 32px 24px 60px;
}

/* ---------- Header ---------- */
header.app-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 28px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--border-subtle);
}
.brand {
  font-family: "IBM Plex Serif", Georgia, serif;
  font-size: 22px;
  letter-spacing: -0.01em;
  color: var(--ink);
}
.brand strong { color: var(--accent-primary); font-weight: 600; }
.meta {
  text-align: right;
  color: var(--ink-muted);
  font-size: 12px;
  font-family: "JetBrains Mono", monospace;
  line-height: 1.7;
}

/* ---------- Score hero ---------- */
.score-hero {
  background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 36px;
  margin-bottom: 28px;
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 36px;
  align-items: center;
}
.score-circle {
  position: relative;
  width: 170px;
  height: 170px;
  margin: 0 auto;
}
.score-circle svg { transform: rotate(-90deg); }
.score-circle .score-number {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}
.score-circle .score-value {
  font-family: "IBM Plex Serif", Georgia, serif;
  font-size: 48px;
  font-weight: 600;
  line-height: 1;
}
.score-circle .score-max {
  font-size: 13px;
  color: var(--ink-muted);
  font-family: "JetBrains Mono", monospace;
  margin-top: 4px;
}
.verdict-block { min-width: 0; }
.verdict-label {
  font-size: 11px;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 6px;
}
.verdict-tag {
  display: inline-block;
  padding: 5px 14px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 13px;
  letter-spacing: 0.05em;
  margin-bottom: 14px;
}
.verdict-tag.ready { background: rgba(163,190,140,0.15); color: var(--good); border: 1px solid rgba(163,190,140,0.35); }
.verdict-tag.conditional { background: rgba(246,193,119,0.15); color: var(--warn); border: 1px solid rgba(246,193,119,0.35); }
.verdict-tag.not_ready { background: rgba(235,111,146,0.15); color: var(--bad); border: 1px solid rgba(235,111,146,0.35); }
.verdict-tag.reject { background: rgba(235,111,146,0.22); color: var(--bad); border: 1px solid rgba(235,111,146,0.55); }
.recommendation {
  font-size: 14.5px;
  line-height: 1.6;
  color: var(--ink);
  max-width: 760px;
}
.recommendation .concerns {
  display: block;
  margin-top: 10px;
  color: var(--ink-muted);
  font-size: 13px;
}

/* ---------- Component grid ---------- */
.components {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
}
.comp {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
.comp-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}
.comp-name {
  font-size: 12px;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.comp-score {
  font-family: "JetBrains Mono", monospace;
  font-size: 22px;
  font-weight: 600;
  line-height: 1;
}
.comp-score.pass { color: var(--good); }
.comp-score.warn { color: var(--warn); }
.comp-score.fail { color: var(--bad); }
.comp-bar {
  height: 4px;
  background: var(--border-subtle);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 8px;
}
.comp-bar > div {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s;
}
.comp-bar > div.pass { background: var(--good); }
.comp-bar > div.warn { background: var(--warn); }
.comp-bar > div.fail { background: var(--bad); }
.comp-value {
  font-size: 12.5px;
  color: var(--ink);
  font-family: "JetBrains Mono", monospace;
  line-height: 1.45;
}

/* ---------- Headline stats strip ---------- */
.stats-strip {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 24px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 24px;
  margin-bottom: 28px;
}
.stat {
  border-left: 1px solid var(--border-subtle);
  padding-left: 16px;
}
.stat:first-child { border-left: 0; padding-left: 0; }
.stat-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
  margin-bottom: 4px;
}
.stat-value {
  font-family: "JetBrains Mono", monospace;
  font-size: 19px;
  color: var(--ink);
  font-weight: 500;
}
.stat-value.pos { color: var(--good); }
.stat-value.neg { color: var(--bad); }

/* ---------- Tabs ---------- */
.tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
  overflow-x: auto;
}
.tab {
  padding: 11px 18px;
  background: none;
  border: 0;
  color: var(--ink-muted);
  font-size: 13px;
  font-family: inherit;
  font-weight: 500;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
  white-space: nowrap;
}
.tab:hover { color: var(--ink); }
.tab.active {
  color: var(--accent-primary);
  border-bottom-color: var(--accent-primary);
}

.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ---------- Section / panel ---------- */
.panel {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 22px 24px;
  margin-bottom: 18px;
}
.panel h2 {
  font-family: "IBM Plex Serif", Georgia, serif;
  font-size: 19px;
  font-weight: 600;
  margin-bottom: 4px;
}
.panel .subtitle {
  font-size: 13px;
  color: var(--ink-muted);
  margin-bottom: 18px;
}
.panel + .panel { margin-top: 12px; }

.panel-grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 900px) {
  .panel-grid-2 { grid-template-columns: 1fr; }
  .score-hero { grid-template-columns: 1fr; text-align: center; }
}

/* ---------- Tables ---------- */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table th, table td {
  padding: 9px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-subtle);
}
table th {
  color: var(--ink-muted);
  font-weight: 500;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
table td.num {
  font-family: "JetBrains Mono", monospace;
  text-align: right;
}
table td.pos { color: var(--good); }
table td.neg { color: var(--bad); }
table tr:last-child td { border-bottom: 0; }

/* ---------- Pills ---------- */
.pill {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 10px;
  font-size: 11px;
  font-family: "JetBrains Mono", monospace;
  letter-spacing: 0.04em;
}
.pill.bull { background: rgba(163,190,140,0.18); color: var(--good); }
.pill.bear { background: rgba(235,111,146,0.18); color: var(--bad); }
.pill.high_vol { background: rgba(246,193,119,0.18); color: var(--warn); }
.pill.sideways { background: rgba(126,138,153,0.22); color: var(--ink-muted); }

/* ---------- Risk list ---------- */
.risks {
  list-style: none;
  margin-top: 6px;
}
.risks li {
  padding: 7px 0 7px 22px;
  border-bottom: 1px solid var(--border-subtle);
  position: relative;
  font-size: 13px;
}
.risks li:last-child { border-bottom: 0; }
.risks li::before {
  content: "▸";
  position: absolute;
  left: 0;
  color: var(--warn);
  font-size: 11px;
  top: 9px;
}

/* ---------- Empty state ---------- */
.empty-note {
  color: var(--ink-muted);
  font-style: italic;
  padding: 24px;
  text-align: center;
}

/* ---------- Footer ---------- */
footer {
  margin-top: 40px;
  padding-top: 18px;
  border-top: 1px solid var(--border-subtle);
  color: var(--ink-faint);
  font-size: 12px;
  text-align: center;
}
footer a { color: var(--ink-muted); }

/* Plotly overrides */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
.plot-container { margin: 6px 0; }
</style>
</head>
<body>
<div class="container">

<header class="app-header">
  <div class="brand">alpha&#8209;<strong>lens</strong> &nbsp;<span style="color:var(--ink-muted);font-size:13px;font-family:'JetBrains Mono',monospace;">autopsy report</span></div>
  <div class="meta">
    $meta_period<br/>
    $meta_obs<br/>
    Generated $meta_generated
  </div>
</header>

<!-- Score hero -->
<section class="score-hero">
  <div class="score-circle">$score_svg</div>
  <div class="verdict-block">
    <div class="verdict-label">Production Readiness</div>
    <span class="verdict-tag $verdict_class">$verdict_text</span>
    <div class="recommendation">
      $recommendation
      $risks_block
    </div>
  </div>
</section>

<!-- Component grid -->
<section class="components">
  $components_html
</section>

<!-- Headline stats -->
<section class="stats-strip">
  $stats_html
</section>

<!-- Tabbed analyses -->
<nav class="tabs" id="tabs">
  $tabs_html
</nav>
<div id="tab-panels">
  $panels_html
</div>

<footer>
  Generated by <a href="https://github.com/alpha-lens/alpha-lens">alpha&#8209;lens</a> &nbsp;·&nbsp;
  Built for quant researchers who care about what their backtest actually means.
</footer>

</div>

<script>$plotly_js</script>
<script>
const CHART_DATA = $chart_data_json;

function renderCharts() {
  for (const id in CHART_DATA) {
    const target = document.getElementById(id);
    if (!target) continue;
    const spec = CHART_DATA[id];
    Plotly.newPlot(
      target,
      spec.data,
      spec.layout,
      { displaylogo: false, responsive: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] }
    );
  }
}

function setupTabs() {
  const buttons = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-panel');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.target;
      buttons.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = document.getElementById(target);
      if (panel) panel.classList.add('active');
      // Force-resize charts in the newly active panel.
      if (panel) {
        panel.querySelectorAll('.plot-container').forEach(c => {
          if (window.Plotly) Plotly.Plots.resize(c);
        });
      }
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  renderCharts();
  setupTabs();
});
</script>
</body>
</html>
"""
