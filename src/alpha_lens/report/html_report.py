"""Render an :class:`AutopsyReport` as a standalone HTML file.

Design philosophy:

* **One file, no external dependencies.** Embed Plotly.js (about 4MB
  uncompressed) so the report works offline and on air-gapped machines.
  Yes, that's chunky — but it's the difference between "useful artifact"
  and "broken when the CDN moves."
* **All data is embedded as JSON, deserialized at load.** This keeps
  the Python side simple (one ``Plotly.newPlot`` per chart container)
  and the HTML readable in source view.
* **Sections degrade gracefully.** If a feature (factor attribution,
  decay analysis, etc.) wasn't run because the user didn't supply the
  necessary input, the section is replaced with a friendly note.
"""

from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path
from string import Template

import plotly.io as pio
from plotly.offline import get_plotlyjs

from alpha_lens.core.types import AutopsyReport, ReadinessVerdict
from alpha_lens.report import sections
from alpha_lens.report.template import HTML_TEMPLATE
from alpha_lens.viz.styles import BAD, GOOD, INK_MUTED, NEUTRAL, WARN

logger = logging.getLogger(__name__)

__all__ = ["render_html_report"]


def render_html_report(report: AutopsyReport, path: str | Path) -> str:
    """Render the report and write it to disk.

    Args:
        report: The autopsy report.
        path: Output path. Parent directories will be created.

    Returns:
        The path written to (as a string).
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    chart_data, sections_html, tabs_html = sections.build_all(report)

    # Header metadata.
    meta_period = (
        f"{html.escape(str(report.statistics.start_date.date()))} "
        f"&rarr; {html.escape(str(report.statistics.end_date.date()))}"
    )
    meta_obs = (
        f"{report.statistics.n_observations} obs &middot; "
        f"{report.statistics.years:.2f} yrs"
    )
    meta_generated = html.escape(
        str(report.metadata.get("generated_at", "")).split("T")[0]
    )

    # Score hero pieces.
    readiness = report.readiness
    score_svg = _score_circle_svg(readiness.overall_score, readiness.verdict)
    verdict_text = _verdict_display(readiness.verdict)
    verdict_class = readiness.verdict.value
    recommendation_html = html.escape(readiness.recommendation)
    risks_block = ""
    if readiness.top_risks:
        items = "".join(
            f"<li>{html.escape(r)}</li>" for r in readiness.top_risks[:3]
        )
        risks_block = (
            f'<ul class="risks" style="margin-top:14px;">{items}</ul>'
        )

    components_html = _components_html(readiness.components)
    stats_html = _stats_strip_html(report)

    # Serialize chart data to JSON.
    chart_data_json = json.dumps(chart_data, allow_nan=False)

    # Inline Plotly.js. ``include_plotlyjs="inline"`` would also work, but
    # ``get_plotlyjs()`` gives us direct access to the source so we can
    # avoid the wrapping ``pio.to_html`` machinery.
    plotly_js = get_plotlyjs()

    rendered = Template(HTML_TEMPLATE).safe_substitute(
        title=f"alpha-lens autopsy &mdash; {readiness.verdict.value.upper()}",
        meta_period=meta_period,
        meta_obs=meta_obs,
        meta_generated=meta_generated,
        score_svg=score_svg,
        verdict_text=verdict_text,
        verdict_class=verdict_class,
        recommendation=recommendation_html,
        risks_block=risks_block,
        components_html=components_html,
        stats_html=stats_html,
        tabs_html=tabs_html,
        panels_html=sections_html,
        chart_data_json=chart_data_json,
        plotly_js=plotly_js,
    )

    path_obj.write_text(rendered, encoding="utf-8")
    logger.info(
        "Wrote autopsy report to %s (%.1f MB)",
        path_obj,
        path_obj.stat().st_size / 1024 / 1024,
    )
    return str(path_obj)


# ----------------------------------------------------------------------------
# Score circle
# ----------------------------------------------------------------------------


def _score_circle_svg(score: float, verdict: ReadinessVerdict) -> str:
    """Build the donut-ring score visualization as inline SVG."""
    size = 170
    stroke = 12
    radius = (size - stroke) / 2
    circumference = 2 * math.pi * radius
    pct = max(0.0, min(100.0, score)) / 100.0
    dash = circumference * pct
    gap = circumference - dash

    color = _verdict_color(verdict)

    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{size/2}" cy="{size/2}" r="{radius}"
          fill="none" stroke="#21262d" stroke-width="{stroke}" />
  <circle cx="{size/2}" cy="{size/2}" r="{radius}"
          fill="none" stroke="{color}" stroke-width="{stroke}"
          stroke-linecap="round"
          stroke-dasharray="{dash:.2f} {gap:.2f}" />
</svg>
<div class="score-number">
  <div class="score-value" style="color: {color};">{score:.0f}</div>
  <div class="score-max">/ 100</div>
</div>
""".strip()


def _verdict_color(verdict: ReadinessVerdict) -> str:
    return {
        ReadinessVerdict.READY: GOOD,
        ReadinessVerdict.CONDITIONAL: WARN,
        ReadinessVerdict.NOT_READY: BAD,
        ReadinessVerdict.REJECT: BAD,
    }[verdict]


def _verdict_display(verdict: ReadinessVerdict) -> str:
    return {
        ReadinessVerdict.READY: "READY",
        ReadinessVerdict.CONDITIONAL: "CONDITIONAL",
        ReadinessVerdict.NOT_READY: "NOT READY",
        ReadinessVerdict.REJECT: "REJECT",
    }[verdict]


# ----------------------------------------------------------------------------
# Component cards
# ----------------------------------------------------------------------------


def _components_html(components: list) -> str:
    """Build the grid of component score cards."""
    cards: list[str] = []
    for c in components:
        status = c.status
        bar_pct = max(0.0, min(100.0, c.score))
        cards.append(f"""
<div class="comp">
  <div class="comp-head">
    <div class="comp-name">{html.escape(c.name)}</div>
    <div class="comp-score {status}">{c.score:.0f}</div>
  </div>
  <div class="comp-bar"><div class="{status}" style="width: {bar_pct:.0f}%;"></div></div>
  <div class="comp-value">{html.escape(c.value)}</div>
</div>
""".strip())
    return "\n".join(cards)


# ----------------------------------------------------------------------------
# Headline stats strip
# ----------------------------------------------------------------------------


def _stats_strip_html(report: AutopsyReport) -> str:
    """Build the strip of headline statistics."""
    stats = report.statistics

    def fmt_pct(v: float) -> str:
        return f"{v * 100:+.1f}%"

    def cls_signed(v: float) -> str:
        if v > 0:
            return "pos"
        if v < 0:
            return "neg"
        return ""

    items = [
        ("CAGR", fmt_pct(stats.cagr), cls_signed(stats.cagr)),
        ("Sharpe", f"{stats.sharpe_ratio:.2f}", cls_signed(stats.sharpe_ratio - 1.0)),
        ("Sortino", f"{stats.sortino_ratio:.2f}", ""),
        ("Calmar", f"{stats.calmar_ratio:.2f}", ""),
        ("Vol (ann)", f"{stats.annualized_volatility * 100:.1f}%", ""),
        ("Max DD", fmt_pct(stats.max_drawdown), cls_signed(stats.max_drawdown)),
        ("Win Rate", f"{stats.win_rate * 100:.0f}%", ""),
    ]
    return "\n".join(
        f"""
<div class="stat">
  <div class="stat-label">{html.escape(label)}</div>
  <div class="stat-value {cls}">{html.escape(value)}</div>
</div>
""".strip()
        for label, value, cls in items
    )


# Suppress unused-import warning for visual tokens used elsewhere.
_ = (NEUTRAL, INK_MUTED, pio)
