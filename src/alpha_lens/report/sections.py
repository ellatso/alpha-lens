"""Per-tab content builders for the autopsy report.

Each section function returns:
    * A dict of {chart_id: chart_spec_dict} to be JSON-serialized.
    * An HTML string for the tab panel body.
    * A short label for the tab button.

The main entry point ``build_all`` orchestrates these.
"""

from __future__ import annotations

import html
from typing import Any

from alpha_lens.core.types import AutopsyReport
from alpha_lens.viz import charts


def build_all(report: AutopsyReport) -> tuple[dict, str, str]:
    """Build chart data, panels HTML, and tabs HTML for the entire report.

    Returns:
        Tuple of (chart_data, panels_html, tabs_html).
    """
    chart_data: dict[str, Any] = {}
    sections: list[tuple[str, str, str]] = []  # (tab_id, label, panel_html)

    # 1. Overview — always present.
    cd, panel_html = _overview_section(report)
    chart_data.update(cd)
    sections.append(("overview", "Overview", panel_html))

    # 2. Regime — if regime analysis ran.
    if report.regime is not None:
        cd, panel_html = _regime_section(report)
        chart_data.update(cd)
        sections.append(("regime", "Regime", panel_html))

    # 3. Drawdowns — always present.
    cd, panel_html = _drawdown_section(report)
    chart_data.update(cd)
    sections.append(("drawdowns", "Drawdowns", panel_html))

    # 4. Attribution — if benchmark or factors supplied.
    if report.attribution is not None:
        cd, panel_html = _attribution_section(report)
        chart_data.update(cd)
        sections.append(("attribution", "Attribution", panel_html))

    # 5. Decay — if factor value series supplied.
    if report.decay is not None:
        cd, panel_html = _decay_section(report)
        chart_data.update(cd)
        sections.append(("decay", "Alpha Decay", panel_html))

    # 6. Production readiness — always present (this is the differentiator).
    cd, panel_html = _readiness_section(report)
    chart_data.update(cd)
    sections.append(("readiness", "Production Readiness", panel_html))

    # Build tab buttons HTML.
    tabs_html_parts: list[str] = []
    panels_html_parts: list[str] = []
    for i, (tab_id, label, panel_html) in enumerate(sections):
        active = "active" if i == 0 else ""
        tabs_html_parts.append(
            f'<button class="tab {active}" data-target="panel-{tab_id}">{html.escape(label)}</button>'
        )
        panels_html_parts.append(
            f'<div class="tab-panel {active}" id="panel-{tab_id}">{panel_html}</div>'
        )

    return chart_data, "\n".join(panels_html_parts), "\n".join(tabs_html_parts)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _chart_html(chart_id: str, height: int = 380) -> str:
    return (
        f'<div id="{chart_id}" class="plot-container" '
        f'style="width:100%; height:{height}px;"></div>'
    )


def _fig_pair(fig) -> dict:
    """Plotly figure → {data, layout} dict via to_dict, JSON-cleaned.

    The JS side calls ``Plotly.newPlot(target, spec.data, spec.layout, ...)``
    so we mirror that shape exactly.
    """
    raw = fig.to_dict()
    return {
        "data": _json_safe(raw.get("data", [])),
        "layout": _json_safe(raw.get("layout", {})),
    }


def _json_safe(obj: Any) -> Any:
    """Convert pandas / numpy / datetime values into JSON-serializable forms."""
    import datetime as _dt

    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, float):
        # Python float — guard against NaN/inf that would break JSON.
        import math as _math

        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    return obj


# ----------------------------------------------------------------------------
# Section: Overview
# ----------------------------------------------------------------------------


def _overview_section(report: AutopsyReport) -> tuple[dict, str]:
    """Equity curve, rolling Sharpe, monthly heatmap."""
    # NOTE: We need access to the actual returns series to draw the
    # equity curve. The report doesn't carry the raw returns by design
    # (memory/serialization), so we must reconstruct from drawdown analysis
    # — but in the current architecture we don't keep raw returns either.
    #
    # The cleanest fix: have the orchestrator stash the cleaned returns
    # in the report metadata. We do that below.
    returns = report.metadata.get("_returns")
    benchmark = report.metadata.get("_benchmark")
    chart_data: dict[str, Any] = {}
    parts: list[str] = []

    parts.append('<div class="panel">')
    parts.append('<h2>Equity & Rolling Performance</h2>')
    parts.append('<div class="subtitle">Headline performance, with rolling Sharpe to show stability over time.</div>')

    if returns is not None:
        fig1 = charts.chart_cumulative_returns(
            returns, benchmark_returns=benchmark, title="Cumulative Returns"
        )
        chart_data["chart-equity"] = _fig_pair(fig1)
        parts.append(_chart_html("chart-equity", height=320))

        fig2 = charts.chart_rolling_sharpe(
            returns, window=63, regime=report.regime, title="Rolling 63-day Sharpe"
        )
        chart_data["chart-rolling-sharpe"] = _fig_pair(fig2)
        parts.append(_chart_html("chart-rolling-sharpe", height=280))

        # Monthly heatmap if we have at least 6 months of data.
        if len(returns) > 120:
            fig3 = charts.chart_monthly_returns_heatmap(returns, title="Monthly Returns (%)")
            chart_data["chart-monthly"] = _fig_pair(fig3)
            parts.append(_chart_html("chart-monthly", height=300))
    else:
        parts.append('<div class="empty-note">No returns available for plotting.</div>')

    parts.append("</div>")
    return chart_data, "\n".join(parts)


# ----------------------------------------------------------------------------
# Section: Regime
# ----------------------------------------------------------------------------


def _regime_section(report: AutopsyReport) -> tuple[dict, str]:
    regime = report.regime
    assert regime is not None
    returns = report.metadata.get("_returns")
    chart_data: dict[str, Any] = {}
    parts: list[str] = []

    parts.append('<div class="panel">')
    parts.append('<h2>Regime Decomposition</h2>')
    parts.append(
        '<div class="subtitle">'
        f'Method: <span class="mono">{html.escape(regime.method.value)}</span> '
        f'&middot; {len(regime.summaries)} regimes detected'
        '</div>'
    )

    if returns is not None:
        fig1 = charts.chart_regime_overlay(returns, regime, title="Equity by Regime")
        chart_data["chart-regime-overlay"] = _fig_pair(fig1)
        parts.append(_chart_html("chart-regime-overlay", height=340))

    fig2 = charts.chart_regime_performance(regime, title="Performance by Regime")
    chart_data["chart-regime-perf"] = _fig_pair(fig2)
    parts.append(_chart_html("chart-regime-perf", height=280))

    # Regime summary table.
    rows: list[str] = []
    for s in regime.summaries:
        rows.append(f"""
<tr>
  <td><span class="pill {s.regime.value}">{s.regime.value}</span></td>
  <td class="num">{s.n_days}</td>
  <td class="num">{s.fraction:.0%}</td>
  <td class="num {'pos' if s.mean_return_annualized > 0 else 'neg'}">{s.mean_return_annualized:+.1%}</td>
  <td class="num">{s.volatility_annualized:.1%}</td>
  <td class="num {'pos' if s.sharpe_ratio > 0 else 'neg'}">{s.sharpe_ratio:+.2f}</td>
  <td class="num {'neg' if s.max_drawdown < -0.10 else ''}">{s.max_drawdown:.1%}</td>
</tr>""")

    parts.append(f"""
<table style="margin-top:18px;">
<thead><tr>
  <th>Regime</th><th style="text-align:right">Days</th><th style="text-align:right">Time</th>
  <th style="text-align:right">CAGR</th><th style="text-align:right">Vol</th>
  <th style="text-align:right">Sharpe</th><th style="text-align:right">Max DD</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
""")

    parts.append('<div class="subtitle" style="margin-top:14px;">')
    parts.append(
        f'Cross-regime robustness score: <span class="mono" '
        f'style="color:var(--accent-primary)">{regime.robustness_score:.2f}</span> '
        '<span style="color:var(--ink-muted);">'
        '(weighted positive-Sharpe fraction, penalized for high cross-regime variance)'
        '</span>'
    )
    parts.append('</div>')

    parts.append("</div>")
    return chart_data, "\n".join(parts)


# ----------------------------------------------------------------------------
# Section: Drawdowns
# ----------------------------------------------------------------------------


def _drawdown_section(report: AutopsyReport) -> tuple[dict, str]:
    dd = report.drawdowns
    returns = report.metadata.get("_returns")
    chart_data: dict[str, Any] = {}
    parts: list[str] = []

    worst_depth = dd.events[0].depth if dd.events else 0.0
    parts.append('<div class="panel">')
    parts.append('<h2>Drawdown Analysis</h2>')
    parts.append(
        '<div class="subtitle">'
        f'{dd.n_drawdowns} drawdowns &geq; 5% recorded &middot; '
        f'worst depth <span class="mono" style="color:var(--bad)">{worst_depth:.1%}</span>'
        '</div>'
    )

    if returns is not None:
        fig_uw = charts.chart_drawdown_underwater(returns, title="Underwater Plot")
        chart_data["chart-underwater"] = _fig_pair(fig_uw)
        parts.append(_chart_html("chart-underwater", height=240))

    if dd.events:
        fig_w = charts.chart_drawdown_waterfall(dd, title="Worst Drawdown Events")
        chart_data["chart-dd-waterfall"] = _fig_pair(fig_w)
        parts.append(_chart_html("chart-dd-waterfall", height=300))

        rows: list[str] = []
        for e in dd.events[:10]:
            regime_pill = ""
            if e.dominant_regime is not None:
                regime_pill = f'<span class="pill {e.dominant_regime.value}">{e.dominant_regime.value}</span>'
            recovery = (
                f'{e.recovery_days}d'
                if e.recovery_days is not None
                else '<span style="color:var(--bad)">unrecovered</span>'
            )
            rows.append(f"""
<tr>
  <td class="mono">{e.peak_date.date()}</td>
  <td class="mono">{e.trough_date.date()}</td>
  <td class="num neg">{e.depth:.1%}</td>
  <td class="num">{e.duration_days}</td>
  <td class="num">{recovery}</td>
  <td>{regime_pill}</td>
</tr>""")

        parts.append(f"""
<table style="margin-top:18px;">
<thead><tr>
  <th>Peak</th><th>Trough</th>
  <th style="text-align:right">Depth</th>
  <th style="text-align:right">Days to Trough</th>
  <th style="text-align:right">Recovery</th>
  <th>Regime</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
""")
    else:
        parts.append('<div class="empty-note">No significant drawdowns detected.</div>')

    parts.append("</div>")
    return chart_data, "\n".join(parts)


# ----------------------------------------------------------------------------
# Section: Attribution
# ----------------------------------------------------------------------------


def _attribution_section(report: AutopsyReport) -> tuple[dict, str]:
    attr = report.attribution
    assert attr is not None
    chart_data: dict[str, Any] = {}
    parts: list[str] = []

    parts.append('<div class="panel">')
    parts.append('<h2>Factor Attribution</h2>')
    parts.append(
        '<div class="subtitle">'
        f'R&sup2; = <span class="mono">{attr.r_squared:.3f}</span> &middot; '
        f'Annual &alpha; = <span class="mono {"pos" if attr.alpha_annualized > 0 else "neg"}">'
        f'{attr.alpha_annualized:+.1%}</span> &middot; '
        f'Uniqueness = <span class="mono">{attr.uniqueness_score:.2f}</span>'
        '</div>'
    )

    fig = charts.chart_attribution_bars(attr, title="Factor Loadings (β)")
    chart_data["chart-attribution"] = _fig_pair(fig)
    parts.append(_chart_html("chart-attribution", height=max(220, 60 + len(attr.factor_names) * 32)))

    # Coefficient table.
    rows: list[str] = []
    for name in attr.factor_names:
        beta = attr.betas[name]
        t = attr.t_stats[name]
        sig = "pos" if abs(t) > 2 and beta > 0 else ("neg" if abs(t) > 2 and beta < 0 else "")
        sig_marker = "★★" if abs(t) > 2.58 else ("★" if abs(t) > 1.96 else "")
        rows.append(f"""
<tr>
  <td><strong>{html.escape(name)}</strong></td>
  <td class="num {sig}">{beta:+.4f}</td>
  <td class="num">{t:+.2f}</td>
  <td>{sig_marker}</td>
</tr>""")
    parts.append(f"""
<table style="margin-top:18px;">
<thead><tr>
  <th>Factor</th><th style="text-align:right">β</th>
  <th style="text-align:right">t-stat</th><th>significance</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<div class="subtitle" style="margin-top:8px;">
★ = |t| &gt; 1.96 (p &lt; 0.05) &nbsp;&nbsp; ★★ = |t| &gt; 2.58 (p &lt; 0.01)
</div>
""")
    parts.append("</div>")

    # Correlation panel if present.
    if report.correlation is not None and not report.correlation.correlation_matrix.empty:
        parts.append('<div class="panel">')
        parts.append('<h2>Factor Correlations</h2>')
        parts.append(
            '<div class="subtitle">Pairwise correlations and variance-inflation factors.</div>'
        )
        fig = charts.chart_correlation_heatmap(report.correlation, title="")
        chart_data["chart-corr"] = _fig_pair(fig)
        parts.append(_chart_html("chart-corr", height=320))

        # VIF table.
        vif_rows: list[str] = []
        for name, vif in report.correlation.vif_scores.items():
            cls = "neg" if vif > 10 else ("warn-color" if vif > 5 else "")
            vif_rows.append(f"""
<tr><td>{html.escape(name)}</td><td class="num {cls}">{vif:.2f}</td></tr>""")
        if vif_rows:
            parts.append(f"""
<table style="margin-top:14px;">
<thead><tr><th>Factor</th><th style="text-align:right">VIF</th></tr></thead>
<tbody>{"".join(vif_rows)}</tbody>
</table>
<div class="subtitle" style="margin-top:8px;">
VIF &gt; 5 indicates noteworthy multicollinearity; &gt; 10 suggests redundant factors.
</div>
""")

        parts.append("</div>")

    return chart_data, "\n".join(parts)


# ----------------------------------------------------------------------------
# Section: Decay
# ----------------------------------------------------------------------------


def _decay_section(report: AutopsyReport) -> tuple[dict, str]:
    decay = report.decay
    assert decay is not None
    chart_data: dict[str, Any] = {}
    parts: list[str] = []

    parts.append('<div class="panel">')
    parts.append('<h2>Information Coefficient & Decay</h2>')
    hl_str = (
        f'{decay.estimated_half_life_days:.1f} days'
        if decay.estimated_half_life_days is not None
        else 'not detected'
    )
    parts.append(
        '<div class="subtitle">'
        f'IC stability = <span class="mono">{decay.ic_stability:.0%}</span> &middot; '
        f'estimated half-life = <span class="mono">{hl_str}</span>'
        '</div>'
    )

    fig = charts.chart_ic_decay(decay, title="Rank IC by Forward Horizon")
    chart_data["chart-ic-decay"] = _fig_pair(fig)
    parts.append(_chart_html("chart-ic-decay", height=300))

    # IC table.
    rows: list[str] = []
    for h in sorted(decay.ic_by_horizon.keys()):
        ic = decay.ic_by_horizon[h]
        p = decay.ic_pvalue_by_horizon[h]
        sig = "pos" if p < 0.05 and ic > 0 else ("neg" if p < 0.05 and ic < 0 else "")
        rows.append(f"""
<tr>
  <td class="mono">{h}d</td>
  <td class="num {sig}">{ic:+.4f}</td>
  <td class="num">{p:.3f}</td>
</tr>""")
    parts.append(f"""
<table style="margin-top:18px;">
<thead><tr>
  <th>Horizon</th><th style="text-align:right">Rank IC</th>
  <th style="text-align:right">p-value</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
""")
    parts.append("</div>")
    return chart_data, "\n".join(parts)


# ----------------------------------------------------------------------------
# Section: Production Readiness
# ----------------------------------------------------------------------------


def _readiness_section(report: AutopsyReport) -> tuple[dict, str]:
    chart_data: dict[str, Any] = {}
    parts: list[str] = []

    overfit = report.overfitting
    val = report.validation
    rob = report.robustness
    costs = report.costs

    # Overfitting panel.
    parts.append('<div class="panel">')
    parts.append('<h2>Overfitting Diagnostics</h2>')
    parts.append(
        '<div class="subtitle">'
        'Deflated Sharpe (Bailey &amp; L&oacute;pez de Prado 2014) and '
        'Probability of Backtest Overfitting via CSCV.'
        '</div>'
    )
    pbo_row = ""
    if overfit.probability_of_backtest_overfitting is not None:
        pbo_val = overfit.probability_of_backtest_overfitting
        pbo_cls = "pos" if pbo_val < 0.4 else ("neg" if pbo_val > 0.6 else "")
        pbo_row = (
            f'<tr><td>PBO (combinatorially symmetric)</td>'
            f'<td class="num {pbo_cls}">{pbo_val:.3f}</td>'
            f'<td>Probability the in-sample-best strategy is below the OOS median.</td></tr>'
        )

    p_cls = "pos" if overfit.deflated_sharpe_pvalue < 0.05 else (
        "neg" if overfit.deflated_sharpe_pvalue > 0.50 else ""
    )
    minlen_cls = "pos" if overfit.minimum_length_satisfied else "neg"

    parts.append(f"""
<table style="margin-top:8px;">
<thead><tr>
  <th>Metric</th><th style="text-align:right">Value</th><th>Interpretation</th>
</tr></thead>
<tbody>
<tr><td>Deflated Sharpe (z-score)</td>
    <td class="num">{overfit.deflated_sharpe_ratio:.2f}</td>
    <td>Sharpe corrected for multiple testing and non-normality.</td></tr>
<tr><td>DSR p-value</td>
    <td class="num {p_cls}">{overfit.deflated_sharpe_pvalue:.3f}</td>
    <td>Probability the true Sharpe is &leq; 0 after correction.</td></tr>
{pbo_row}
<tr><td>Minimum backtest length</td>
    <td class="num">{overfit.minimum_backtest_length_years:.1f} yrs</td>
    <td>Data needed for the observed Sharpe to be statistically distinguishable from chance.</td></tr>
<tr><td>Actual length</td>
    <td class="num {minlen_cls}">{overfit.actual_backtest_length_years:.2f} yrs</td>
    <td>{"Sufficient." if overfit.minimum_length_satisfied else "Insufficient relative to the observed Sharpe."}</td></tr>
</tbody></table>
""")
    parts.append("</div>")

    # Validation + walk-forward panel.
    parts.append('<div class="panel">')
    parts.append('<h2>Out-of-Sample Validation</h2>')
    parts.append('<div class="subtitle">70/30 chronological split and 5-window walk-forward.</div>')

    deg_cls = "pos" if val.degradation_ratio > 0.7 else ("neg" if val.degradation_ratio < 0.3 else "")
    parts.append(f"""
<table>
<thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>
<tbody>
<tr><td>In-sample Sharpe (first 70%)</td><td class="num">{val.in_sample_sharpe:+.2f}</td></tr>
<tr><td>Out-of-sample Sharpe (last 30%)</td><td class="num">{val.out_of_sample_sharpe:+.2f}</td></tr>
<tr><td>Degradation ratio (OOS / IS)</td><td class="num {deg_cls}">{val.degradation_ratio:.2f}</td></tr>
<tr><td>Walk-forward consistency</td><td class="num">{val.walk_forward_consistency:.0%}</td></tr>
</tbody></table>
""")

    if val.walk_forward_sharpes:
        fig = charts.chart_walkforward_bars(
            val.walk_forward_sharpes, title="Walk-Forward Windows"
        )
        chart_data["chart-wf"] = _fig_pair(fig)
        parts.append(_chart_html("chart-wf", height=240))

    parts.append("</div>")

    # Robustness panel.
    parts.append('<div class="panel">')
    parts.append('<h2>Robustness</h2>')
    parts.append(
        '<div class="subtitle">'
        'Bootstrap confidence interval (1000 IID resamples) and 4-subsample stability check.'
        '</div>'
    )
    lo, hi = rob.sharpe_confidence_interval
    ci_cls = "pos" if lo > 0 else ("neg" if hi < 0 else "")
    parts.append(f"""
<table>
<thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>
<tbody>
<tr><td>Sharpe 95% bootstrap CI</td>
    <td class="num {ci_cls}">[{lo:+.2f}, {hi:+.2f}]</td></tr>
<tr><td>Sharpe std (bootstrap)</td>
    <td class="num">{rob.sharpe_std_bootstrap:.3f}</td></tr>
<tr><td>Subsample positive fraction</td>
    <td class="num">{rob.subsample_positive_fraction:.0%}</td></tr>
</tbody></table>
""")
    parts.append("</div>")

    # Cost panel.
    if costs is not None:
        parts.append('<div class="panel">')
        parts.append('<h2>Transaction Cost Sensitivity</h2>')
        be = (
            f'{costs.breakeven_cost_bps:.1f} bps'
            if costs.breakeven_cost_bps is not None
            else "&gt; 1000 bps"
        )
        turnover = (
            f'{costs.annualized_turnover:.1f}x/yr'
            if costs.annualized_turnover is not None
            else "not estimated (no positions provided)"
        )
        parts.append(
            f'<div class="subtitle">'
            f'Break-even cost = <span class="mono">{be}</span> &middot; '
            f'annualized turnover = <span class="mono">{turnover}</span>'
            '</div>'
        )
        fig = charts.chart_cost_sensitivity(costs, title="Sharpe vs Transaction Cost")
        chart_data["chart-costs"] = _fig_pair(fig)
        parts.append(_chart_html("chart-costs", height=300))
        parts.append("</div>")

    return chart_data, "\n".join(parts)
