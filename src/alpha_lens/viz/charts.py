"""Plotly chart builders for the autopsy report.

Each function takes typed analysis results and returns a
:class:`plotly.graph_objects.Figure`. The charts are designed to be
read by a PM or CIO in <5 seconds each — clear headline, supporting
detail, no chartjunk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from alpha_lens.analysis.statistics import (
    cumulative_returns,
    drawdown_series,
    rolling_sharpe,
)
from alpha_lens.core.types import (
    CorrelationAnalysis,
    CostAnalysis,
    DecayMetrics,
    DrawdownAnalysis,
    FactorAttribution,
    RegimeAnalysis,
)
from alpha_lens.viz.styles import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    ACCENT_TERTIARY,
    BAD,
    BORDER,
    DIVERGING_COLORSCALE,
    GOOD,
    GRID,
    INK,
    INK_MUTED,
    NEUTRAL,
    REGIME_COLORS,
    WARN,
    base_layout,
)

__all__ = [
    "chart_attribution_bars",
    "chart_correlation_heatmap",
    "chart_cost_sensitivity",
    "chart_cumulative_returns",
    "chart_drawdown_underwater",
    "chart_ic_decay",
    "chart_monthly_returns_heatmap",
    "chart_regime_overlay",
    "chart_regime_performance",
    "chart_rolling_sharpe",
    "chart_walkforward_bars",
]


# ----------------------------------------------------------------------------
# Equity curve and overlays
# ----------------------------------------------------------------------------


def chart_cumulative_returns(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    title: str = "Cumulative Returns",
) -> go.Figure:
    """Equity curve, optionally with a benchmark overlay."""
    cum = cumulative_returns(returns)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum.values,
            mode="lines",
            name="Strategy",
            line={"color": ACCENT_PRIMARY, "width": 2.0},
            fill="tozeroy",
            fillcolor="rgba(246, 193, 119, 0.08)",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Cum: %{y:.3f}<extra></extra>",
        )
    )
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bench_cum = cumulative_returns(benchmark_returns.reindex(returns.index).ffill())
        fig.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode="lines",
                name="Benchmark",
                line={"color": INK_MUTED, "width": 1.2, "dash": "dot"},
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Bench: %{y:.3f}<extra></extra>",
            )
        )
    layout = base_layout(height=340, title=title)
    layout["yaxis"]["title"] = "Wealth (start = 1)"
    fig.update_layout(**layout)
    return fig


def chart_regime_overlay(
    returns: pd.Series,
    regime: RegimeAnalysis,
    *,
    title: str = "Cumulative Returns with Regime Overlay",
) -> go.Figure:
    """Equity curve with regime-colored background bands."""
    cum = cumulative_returns(returns)
    fig = go.Figure()

    # Add background shapes per regime run.
    shapes = _regime_shapes(regime.labels, y_min=float(cum.min()) * 0.95, y_max=float(cum.max()) * 1.05)
    fig.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum.values,
            mode="lines",
            name="Strategy",
            line={"color": ACCENT_PRIMARY, "width": 2.0},
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Cum: %{y:.3f}<extra></extra>",
        )
    )

    # Synthetic legend entries for each regime that appears.
    seen_regimes = set(regime.labels.unique())
    for r_label, r_color in REGIME_COLORS.items():
        if r_label in seen_regimes:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={"size": 12, "color": r_color, "symbol": "square"},
                    name=r_label,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    layout = base_layout(height=380, title=title)
    layout["shapes"] = shapes
    layout["yaxis"]["title"] = "Wealth (start = 1)"
    fig.update_layout(**layout)
    return fig


def _regime_shapes(labels: pd.Series, *, y_min: float, y_max: float) -> list[dict]:
    """Build Plotly background rectangles for contiguous regime runs."""
    shapes: list[dict] = []
    if len(labels) == 0:
        return shapes

    current_label = labels.iloc[0]
    run_start = labels.index[0]
    for i in range(1, len(labels)):
        if labels.iloc[i] != current_label:
            shapes.append(
                {
                    "type": "rect",
                    "x0": run_start,
                    "x1": labels.index[i],
                    "y0": y_min,
                    "y1": y_max,
                    "fillcolor": REGIME_COLORS.get(current_label, INK_MUTED),
                    "opacity": 0.10,
                    "layer": "below",
                    "line": {"width": 0},
                }
            )
            current_label = labels.iloc[i]
            run_start = labels.index[i]
    # Final run.
    shapes.append(
        {
            "type": "rect",
            "x0": run_start,
            "x1": labels.index[-1],
            "y0": y_min,
            "y1": y_max,
            "fillcolor": REGIME_COLORS.get(current_label, INK_MUTED),
            "opacity": 0.10,
            "layer": "below",
            "line": {"width": 0},
        }
    )
    return shapes


# ----------------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------------


def chart_drawdown_underwater(returns: pd.Series, *, title: str = "Underwater Plot") -> go.Figure:
    """Drawdown depth over time."""
    dd = drawdown_series(returns)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            mode="lines",
            line={"color": BAD, "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(235, 111, 146, 0.20)",
            name="Drawdown",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>DD: %{y:.1f}%<extra></extra>",
        )
    )
    layout = base_layout(height=260, title=title)
    layout["yaxis"]["title"] = "Drawdown (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)
    return fig


def chart_drawdown_waterfall(dd_analysis: DrawdownAnalysis, *, title: str = "Worst Drawdowns") -> go.Figure:
    """Bar chart of the top drawdown events sorted by depth."""
    events = dd_analysis.events[:10]
    fig = go.Figure()
    if not events:
        fig.add_annotation(
            text="No drawdowns to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"color": INK_MUTED, "size": 13},
        )
        fig.update_layout(**base_layout(height=300, title=title))
        return fig

    x = [e.peak_date.strftime("%Y-%m") for e in events]
    y = [e.depth * 100 for e in events]
    text = [
        f"Depth: {e.depth:.1%}<br>Duration: {e.duration_days}d<br>Recovery: "
        f"{e.recovery_days}d" if e.recovery_days is not None else
        f"Depth: {e.depth:.1%}<br>Duration: {e.duration_days}d<br>Recovery: unrecovered"
        for e in events
    ]
    colors = [
        REGIME_COLORS.get(e.dominant_regime.value, BAD) if e.dominant_regime else BAD
        for e in events
    ]
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker={"color": colors},
            hovertext=text,
            hovertemplate="<b>Peak: %{x}</b><br>%{hovertext}<extra></extra>",
        )
    )
    layout = base_layout(height=300, title=title)
    layout["yaxis"]["title"] = "Drawdown (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["title"] = "Peak date"
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# Rolling Sharpe
# ----------------------------------------------------------------------------


def chart_rolling_sharpe(
    returns: pd.Series,
    *,
    window: int = 63,
    regime: RegimeAnalysis | None = None,
    title: str | None = None,
) -> go.Figure:
    """Rolling Sharpe ratio, optionally with regime background."""
    title = title or f"Rolling Sharpe (window={window} days)"
    rs = rolling_sharpe(returns, window=window).dropna()
    fig = go.Figure()

    if regime is not None:
        y_min = float(rs.min()) - 0.5
        y_max = float(rs.max()) + 0.5
        layout = base_layout(height=300, title=title)
        layout["shapes"] = _regime_shapes(regime.labels, y_min=y_min, y_max=y_max)
    else:
        layout = base_layout(height=300, title=title)

    fig.add_trace(
        go.Scatter(
            x=rs.index,
            y=rs.values,
            mode="lines",
            line={"color": ACCENT_SECONDARY, "width": 1.6},
            name="Sharpe",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )
    # Zero-reference line.
    fig.add_hline(y=0, line={"color": BORDER, "width": 1, "dash": "dot"})

    layout["yaxis"]["title"] = "Annualized Sharpe"
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# Regime performance
# ----------------------------------------------------------------------------


def chart_regime_performance(regime: RegimeAnalysis, *, title: str = "Performance by Regime") -> go.Figure:
    """Bar chart: Sharpe ratio per regime."""
    summaries = regime.summaries
    fig = go.Figure()
    if not summaries:
        return fig

    labels = [s.regime.value for s in summaries]
    sharpes = [s.sharpe_ratio for s in summaries]
    fractions = [s.fraction for s in summaries]
    colors = [REGIME_COLORS.get(label, INK_MUTED) for label in labels]
    text = [
        f"{f:.0%} of time<br>annRet: {s.mean_return_annualized:.1%}<br>vol: {s.volatility_annualized:.1%}"
        for s, f in zip(summaries, fractions, strict=False)
    ]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=sharpes,
            marker={"color": colors},
            hovertext=text,
            hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<br>%{hovertext}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": BORDER, "width": 1, "dash": "dot"})
    layout = base_layout(height=280, title=title)
    layout["yaxis"]["title"] = "Annualized Sharpe"
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------------


def chart_attribution_bars(attr: FactorAttribution, *, title: str = "Factor Attribution") -> go.Figure:
    """Horizontal bar chart of factor loadings with t-stats annotations."""
    fig = go.Figure()
    names = attr.factor_names
    betas = [attr.betas[n] for n in names]
    t_stats = [attr.t_stats[n] for n in names]

    # Color by significance: |t| > 2 is "significant".
    colors = [
        GOOD if abs(t) > 2 and beta > 0 else
        BAD if abs(t) > 2 and beta < 0 else
        INK_MUTED
        for t, beta in zip(t_stats, betas, strict=False)
    ]

    fig.add_trace(
        go.Bar(
            x=betas,
            y=names,
            orientation="h",
            marker={"color": colors},
            text=[f"t = {t:+.1f}" for t in t_stats],
            textposition="auto",
            insidetextfont={"color": INK, "size": 11},
            outsidetextfont={"color": INK_MUTED, "size": 11},
            hovertemplate="<b>%{y}</b><br>β = %{x:.3f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": BORDER, "width": 1, "dash": "dot"})
    layout = base_layout(height=max(220, 60 + len(names) * 30), title=title)
    layout["xaxis"]["title"] = "Factor loading (β)"
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# IC decay
# ----------------------------------------------------------------------------


def chart_ic_decay(decay: DecayMetrics, *, title: str = "Information Coefficient by Horizon") -> go.Figure:
    """IC at each forward horizon, with significance shading."""
    horizons = sorted(decay.ic_by_horizon.keys())
    ics = [decay.ic_by_horizon[h] for h in horizons]
    pvals = [decay.ic_pvalue_by_horizon[h] for h in horizons]
    # Color by significance.
    colors = [GOOD if p < 0.05 else INK_MUTED for p in pvals]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(h) for h in horizons],
            y=ics,
            marker={"color": colors},
            text=[f"p={p:.2f}" for p in pvals],
            textposition="outside",
            textfont={"color": INK_MUTED, "size": 10},
            hovertemplate="<b>Horizon: %{x}d</b><br>IC: %{y:.4f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": BORDER, "width": 1, "dash": "dot"})

    if decay.estimated_half_life_days is not None:
        fig.add_annotation(
            text=f"Half-life ≈ {decay.estimated_half_life_days:.0f} days",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.96,
            xanchor="right",
            showarrow=False,
            font={"color": ACCENT_PRIMARY, "size": 12, "family": "JetBrains Mono, monospace"},
            bgcolor="rgba(0,0,0,0)",
        )

    layout = base_layout(height=280, title=title)
    layout["xaxis"]["title"] = "Forward horizon (trading days)"
    layout["yaxis"]["title"] = "Rank IC (Spearman)"
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------------


def chart_correlation_heatmap(corr: CorrelationAnalysis, *, title: str = "Factor Correlations") -> go.Figure:
    """Correlation matrix heatmap."""
    matrix = corr.correlation_matrix
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=DIVERGING_COLORSCALE,
            zmin=-1,
            zmax=1,
            colorbar={"tickfont": {"color": INK_MUTED}, "outlinewidth": 0},
            text=matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"color": INK, "family": "JetBrains Mono, monospace", "size": 11},
            hovertemplate="<b>%{y} × %{x}</b><br>ρ = %{z:.3f}<extra></extra>",
        )
    )
    layout = base_layout(height=max(280, 80 + len(matrix.columns) * 30), title=title)
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# Cost sensitivity
# ----------------------------------------------------------------------------


def chart_cost_sensitivity(costs: CostAnalysis, *, title: str = "Sharpe vs Transaction Cost") -> go.Figure:
    """Sharpe at each cost level, with the breakeven point marked."""
    costs_bps = sorted(costs.cost_sensitivity.keys())
    sharpes = [costs.cost_sensitivity[c] for c in costs_bps]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=costs_bps,
            y=sharpes,
            mode="lines+markers",
            marker={"color": ACCENT_PRIMARY, "size": 8},
            line={"color": ACCENT_PRIMARY, "width": 2.0},
            name="Sharpe",
            hovertemplate="Cost: %{x:.1f} bps<br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": BORDER, "width": 1, "dash": "dot"})

    if costs.breakeven_cost_bps is not None:
        fig.add_vline(
            x=costs.breakeven_cost_bps,
            line={"color": WARN, "width": 1.5, "dash": "dash"},
            annotation_text=f"breakeven: {costs.breakeven_cost_bps:.1f}",
            annotation_position="top",
            annotation_font={"color": WARN, "size": 11},
        )

    layout = base_layout(height=300, title=title)
    layout["xaxis"]["title"] = "Cost per unit turnover (bps)"
    layout["yaxis"]["title"] = "Annualized Sharpe"
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------


def chart_walkforward_bars(walk_forward_sharpes: list[float], *, title: str = "Walk-Forward Sharpes") -> go.Figure:
    """Bar chart of Sharpe ratio in each walk-forward window."""
    fig = go.Figure()
    if not walk_forward_sharpes:
        fig.update_layout(**base_layout(height=240, title=title))
        return fig

    labels = [f"W{i + 1}" for i in range(len(walk_forward_sharpes))]
    colors = [GOOD if s > 0 else BAD for s in walk_forward_sharpes]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=walk_forward_sharpes,
            marker={"color": colors},
            hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": BORDER, "width": 1, "dash": "dot"})
    layout = base_layout(height=240, title=title)
    layout["yaxis"]["title"] = "Annualized Sharpe"
    fig.update_layout(**layout)
    return fig


# ----------------------------------------------------------------------------
# Monthly returns heatmap
# ----------------------------------------------------------------------------


def chart_monthly_returns_heatmap(returns: pd.Series, *, title: str = "Monthly Returns") -> go.Figure:
    """Calendar heatmap of monthly returns."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)  # type: ignore[arg-type]
    df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }
    )
    pivot = df.pivot_table(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))

    z = (pivot.values * 100).round(2)
    text = [
        [f"{v:.1f}" if not np.isnan(v) else "" for v in row]
        for row in z
    ]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=month_labels,
            y=pivot.index.astype(str).tolist(),
            colorscale=DIVERGING_COLORSCALE,
            zmid=0,
            text=text,
            texttemplate="%{text}",
            textfont={"color": INK, "family": "JetBrains Mono, monospace", "size": 10},
            colorbar={"ticksuffix": "%", "tickfont": {"color": INK_MUTED}, "outlinewidth": 0},
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        )
    )
    layout = base_layout(height=max(220, 80 + len(pivot.index) * 24), title=title)
    fig.update_layout(**layout)
    return fig


# Suppress unused import warnings.
_ = (NEUTRAL, ACCENT_TERTIARY, GRID)
