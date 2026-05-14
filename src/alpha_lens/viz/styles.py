"""Visual styling for alpha-lens reports.

A consistent visual language for every chart in the report. We use a
dark theme with a warm-accent palette — distinct from the generic
Plotly defaults and the "Bloomberg orange-on-black" stereotype.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Color tokens
# ----------------------------------------------------------------------------

BG_DARK = "#0e1116"
BG_PANEL = "#161b22"
BG_PANEL_2 = "#1c2128"
INK = "#e6edf3"
INK_MUTED = "#8b949e"
BORDER = "#30363d"
GRID = "#21262d"

# Headline palette — used to differentiate components.
ACCENT_PRIMARY = "#f6c177"   # warm amber, the "hero" color
ACCENT_SECONDARY = "#9ccfd8"  # cool cyan, complementary
ACCENT_TERTIARY = "#c4a7e7"   # muted lavender

# Sentiment palette.
GOOD = "#a3be8c"
WARN = "#f6c177"
BAD = "#eb6f92"
NEUTRAL = "#7aa2f7"

# Regime palette (must match the 4 RegimeLabel values).
REGIME_COLORS: dict[str, str] = {
    "bull": "#a3be8c",
    "bear": "#eb6f92",
    "high_vol": "#f6c177",
    "sideways": "#7e8a99",
}

# Plotly continuous colorscales.
DIVERGING_COLORSCALE = [
    [0.0, "#eb6f92"],   # negative
    [0.5, "#21262d"],   # zero
    [1.0, "#a3be8c"],   # positive
]

SEQUENTIAL_COLORSCALE = [
    [0.0, "#0e1116"],
    [0.5, "#9ccfd8"],
    [1.0, "#f6c177"],
]


# ----------------------------------------------------------------------------
# Typography
# ----------------------------------------------------------------------------

FONT_BODY = (
    '"IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, '
    '"Apple Color Emoji", "Segoe UI Emoji", sans-serif'
)
FONT_MONO = (
    '"JetBrains Mono", "SF Mono", "Menlo", "Consolas", monospace'
)
FONT_DISPLAY = (
    '"IBM Plex Serif", Georgia, "Times New Roman", serif'
)


# ----------------------------------------------------------------------------
# Plotly layout defaults
# ----------------------------------------------------------------------------

def base_layout(*, height: int = 380, title: str | None = None) -> dict:
    """Default Plotly layout dict for alpha-lens charts."""
    layout: dict = {
        "paper_bgcolor": BG_PANEL,
        "plot_bgcolor": BG_PANEL,
        "font": {
            "family": FONT_BODY,
            "size": 12,
            "color": INK,
        },
        "margin": {"l": 50, "r": 30, "t": 50 if title else 20, "b": 40},
        "height": height,
        "xaxis": {
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "linecolor": BORDER,
            "tickfont": {"color": INK_MUTED, "size": 11},
            "title": {"font": {"color": INK_MUTED, "size": 12}},
        },
        "yaxis": {
            "gridcolor": GRID,
            "zerolinecolor": BORDER,
            "linecolor": BORDER,
            "tickfont": {"color": INK_MUTED, "size": 11},
            "title": {"font": {"color": INK_MUTED, "size": 12}},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": BORDER,
            "borderwidth": 0,
            "font": {"color": INK, "size": 11},
        },
        "hoverlabel": {
            "bgcolor": BG_PANEL_2,
            "bordercolor": BORDER,
            "font": {"color": INK, "family": FONT_MONO, "size": 11},
        },
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"color": INK, "size": 14, "family": FONT_BODY},
            "x": 0.0,
            "xanchor": "left",
        }
    return layout
