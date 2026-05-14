"""Report generation: from raw returns to a self-contained HTML deliverable."""

from alpha_lens.report.generator import autopsy
from alpha_lens.report.html_report import render_html_report

__all__ = ["autopsy", "render_html_report"]
