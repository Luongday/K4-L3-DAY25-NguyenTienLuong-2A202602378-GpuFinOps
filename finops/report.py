"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_economics: dict | None = None,
                 extra_sections: list | None = None) -> str:
    """Return a markdown cost-optimization report.

    unit_economics: optional {"baseline_per_m", "optimized_per_m"} in $/1M-token.
    extra_sections: optional list of markdown blocks appended before the footer.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
    ]
    if unit_economics:
        b, o = unit_economics["baseline_per_m"], unit_economics["optimized_per_m"]
        drop = (1 - o / b) * 100 if b else 0.0
        lines += ["", f"**Inference unit cost:** ${b:.3f} -> ${o:.3f} per 1M tokens (**-{drop:.0f}%**)"]
    lines += [
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) | % of baseline |",
        "|---|---|---|",
    ]
    for name, amount in levers.items():
        share = (amount / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
        lines.append(f"| {name} | ${amount:,.0f} | {share:.1f}% |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
        ]
        if "cheapest_region" in sustainability:
            lines += [
                f"- Cleanest region: {sustainability.get('best_region', 'n/a')}",
                f"- Cheapest region: {sustainability['cheapest_region']}",
                f"- Best balanced region: {sustainability.get('balanced_region', 'n/a')}",
            ]
        else:
            lines.append(f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}")
    for block in extra_sections or []:
        lines += ["", block.rstrip()]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline: float | None = None) -> str:
    """Write the savings chart PNG and return its path ("" if matplotlib is absent).

    With `baseline`, draws a true waterfall: baseline bar, one descending step per lever,
    final optimized bar. Without it, a plain per-lever bar chart.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    if baseline is None:
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
    else:
        labels = ["Baseline"] + names + ["Optimized"]
        level = baseline
        ax.bar(0, baseline, color="#8a8a8a")
        ax.text(0, baseline, f"${baseline:,.0f}", ha="center", va="bottom", fontsize=9)
        for i, v in enumerate(vals, start=1):
            ax.bar(i, v, bottom=level - v, color="#2e8a57")
            ax.text(i, level, f"-${v:,.0f}", ha="center", va="bottom", fontsize=9)
            level -= v
        ax.bar(len(vals) + 1, level, color="#2e548a")
        ax.text(len(vals) + 1, level, f"${level:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("USD / month")
    ax.set_title("GPU cost: baseline to optimized, by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
