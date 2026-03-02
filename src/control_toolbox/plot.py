"""
Plotting for closed-loop simulation results.

Typical usage:
    resp = simulate(sys_cl, K, Nbar, tracked, t_end=5.0)
    plot_response(resp)
    plot_response(resp, metrics=compute_metrics(resp))
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from .metrics import Metrics
from .simulate import SimulationResult

# Colour palette: outputs cycle through blues/reds, inputs through greens
_OUTPUT_COLOURS = ["#2563eb", "#dc2626", "#7c3aed", "#d97706", "#0891b2"]
_INPUT_COLOURS = ["#16a34a", "#65a30d", "#0d9488", "#ca8a04"]


def plot_response(
    resp: SimulationResult,
    metrics: Metrics | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot output trajectories and control inputs from a simulation.

    Creates a figure with one subplot per output plus one subplot per
    control input. Tracked outputs show a dashed reference line. If
    metrics are supplied, the settling-time band and peak overshoot are
    annotated on the relevant subplots.

    Args:
        resp:    SimulationResult from simulate().
        metrics: Optional Metrics from compute_metrics(). When provided,
                 annotates settling band, overshoot, and settling time.
        title:   Figure suptitle. Auto-generated from tracked dict if None.
        figsize: (width, height) in inches. Auto-sized if None.

    Returns:
        The matplotlib Figure, so the caller can save or further modify it.
    """
    n_out = resp.y.shape[0]
    n_in = resp.u.shape[0]
    n_plots = n_out + n_in

    if figsize is None:
        figsize = (10, 2.8 * n_plots)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    if title is None:
        parts = [f"{lbl} → {val}" for lbl, val in resp.tracked.items()]
        title = "Closed-loop response:  " + ",   ".join(parts)
    fig.suptitle(title, fontsize=12)

    # --- Output subplots -----------------------------------------------------
    for i, lbl in enumerate(resp.output_labels):
        ax = axes[i]
        colour = _OUTPUT_COLOURS[i % len(_OUTPUT_COLOURS)]
        y = resp.y[i]

        ax.plot(resp.t, y, color=colour, lw=2, label=lbl)

        if lbl in resp.tracked:
            ref = resp.tracked[lbl]
            ax.axhline(
                ref, color=colour, lw=1, ls="--", alpha=0.5, label=f"reference ({ref})"
            )

            if metrics is not None:
                band = 0.02 * (abs(ref) if abs(ref) > 1e-12 else 1.0)
                ax.axhspan(
                    ref - band, ref + band, alpha=0.08, color=colour, label="±2% band"
                )

                st = metrics.settling_time.get(lbl)
                if st is not None:
                    ax.axvline(st, color=colour, lw=1, ls=":", alpha=0.7)
                    ax.annotate(
                        f"  t_s={st:.2f}s",
                        xy=(st, ref),
                        fontsize=8,
                        color=colour,
                        va="center",
                    )

                os_pct = metrics.overshoot.get(lbl, 0.0)
                if os_pct > 0.1:
                    peak_idx = np.argmax(y) if ref >= 0 else np.argmin(y)
                    ax.annotate(
                        f"OS={os_pct:.1f}%",
                        xy=(resp.t[peak_idx], y[peak_idx]),
                        xytext=(0, 8),
                        textcoords="offset points",
                        fontsize=8,
                        color=colour,
                        ha="center",
                    )

        ax.set_ylabel(lbl)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    # --- Input subplots ------------------------------------------------------
    for j, lbl in enumerate(resp.input_labels):
        ax = axes[n_out + j]
        colour = _INPUT_COLOURS[j % len(_INPUT_COLOURS)]
        ax.plot(resp.t, resp.u[j], color=colour, lw=2, label=f"u: {lbl}")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_ylabel(f"u ({lbl})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    axes[-1].set_xlabel("Time  (s)")
    plt.tight_layout()
    return fig
