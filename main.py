#!/usr/bin/env python3
"""
Example script demonstrating plant setup, control synthesis,
and simulation using controltoolbox.
"""

from control_toolbox import (
    InvertedPendulum,
    add_delay,
    build_controller,
    compute_metrics,
    discretize,
    plot_response,
    simulate,
)


def main() -> None:
    """Demo script: create plant, design controller, simulate and save plot."""
    plant = InvertedPendulum()
    plant = add_delay(plant, T_input=0.5, T_output=0.5)
    plant = discretize(plant, dt=0.01)

    ctrl = build_controller(plant, track={"x": 0.5, "phi": 0.0}, p=200)
    resp = simulate(ctrl, t_end=5.0)

    fig = plot_response(resp, metrics=compute_metrics(resp))
    fig.savefig("response.png")


if __name__ == "__main__":
    main()
