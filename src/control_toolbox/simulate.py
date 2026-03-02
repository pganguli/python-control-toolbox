"""
Run a closed-loop simulation and collect results.

Typical usage:
    ctrl = build_controller(plant, track={'x': 0.5}, p=100)
    # pass the result object directly to simulate
    resp = simulate(ctrl, t_end=5.0)
"""
# pylint: disable=invalid-name

import warnings
from dataclasses import dataclass

import numpy as np
from control import forced_response

from .controller import ControllerResult


@dataclass
class SimulationResult:
    """Collected output from a closed-loop simulation.

    Attributes:
        t:          Time vector, shape (N,).
        y:          Output trajectories, shape (n_outputs, N).
        u:          Control input trajectory, shape (n_inputs, N).
        output_labels: Output signal names, length n_outputs.
        input_labels:  Input signal names, length n_inputs.
        tracked:    {label: reference_value} for tracked outputs.
        dt:         Sampling period if discrete, 0 if continuous.
    """

    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    output_labels: list[str]
    input_labels: list[str]
    tracked: dict[str, float]
    dt: float


def _derive_input_labels(sys_cl, u_out):
    """Compute actuator input labels from sys_cl and u_out shape.

    This logic was previously in :func:`simulate`.  It returns a list of
    names with the following behaviour:

    * strip leading ``r_`` from ``sys_cl.input_labels``
    * if there are more names than actual actuator signals, truncate and
      warn
    * if there are fewer, append generic ``uN`` labels
    * if none remain, create a generic list based on ``u_out`` rows
    """
    labels = (
        [lbl.removeprefix("r_") for lbl in sys_cl.input_labels]
        if sys_cl.input_labels
        else []
    )
    n_act = u_out.shape[0]
    if len(labels) != n_act:
        if len(labels) > n_act:
            warnings.warn(
                "more sys_cl.input_labels than control inputs; truncating",
                stacklevel=2,
            )
            labels = labels[:n_act]
        else:
            labels += [f"u{i}" for i in range(len(labels), n_act)]
    if not labels:
        labels = [f"u{i}" for i in range(n_act)]
    return labels


def simulate(
    ctrl: ControllerResult,
    t_end: float,
    dt: float | None = None,
) -> SimulationResult:
    """Simulate a closed-loop system under constant reference commands.

    Drives the closed-loop system contained in ``ctrl`` with the
    constant reference vector defined by its ``tracked`` dictionary. The
    output and state trajectories are collected, and the control input is
    reconstructed using ``ctrl.K`` and ``ctrl.Nbar``.

    Args:
        ctrl:    ControllerResult returned by :func:`build_controller`.
        t_end:   Simulation end time (seconds).
        dt:      Time step. For continuous systems defaults to t_end/1000.
                 For discrete systems defaults to ``ctrl.sys_cl.dt``.

    Returns:
        SimulationResult with t, y, u, output_labels, input_labels,
        tracked, dt.
    """
    if dt is None:
        dt = (
            float(ctrl.sys_cl.dt) if ctrl.sys_cl.dt not in (0, None) else t_end / 1000.0
        )

    t = np.arange(0.0, t_end + dt / 2, dt)
    refs = np.array(list(ctrl.tracked.values()))

    t_out, y_out, x_out = forced_response(
        ctrl.sys_cl,
        timepts=t,
        inputs=np.outer(refs, np.ones(len(t))),
        return_states=True,
    )

    u_out = -ctrl.K @ x_out
    if ctrl.Nbar is not None:
        u_out = u_out + ctrl.Nbar @ (refs[:, None] * np.ones((1, len(t_out))))

    if y_out.ndim == 1:
        y_out = y_out[np.newaxis, :]
    if u_out.ndim == 1:
        u_out = u_out[np.newaxis, :]

    out_labels = (
        list(ctrl.sys_cl.output_labels)
        if ctrl.sys_cl.output_labels
        else [f"y{i}" for i in range(y_out.shape[0])]
    )

    in_labels = _derive_input_labels(ctrl.sys_cl, u_out)

    return SimulationResult(
        t=t_out,
        y=y_out,
        u=u_out,
        output_labels=out_labels,
        input_labels=in_labels,
        tracked=dict(ctrl.tracked),
        dt=dt,
    )
