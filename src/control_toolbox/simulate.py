"""
Run a closed-loop simulation and collect results.

Typical usage:
    ctrl = build_controller(plant, track={'x': 0.5}, p=100)
    # pass the result object directly to simulate
    resp = simulate(ctrl, t_end=5.0)
"""

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
    sys_cl = ctrl.sys_cl
    K = ctrl.K
    Nbar = ctrl.Nbar
    tracked = dict(ctrl.tracked)

    is_discrete = sys_cl.dt not in (0, None)

    if dt is None:
        dt = float(sys_cl.dt) if is_discrete else t_end / 1000.0

    t = np.arange(0.0, t_end + dt / 2, dt)

    # Reference matrix: one row per tracked output, constant step
    refs = np.array([tracked[lbl] for lbl in tracked])
    U = np.outer(refs, np.ones(len(t)))  # (n_tracked, N)

    t_out, y_out, x_out = forced_response(
        sys_cl, timepts=t, inputs=U, return_states=True
    )

    # Reconstruct u(t) = -K @ x(t) + Nbar @ r
    r_vec = refs[:, None] * np.ones((1, len(t_out)))  # broadcast to (n_tracked, N)
    u_out = -K @ x_out + (
        Nbar @ r_vec if Nbar is not None else np.zeros((K.shape[0], len(t_out)))
    )

    # Ensure y is always 2D (n_outputs, N) regardless of SISO squeeze
    if y_out.ndim == 1:
        y_out = y_out[np.newaxis, :]
    if u_out.ndim == 1:
        u_out = u_out[np.newaxis, :]

    out_labels = (
        list(sys_cl.output_labels)
        if sys_cl.output_labels
        else [f"y{i}" for i in range(y_out.shape[0])]
    )
    # derive input labels from the closed-loop system's input names.
    # sys_cl.input_labels refer to reference channels (r_x, r_phi, ...),
    # which may outnumber the actual actuator signals in u_out.  Use the
    # shape of u_out to size the returned labels correctly, truncating when
    # necessary or padding with generic names.
    # padding as necessary.
    in_labels = (
        [lbl.removeprefix("r_") for lbl in sys_cl.input_labels]
        if sys_cl.input_labels
        else []
    )
    # adjust length
    n_act = u_out.shape[0]
    if len(in_labels) != n_act:
        # this usually occurs when multiple references drive a single
        # actuator (n_tracked > n_inputs).  Keep the first n_act names and
        # issue a warning if we had to drop some.
        if len(in_labels) > n_act:
            warnings.warn(
                "more sys_cl.input_labels than control inputs; truncating",
                stacklevel=2,
            )
            in_labels = in_labels[:n_act]
        else:
            # too few labels, generate generic names for remaining inputs
            in_labels += [f"u{i}" for i in range(len(in_labels), n_act)]
    if not in_labels:
        in_labels = [f"u{i}" for i in range(n_act)]

    return SimulationResult(
        t=t_out,
        y=y_out,
        u=u_out,
        output_labels=out_labels,
        input_labels=in_labels,
        tracked=dict(tracked),
        dt=dt,
    )
