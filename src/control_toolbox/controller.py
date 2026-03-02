"""
LQR controller synthesis for StateSpace plant models.

Typical workflow:
    from controltoolbox import build_controller

    ctrl = build_controller(plant, track={'x': 1.0, 'phi': 0.0}, p=50)
    # ctrl is a ControllerResult with attributes K, Nbar, sys_cl, tracked

    # familiar unpacking is still possible:
    K, Nbar, sys_cl = ctrl.K, ctrl.Nbar, ctrl.sys_cl

    sys_cl  — closed-loop StateSpace; inputs are reference signals r_<label>
    K       — state feedback gain matrix  (n_inputs x n_states)
    Nbar    — precompensator              (n_inputs x n_tracked)
              None if closed-loop DC gain is singular (regulates to zero only)
"""
# pylint: disable=invalid-name

import warnings
from dataclasses import dataclass

import numpy as np
from control import StateSpace, dlqr, lqr, ss


@dataclass
class ControllerResult:
    """Collected controller synthesis outputs.

    Attributes:
        K:       State feedback gain matrix (n_inputs x n_states).
        Nbar:    Precompensator (n_inputs x n_tracked) or ``None``.
        sys_cl:  Closed‑loop StateSpace model returned by ``build_controller``.
        tracked: Dictionary of tracked outputs and their reference values.
    """

    K: np.ndarray
    Nbar: np.ndarray | None
    sys_cl: StateSpace
    tracked: dict[str, float]


def _parse_tracking(
    plant: StateSpace, track: dict | None
) -> tuple[list[str], dict[str, float], np.ndarray]:
    """Interpret the ``track`` argument.

    Returns ``(labels, tracked, C_track)`` where ``labels`` is the list of
    plant output names, ``tracked`` maps those names to non-None reference
    values, and ``C_track`` is the corresponding rows of the plant output
    matrix.  This isolates a bulky dictionary comprehension from
    ``build_controller``.
    """
    C = np.array(plant.C, dtype=float)
    labels = (
        list(plant.output_labels)
        if plant.output_labels
        else [f"y{i}" for i in range(C.shape[0])]
    )
    track = track or {lbl: 0.0 for lbl in labels}
    tracked = {
        (labels[k] if isinstance(k, int) else k): float(v)
        for k, v in track.items()
        if v is not None
    }
    idx = [labels.index(lbl) for lbl in tracked]
    C_track = C[idx, :]
    return labels, tracked, C_track


def _compute_Nbar(A, B, C_track, K, is_discrete):
    """Calculate the precompensator Nbar from closed-loop data."""
    A_cl = A - B @ K
    if is_discrete:
        M = C_track @ np.linalg.pinv(np.eye(A_cl.shape[0]) - A_cl) @ B
    else:
        M = -C_track @ np.linalg.pinv(A_cl) @ B
    return np.linalg.pinv(M) if np.all(np.isfinite(M)) else None


def build_controller(
    plant: StateSpace,
    track: dict[str | int, float | None] | None = None,
    p: float = 1.0,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> ControllerResult:
    """LQR controller for a StateSpace plant.  Supports both continuous-
    and discrete-time systems (uses ``lqr`` or ``dlqr`` depending on
    ``plant.dt``). The returned closed-loop model preserves the sampling
    period so that downstream simulation routines behave correctly.
    """
    A = np.array(plant.A, dtype=float)
    B = np.array(plant.B, dtype=float)

    labels, tracked, C_track = _parse_tracking(plant, track)

    # LQR / DLQR depending on plant type; inline Q and R defaults
    if plant.dt not in (0, None):
        K, _, _ = dlqr(
            A,
            B,
            p * C_track.T @ C_track if Q is None else np.array(Q, dtype=float),
            np.eye(B.shape[1]) if R is None else np.array(R, dtype=float),
        )
        discrete = True
    else:
        K, _, _ = lqr(
            A,
            B,
            p * C_track.T @ C_track if Q is None else np.array(Q, dtype=float),
            np.eye(B.shape[1]) if R is None else np.array(R, dtype=float),
        )
        discrete = False

    Nbar = _compute_Nbar(A, B, C_track, K, discrete)
    if Nbar is not None and len(tracked) != B.shape[1]:
        warnings.warn(
            "n_tracked != n_inputs: Nbar is a pseudoinverse; "
            "exact tracking not guaranteed.",
            stacklevel=2,
        )

    sys_cl = ss(
        A - B @ K,
        B @ (Nbar if Nbar is not None else np.zeros((B.shape[1], len(tracked)))),
        np.array(plant.C, dtype=float),
        np.zeros((np.array(plant.C, dtype=float).shape[0], len(tracked))),
        inputs=[f"r_{lbl}" for lbl in tracked],
        outputs=labels,
        dt=plant.dt,
    )

    return ControllerResult(K=K, Nbar=Nbar, sys_cl=sys_cl, tracked=tracked)
