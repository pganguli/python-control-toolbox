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

    Args:
        plant:  Any StateSpace instance from systems.py.
        track:  Dict of {output_name_or_index: target_value | None}.
                None value = don't care. Omitted outputs = don't care.
                track=None regulates all outputs to zero.
        p:      Output-error weight (Q = p * C_track' @ C_track). Ignored if Q given.
        Q:      State-cost matrix  (n_states x n_states). Overrides p.
        R:      Input-cost matrix  (n_inputs x n_inputs). Defaults to identity.

    Returns:
        ControllerResult containing ``K``, ``Nbar`` and ``sys_cl`` (plus
        the ``tracked`` dictionary used to compute the controller).  If
        ``plant`` was discrete-time, the returned ``sys_cl`` has its
        ``dt`` field set accordingly and ``K`` is computed via ``dlqr``.  The
        precompensator ``Nbar`` is computed with the appropriate steady-
        state formula for continuous or discrete dynamics.
    """
    A = np.array(plant.A, dtype=float)
    B = np.array(plant.B, dtype=float)
    C = np.array(plant.C, dtype=float)
    n_in = B.shape[1]

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

    # LQR / DLQR depending on plant type
    Q = p * C_track.T @ C_track if Q is None else np.array(Q, dtype=float)
    R = np.eye(n_in) if R is None else np.array(R, dtype=float)
    is_discrete = plant.dt not in (0, None)
    if is_discrete:
        K, _, _ = dlqr(A, B, Q, R)
    else:
        K, _, _ = lqr(A, B, Q, R)

    # Nbar: solves C_track @ DC-gain(A_cl,B) @ Nbar = -I
    # For continuous DC gain = -C_track @ inv(A_cl) @ B
    # For discrete DC gain = C_track @ inv(I - A_cl) @ B
    A_cl = A - B @ K
    if is_discrete:
        # discrete-time steady-state: x = inv(I - A_cl) B Nbar r
        # require C_track @ inv(I - A_cl) B Nbar = I
        # use pseudo-inverse in case (I-A_cl) is singular
        M = C_track @ np.linalg.pinv(np.eye(A_cl.shape[0]) - A_cl) @ B
    else:
        # continuous-time steady-state: x = -inv(A_cl) B Nbar r
        # require -C_track @ inv(A_cl) B Nbar = I
        # use pseudo-inverse in case A_cl is singular
        M = -C_track @ np.linalg.pinv(A_cl) @ B
    Nbar = np.linalg.pinv(M) if np.all(np.isfinite(M)) else None
    if Nbar is not None and len(idx) != n_in:
        warnings.warn(
            "n_tracked != n_inputs: Nbar is a pseudoinverse; "
            "exact tracking not guaranteed.",
            stacklevel=2,
        )

    n_tracked = len(idx)
    B_cl = B @ (Nbar if Nbar is not None else np.zeros((n_in, n_tracked)))
    D_cl = np.zeros((C.shape[0], n_tracked))
    # preserve sampling period if discrete
    sys_cl = ss(
        A_cl,
        B_cl,
        C,
        D_cl,
        inputs=[f"r_{lbl}" for lbl in tracked],
        outputs=labels,
        dt=plant.dt,
    )

    return ControllerResult(K=K, Nbar=Nbar, sys_cl=sys_cl, tracked=tracked)
