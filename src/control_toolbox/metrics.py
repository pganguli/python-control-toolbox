"""
Performance metrics computed from a SimulationResult.

Typical usage:
    m = compute_metrics(resp)
    print(m.overshoot)        # {'x': 12.3}   (percent)
    print(m.settling_time)    # {'x': 1.41}   (seconds)
    print(m.steady_state_error) # {'x': 0.002}
"""

from dataclasses import dataclass

import numpy as np

from .simulate import SimulationResult


@dataclass
class Metrics:
    """Performance metrics for each tracked output.

    All dicts are keyed by output label and contain only tracked outputs
    (don't-cares are excluded since there is no reference to measure against).

    Attributes:
        overshoot:          Peak overshoot as a percentage of the reference
                            step. Positive means the output exceeded the
                            reference; 0 if no overshoot occurred.
        settling_time:      Time (s) for the output to enter and remain
                            within ±2% of the reference. None if the output
                            has not settled by the end of the simulation.
        steady_state_error: Absolute error between the final output value
                            and the reference, as a percentage of the
                            reference magnitude. Uses the mean of the last
                            5% of the time vector.
        rise_time:          Time (s) from 10% to 90% of the reference step.
                            None if the output never reaches 90%.
    """

    overshoot: dict[str, float]
    settling_time: dict[str, float | None]
    steady_state_error: dict[str, float]
    rise_time: dict[str, float | None]


def compute_metrics(resp: SimulationResult, settling_band: float = 0.02) -> Metrics:
    """Compute standard closed-loop performance metrics from a simulation.

    Args:
        resp:          SimulationResult from simulate().
        settling_band: Fraction of reference used as the settling tolerance
                       band (default 0.02 = ±2%).

    Returns:
        Metrics dataclass with per-output dicts for each tracked output.

    Notes:
        - Metrics are only computed for outputs present in resp.tracked.
        - All metrics assume a step reference starting from an initial
          output of zero (system starts at rest).
        - Overshoot and steady-state error are expressed as percentages of
          the reference magnitude.
        - If the reference is zero the percentage metrics are computed
          against the absolute deviation instead of a percentage.
    """
    # pylint: disable=too-many-locals
    overshoot_d = {}
    settling_time_d = {}
    steady_state_error_d = {}
    rise_time_d = {}

    t = resp.t
    n_tail = max(1, int(0.05 * len(t)))  # last 5% of samples for SS estimate

    for lbl, ref in resp.tracked.items():
        if lbl not in resp.output_labels:
            continue
        idx = resp.output_labels.index(lbl)
        y = resp.y[idx]

        ref_mag = abs(ref) if abs(ref) > 1e-12 else 1.0  # avoid div-by-zero

        # --- Steady-state estimate -------------------------------------------
        y_ss = float(np.mean(y[-n_tail:]))
        ss_err = abs(y_ss - ref) / ref_mag * 100.0
        steady_state_error_d[lbl] = ss_err

        # --- Overshoot -------------------------------------------------------
        if ref >= 0:
            peak = float(np.max(y))
            os = max(0.0, (peak - ref) / ref_mag * 100.0)
        else:
            peak = float(np.min(y))
            os = max(0.0, (ref - peak) / ref_mag * 100.0)
        overshoot_d[lbl] = os

        # --- Settling time (2% band by default) ------------------------------
        band = settling_band * ref_mag
        in_band = np.abs(y - ref) <= band
        # Find the last time the signal is outside the band, then add one step
        outside = np.where(~in_band)[0]
        if len(outside) == 0:
            settling_time_d[lbl] = float(t[0])  # already settled at t=0
        elif in_band[-1]:
            settling_time_d[lbl] = float(t[outside[-1] + 1])
        else:
            settling_time_d[lbl] = None  # not settled by end of sim

        # --- Rise time (10% to 90% of step) ----------------------------------
        lo = 0.10 * ref
        hi = 0.90 * ref
        if ref >= 0:
            i_lo = np.where(y >= lo)[0]
            i_hi = np.where(y >= hi)[0]
        else:
            i_lo = np.where(y <= lo)[0]
            i_hi = np.where(y <= hi)[0]

        if len(i_lo) > 0 and len(i_hi) > 0:
            rise_time_d[lbl] = float(t[i_hi[0]] - t[i_lo[0]])
        else:
            rise_time_d[lbl] = None

    return Metrics(
        overshoot=overshoot_d,
        settling_time=settling_time_d,
        steady_state_error=steady_state_error_d,
        rise_time=rise_time_d,
    )
