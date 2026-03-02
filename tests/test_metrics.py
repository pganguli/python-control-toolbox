"""Unit tests for the metrics computation helper."""

import numpy as np
import pytest

from control_toolbox import SimulationResult, compute_metrics


def make_response(y_values, ref, dt=0.1):
    """Helper constructing a minimal SimulationResult for testing."""
    t = np.arange(0.0, dt * len(y_values), dt)
    y = np.array([y_values])
    u = np.zeros_like(y)
    return SimulationResult(
        t=t,
        y=y,
        u=u,
        output_labels=["out"],
        input_labels=["in"],
        tracked={"out": ref},
        dt=dt,
    )


def test_metrics_standard_step():
    """Metrics reflect overshoot, settling and rise time for a typical step."""
    resp = make_response([0, 0.5, 1.2, 1.05, 1.0, 1.0], ref=1.0, dt=0.1)
    m = compute_metrics(resp)
    assert m.overshoot["out"] == pytest.approx(20.0, rel=1e-3)
    assert m.settling_time["out"] is not None
    assert m.steady_state_error["out"] == pytest.approx(0.0, abs=1e-6)
    assert m.rise_time["out"] is not None


def test_metrics_no_overshoot_or_rise():
    """Handles zero‐reference signals correctly (no overshoot, zero rise)."""
    resp = make_response([0, 0, 0, 0], ref=0.0, dt=0.1)
    m = compute_metrics(resp)
    assert m.overshoot["out"] == 0.0
    assert m.rise_time["out"] == pytest.approx(0.0)


def test_metrics_not_settled_and_negative_ref():
    """Non‑settling and negative reference cases are handled gracefully."""
    resp = make_response([0, -0.5, -0.8, -0.7], ref=-1.0, dt=0.1)
    m = compute_metrics(resp)
    assert m.settling_time["out"] is None
    # overshoot formula returns 0 for this trajectory
    assert m.overshoot["out"] == pytest.approx(0.0)
