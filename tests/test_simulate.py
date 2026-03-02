"""Tests for the simulate() function and related behaviours."""

import pytest

from control_toolbox import InvertedPendulum, SpringMass, build_controller, simulate


def test_simulate_basic_shapes_and_labels():
    """Output and input array shapes/labels should be consistent."""
    plant = SpringMass()
    ctrl = build_controller(plant, track={"x": 1.0})
    resp = simulate(ctrl, t_end=1.0, dt=0.1)
    # 1 input, 1 output
    assert resp.y.shape[0] == 1
    assert resp.u.shape[0] == 1
    assert resp.output_labels == ["x"]
    # input label equals the tracked output name after stripping 'r_'
    assert resp.input_labels == ["x"]
    assert resp.dt == pytest.approx(0.1)


def test_simulate_default_dt_continuous():
    """If dt not provided for a continuous plant it is set to t_end/1000."""
    plant = SpringMass()
    ctrl = build_controller(plant, track={"x": 0.5})
    resp = simulate(ctrl, t_end=0.5)
    # dt should be t_end/1000
    assert resp.dt == pytest.approx(0.5 / 1000)


def test_input_label_truncation_warning():
    """Simulation warns and truncates input labels when too many references."""
    # build a controller with two tracked outputs on a single-input plant
    pend = InvertedPendulum()
    # tracking two signals on a 1-input system triggers the warning in
    # build_controller and also leads simulate() to issue a truncation warning
    with pytest.warns(UserWarning, match="n_tracked != n_inputs"):
        ctrl = build_controller(pend, track={"x": 0.1, "phi": 0.0}, p=50)

    with pytest.warns(UserWarning, match="truncating"):
        resp = simulate(ctrl, t_end=0.1, dt=0.01)
    # ensure labels were truncated (should still drop the prefix)
    assert resp.input_labels == ["x"]
