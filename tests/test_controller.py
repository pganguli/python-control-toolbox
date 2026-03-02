"""Tests for controller synthesis utilities."""
# pylint: disable=invalid-name

import numpy as np
import pytest

from control_toolbox import (
    ControllerResult,
    InvertedPendulum,
    SpringMass,
    build_controller,
)


def test_build_controller_continuous_default():
    """Continuous‑time controller should produce sensible gains and Nbar."""
    plant = SpringMass()
    ctrl = build_controller(plant, track={"x": 1.0}, p=10)
    assert isinstance(ctrl, ControllerResult)
    assert ctrl.K.shape == (1, 2)
    assert ctrl.Nbar is not None
    assert "x" in ctrl.tracked


def test_build_controller_discrete_dt_preserved():
    """Discrete plant controller retains sampling period."""
    plant = SpringMass()
    disc = plant.sample(0.1)  # uses control library sampling
    ctrl = build_controller(disc, track={"x": 2.0})
    assert ctrl.sys_cl.dt == pytest.approx(0.1)
    assert ctrl.K.shape == (1, 2)


def test_controller_custom_Q_R():
    """Custom Q/R matrices should produce different gains."""
    plant = SpringMass()
    Q = np.eye(2) * 5
    R = np.eye(1) * 2
    ctrl1 = build_controller(plant, track={"x": 0.5})
    ctrl2 = build_controller(plant, track={"x": 0.5}, Q=Q, R=R)
    assert not np.allclose(ctrl1.K, ctrl2.K)


def test_nbar_warning_on_mismatched_tracking():
    """Warning issued when tracking more outputs than inputs and Nbar shape."""
    pend = InvertedPendulum()
    # two outputs tracked but only one input -> warning expected
    with pytest.warns(UserWarning, match="n_tracked != n_inputs"):
        ctrl = build_controller(pend, track={"x": 0.2, "phi": 0.0}, p=100)
    assert ctrl.Nbar is not None
    # shape of Nbar should be (1,2) since n_inputs=1, n_tracked=2
    assert ctrl.Nbar.shape == (1, 2)
