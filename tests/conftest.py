"""Shared test fixtures for control_toolbox unit tests."""

import numpy as np
import pytest

from control_toolbox import SimulationResult, SpringMass


@pytest.fixture
def spring_mass():
    """Return a default SpringMass plant."""
    return SpringMass()


@pytest.fixture
def simple_step_response():
    """Build a simple SimulationResult representing a unit step.

    Two samples: at t=0 and t=1, output goes from 0→1 with no overshoot or
    inputs constant zero. Used for metrics and plotting tests.
    """
    t = np.array([0.0, 1.0])
    y = np.array([[0.0, 1.0]])
    u = np.array([[0.0, 0.0]])
    return SimulationResult(
        t=t,
        y=y,
        u=u,
        output_labels=["y"],
        input_labels=["u"],
        tracked={"y": 1.0},
        dt=0.0,
    )
