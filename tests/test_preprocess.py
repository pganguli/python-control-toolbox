"""Tests covering preprocessing helpers (discretise/delay)."""

import pytest

from control_toolbox import SpringMass, add_delay, discretize


def test_discretize_preserves_labels_and_dt():
    """Discretizing a continuous plant keeps labels and sets dt."""
    plant = SpringMass()
    disc = discretize(plant, dt=0.2)
    assert disc.dt == pytest.approx(0.2)
    assert disc.input_labels == plant.input_labels
    assert disc.output_labels == plant.output_labels


def test_discretize_errors_on_discrete():
    """Attempting to discretize an already-discrete plant raises ValueError."""
    plant = SpringMass().sample(0.1)
    with pytest.raises(ValueError):
        discretize(plant, dt=0.1)


def test_add_delay_changes_states_and_preserves_labels():
    """Adding delay increases state dimension while keeping labels."""
    plant = SpringMass()
    orig_n = plant.A.shape[0]
    delayed = add_delay(plant, T_input=0.1, T_output=0.05, pade_order=2)
    assert delayed.A.shape[0] > orig_n
    assert delayed.input_labels == plant.input_labels
    assert delayed.output_labels == plant.output_labels


def test_add_delay_errors():
    """Invalid uses of add_delay raise ValueError."""
    plant = SpringMass().sample(0.1)
    with pytest.raises(ValueError):
        add_delay(plant, T_input=0.1)
    with pytest.raises(ValueError):
        add_delay(SpringMass(), T_input=-0.1)
    with pytest.raises(ValueError):
        add_delay(SpringMass(), T_output=-0.2)
    with pytest.raises(ValueError):
        add_delay(SpringMass(), T_input=0.1, pade_order=0)
