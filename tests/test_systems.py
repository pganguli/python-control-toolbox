"""Smoke tests for plant model classes."""

from control_toolbox import InvertedPendulum, SpringMass


def test_spring_mass_properties():
    """SpringMass state‑space dimensions and labels are correct."""
    plant = SpringMass()
    # 2 states, 1 input, 1 output
    assert plant.A.shape == (2, 2)
    assert plant.B.shape == (2, 1)
    assert plant.C.shape == (1, 2)
    assert plant.input_labels == ["F"]
    assert plant.output_labels == ["x"]


def test_inverted_pendulum_labels_and_dims():
    """InvertedPendulum dimensions and I/O labels are as documented."""
    plant = InvertedPendulum()
    assert plant.A.shape == (4, 4)
    assert plant.B.shape == (4, 1)
    assert plant.C.shape == (2, 4)
    assert plant.input_labels == ["F"]
    assert plant.output_labels == ["x", "phi"]
