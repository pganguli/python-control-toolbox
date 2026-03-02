"""
LQR-based control system design and analysis.

Quick start:
    from controltoolbox import *

    plant = InvertedPendulum()
    # model 0.1 second actuator delay and discretize
    plant = add_delay(plant, T_input=0.1)
    plant = discretize(plant, dt=0.01)

    ctrl = build_controller(plant, track={'x': 0.5, 'phi': 0.0}, p=100)
    resp = simulate(ctrl, t_end=5.0)

    fig = plot_response(resp, metrics=compute_metrics(resp))
    fig.savefig('response.png')
"""

# Controller synthesis
from .controller import ControllerResult, build_controller

# Metrics
from .metrics import Metrics, compute_metrics

# Plotting
from .plot import plot_response

# Preprocessing
from .preprocess import add_delay, discretize

# Simulation
from .simulate import SimulationResult, simulate

# Plants
from .systems import (
    AircraftPitch,
    BallBeam,
    BusSuspension,
    CruiseControl,
    DCMotorPosition,
    DCMotorSpeed,
    F1TenthCar,
    InvertedPendulum,
    MagSuspendedBall,
    RLCCircuit,
    SpringMass,
)

__all__ = [
    # Plants
    "SpringMass",
    "RLCCircuit",
    "CruiseControl",
    "DCMotorSpeed",
    "DCMotorPosition",
    "InvertedPendulum",
    "BusSuspension",
    "F1TenthCar",
    "AircraftPitch",
    "BallBeam",
    "MagSuspendedBall",
    # Controller
    "build_controller",
    # Preprocessing
    "discretize",
    "add_delay",
    # Simulation
    "simulate",
    "SimulationResult",
    "ControllerResult",
    # Metrics
    "compute_metrics",
    "Metrics",
    # Plotting
    "plot_response",
]
