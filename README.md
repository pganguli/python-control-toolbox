# control-toolbox

`control-toolbox` is a small Python library for LQR-based
state-space controller design, simulation, preprocessing, and
visualisation. The package wraps `python-control` models with
convenience utilities for tracking references, handling delays, and
plotting closed-loop responses.

## Features

- Predefined plant models (inverted pendulum, mass‑spring, motor,
  etc.) inheriting from `control.StateSpace`.
- `build_controller` — synthesise LQR/DLQR state-feedback plus pre-
  compensator (`Nbar`) for reference tracking.  Works with continuous
  and discrete plants, and accounts for input delays.
- `simulate` — run a closed-loop simulation from a `ControllerResult`.
- `add_delay` / `discretize` utilities for preprocessing plants.
- Plotting helpers and simple performance metrics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .  # installs this package in editable mode
```

## Quick start

```python
from control_toolbox import *

plant = InvertedPendulum()
plant = add_delay(plant, T_input=0.1, T_output=0.05, pade_order=3)
plant = discretize(plant, dt=0.01)

ctrl = build_controller(plant, track={"x":0.5, "phi":0.0}, p=100)
resp = simulate(ctrl, t_end=5.0)

fig = plot_response(resp, metrics=compute_metrics(resp))
fig.savefig("response.png")
```
