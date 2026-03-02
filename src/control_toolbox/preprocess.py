"""
preprocess.py — Plant preprocessing: discretisation and input delay.

Typical usage:
    plant = InvertedPendulum()
    # model 0.1 s input and 0.05 s sensor delays using Padé approximants
    plant = add_delay(plant, T_input=0.1, T_output=0.05, pade_order=3)
    plant = discretize(plant, dt=0.01)
    # synthesize controller; ``ctrl`` is a ControllerResult dataclass
    ctrl = build_controller(plant, ...)
    K, Nbar, sys_cl = ctrl.K, ctrl.Nbar, ctrl.sys_cl
"""
# pylint: disable=invalid-name

import numpy as np
from control import StateSpace, c2d, pade, series, ss, tf


def discretize(plant: StateSpace, dt: float) -> StateSpace:
    """Convert a continuous-time plant to discrete time via ZOH.

    Wraps control.c2d and preserves input/output labels.

    Args:
        plant: Continuous-time StateSpace plant.
        dt:    Sampling period (seconds).

    Returns:
        Discrete-time StateSpace with the same labels as the input plant.

    Raises:
        ValueError: If the plant is already discrete.
    """
    if plant.dt not in (0, None):
        raise ValueError(
            f"Plant is already discrete (dt={plant.dt}). Cannot discretize again."
        )

    disc = c2d(plant, dt, method="zoh")
    disc.input_labels = list(plant.input_labels)
    disc.output_labels = list(plant.output_labels)
    return disc


def add_delay(
    plant: StateSpace,
    T_input: float = 0.0,
    T_output: float = 0.0,
    pade_order: int = 2,
) -> StateSpace:
    """Augment a continuous-time plant with input and/or output delays.

    Uses Padé approximations (via ``control.pade``) to model each delay as a
    rational transfer function, then series-connects it with the plant:

    - **Input delay**: ``delay_in --> plant``
    - **Output delay**: ``plant --> delay_out``
    - **Both**:        ``delay_in --> plant --> delay_out``

    The Padé approximant of order ``n`` approximates ``e^{-Ts}`` as a ratio
    of polynomials in ``s``. Higher orders give better frequency-domain
    accuracy at the cost of added states (each order adds 1 state per
    delay channel per input/output).

    This function intentionally operates on a **continuous-time** plant and
    should be called **before** ``discretize``. Applying Padé approximations
    after discretization is not meaningful, since ``e^{-Ts}`` is already
    exact in discrete time via a shift register.

    Args:
        plant:       Continuous-time StateSpace plant.
        T_input:     Input delay in seconds (≥ 0). Models actuator/transport
                     lag: u(t) reaches the plant as u(t - T_input).
        T_output:    Output delay in seconds (≥ 0). Models sensor lag:
                     the controller observes y(t - T_output).
        pade_order:  Order of the Padé approximation (≥ 1). Order 2–4 is
                     typically sufficient; higher orders can be ill-conditioned.

    Returns:
        Continuous-time StateSpace with delays absorbed into the dynamics.
        Ready to be passed to ``discretize``.

    Raises:
        ValueError: If the plant is discrete, any delay is negative, or
                    ``pade_order`` is less than 1.

    Example:
        >>> plant = InvertedPendulum()                         # continuous SS
        >>> plant = add_delay(plant, T_input=0.01, pade_order=3)
        >>> plant = discretize(plant, dt=0.01)
    """
    if plant.dt not in (0, None):
        raise ValueError(
            "add_delay expects a continuous-time plant. "
            "Call add_delay before discretize(plant, dt)."
        )
    if T_input < 0:
        raise ValueError(f"T_input must be >= 0, got {T_input}.")
    if T_output < 0:
        raise ValueError(f"T_output must be >= 0, got {T_output}.")
    if pade_order < 1:
        raise ValueError(f"pade_order must be >= 1, got {pade_order}.")
    if T_input == 0.0 and T_output == 0.0:
        return plant

    def _pade_ss(T: float, n_channels: int) -> StateSpace:
        """Build a diagonal MIMO Padé delay block for ``n_channels`` channels."""
        num, den = pade(T, pade_order)
        delay_tf = tf(num, den)
        delay_ss = ss(delay_tf)
        # Replicate as a diagonal MIMO system (one independent delay per channel).
        if n_channels == 1:
            return delay_ss
        A_blk = np.zeros((delay_ss.A.shape[0] * n_channels,) * 2)
        B_blk = np.zeros((A_blk.shape[0], n_channels))
        C_blk = np.zeros((n_channels, A_blk.shape[0]))
        D_blk = np.zeros((n_channels, n_channels))
        ns = delay_ss.A.shape[0]
        for i in range(n_channels):
            r, c = slice(i * ns, (i + 1) * ns), slice(i * ns, (i + 1) * ns)
            A_blk[r, c] = delay_ss.A
            B_blk[r, i] = delay_ss.B.ravel()
            C_blk[i, c] = delay_ss.C.ravel()
            D_blk[i, i] = delay_ss.D[0, 0]
        return ss(A_blk, B_blk, C_blk, D_blk)

    result = plant

    if T_input > 0.0:
        delay_in = _pade_ss(T_input, plant.B.shape[1])
        result = series(delay_in, result)

    if T_output > 0.0:
        delay_out = _pade_ss(T_output, plant.C.shape[0])
        result = series(result, delay_out)

    result.input_labels = list(plant.input_labels)
    result.output_labels = list(plant.output_labels)
    return result
