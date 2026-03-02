"""
Collection of StateSpace plant models used by controltoolbox.

Each class subclasses :class:`control.StateSpace` and provides a
self-documenting constructor with physical parameters and
extensive docstrings describing inputs, outputs, states, and control
objectives.

Typical usage:
    from controltoolbox import SpringMass, build_controller, simulate

    plant = SpringMass()
    ctrl = build_controller(plant, track={'x': 1.0}, p=10)
    resp = simulate(ctrl, t_end=5.0)
"""
# pylint: disable=invalid-name,too-many-arguments,too-many-positional-arguments

import math

from control import StateSpace


class SpringMass(StateSpace):
    """Mass-Spring-Damper System

    A mass m is acted on by an external force F, a spring with constant k that
    restores the mass toward x = 0, and a viscous damper with constant b that
    opposes motion. Applying Newton's second law gives the governing equation:

        m*ẍ = F - b*ẋ - k*x

    This is a second-order system. The two state variables capture the energy
    stored in the system: x captures potential energy in the spring and v
    captures kinetic energy of the mass. The damper dissipates energy but
    stores none, so it does not contribute a state.

    Args:
        m: mass (kg)
        k: spring constant (N/m)
        b: damping constant (Ns/m)

    Input:  F (force)
    Output: x (position)
    States: x (position), v (velocity)

    Objective: Track a commanded reference position x_ref (reference tracking).

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=SystemModeling#5
    """

    def __init__(self, m: float = 1.0, k: float = 1.0, b: float = 0.2):
        self.m = m
        self.k = k
        self.b = b
        A = [[0.0, 1.0], [-k / m, -b / m]]
        B = [[0.0], [1 / m]]
        C = [[1.0, 0.0]]
        D = 0.0
        super().__init__(A, B, C, D, inputs=["F"], outputs=["x"])


class RLCCircuit(StateSpace):
    """RLC Circuit

    A series combination of a resistor R, inductor L, and capacitor C driven
    by a voltage source V. Applying Kirchhoff's voltage law around the single
    loop gives the governing equation:

        V(t) - R*i - L*di/dt - q/C = 0

    This is structurally analogous to the mass-spring-damper system, with
    charge q corresponding to displacement, inductance L to mass, resistance R
    to damping, and inverse capacitance 1/C to spring stiffness. The two state
    variables are the charge q on the capacitor (potential energy) and the
    current i through the inductor (magnetic energy).

    Args:
        R: electric resistance (Ohm)
        L: electric inductance (H)
        C: electric capacitance (F)

    Input:  V (voltage)
    Output: i (current)
    States: q (charge), i (current)

    Objective: Track a desired current setpoint i_ref (reference tracking).

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=SystemModeling#15
    """

    def __init__(self, R: float = 10.0, L: float = 100.0, C: float = 0.01):
        self.R = R
        self.L = L
        self.C = C
        A = [[0.0, 1.0], [-1 / (L * C), -R / L]]
        B = [[0.0], [1 / L]]
        Cm = [[0.0, 1.0]]
        D = 0.0
        super().__init__(A, B, Cm, D, inputs=["V"], outputs=["i"])


class CruiseControl(StateSpace):
    """Cruise Control

    A vehicle modelled as a single mass m subject to a driving force F and a
    velocity-proportional drag force b*v. Applying Newton's second law gives:

        m*v̇ = F - b*v

    This is a first-order system — only kinetic energy (velocity) is stored.
    Position is not a state because the control objective concerns only speed.
    The damping term b captures rolling resistance and aerodynamic drag lumped
    into a single linear coefficient.

    Args:
        m: vehicle mass (kg)
        b: damping constant (N.s/m)

    Input:  F (force)
    Output: v (velocity)
    States: v (velocity)

    Objective: Track a desired cruising speed v_ref (reference tracking) with
               rise time < 5 s, overshoot < 10%, and steady-state error < 2%.
               A reference scaling factor (Nbar) is needed to eliminate
               steady-state error when using pole-placement state feedback.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=CruiseControl&section=SystemModeling
        https://ctms.engin.umich.edu/CTMS/index.php?example=CruiseControl&section=ControlStateSpace
    """

    def __init__(self, m: float = 1000.0, b: float = 50.0):
        self.m = m
        self.b = b
        A = [[-b / m]]
        B = [[1 / m]]
        C = [[1.0]]
        D = 0.0
        super().__init__(A, B, C, D, inputs=["F"], outputs=["v"])


class DCMotorSpeed(StateSpace):
    """DC Motor Speed (2-state)

    An armature-controlled DC motor modelled for shaft *speed* control.
    Compare with DCMotorPosition, which is a 3-state model that includes shaft
    angle as a state and uses different (smaller) default parameters from
    Carnegie Mellon's controls lab. Here position is not a state because the
    control objective concerns only angular velocity. The magnetic field is
    assumed constant; motor torque is proportional to armature current i via
    Kt, and back-EMF is proportional to angular velocity ω via Kb. In SI
    units Kt = Kb. Applying Newton's second law (mechanical) and Kirchhoff's
    voltage law (electrical) gives:

        J*ω̇ = Kt*i - b*ω          (mechanical)
        L*di/dt = V - R*i - Kb*ω   (electrical)

    Eliminating position as a state reduces this to a second-order system
    whose two states capture kinetic energy (ω) and magnetic energy (i).

    Args:
        J:  moment of inertia of the rotor (kg.m^2)
        b:  motor viscous friction constant (N.m.s)
        Kb: electromotive force constant (V/rad/sec)
        Kt: motor torque constant (N.m/A)
        R:  electric resistance (Ohm)
        L:  electric inductance (H)

    Input:  V (voltage)
    Output: ω (angular velocity)
    States: ω (angular velocity), i (current)

    Objective: For a 1 rad/s step in desired angular velocity ω_ref, track the
               commanded speed with settling time under 2 s, overshoot under
               5%, and steady-state error under 1% (reference tracking).
               Full-state feedback via pole placement (poles at -5±i; ζ=0.98,
               σ=5, giving ~0.8 s settling) meets the transient requirements.
               A precompensator Nbar (computed via rscale) is required to
               eliminate the steady-state DC offset that state feedback alone
               cannot correct; integral control can be used as a robust
               alternative when model uncertainty or disturbances are present.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling
        https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=ControlStateSpace
    """

    def __init__(
        self,
        J: float = 0.01,
        b: float = 0.1,
        Kb: float = 0.01,
        Kt: float = 0.01,
        R: float = 1.0,
        L: float = 0.5,
    ):
        self.J = J
        self.b = b
        self.Kb = Kb
        self.Kt = Kt
        self.R = R
        self.L = L
        A = [[-b / J, Kt / J], [-Kb / L, -R / L]]
        B = [[0.0], [1 / L]]
        C = [[1.0, 0.0]]
        D = 0.0
        super().__init__(A, B, C, D, inputs=["V"], outputs=["omega"])


class DCMotorPosition(StateSpace):
    """DC Motor Position (3-state)

    An armature-controlled DC motor modelled for shaft *position* control.
    Compare with DCMotorSpeed, which is a 2-state reduction that drops the
    position state and uses different (larger) default parameters suited to
    speed control. Here the magnetic field is assumed constant, so motor
    torque T is proportional only to armature current i via torque constant
    Kt. The back-EMF e is proportional to shaft angular velocity via constant
    Kb. In SI units Kt = Kb. The coupled governing equations from Newton's
    second law (mechanical) and Kirchhoff's voltage law (electrical) are:

        J*θ̈ + b*θ̇ = Kt*i          (mechanical)
        L*di/dt + R*i = V - Kb*θ̇   (electrical)

    This produces a third-order system. The rotor and shaft are assumed rigid
    and friction is modelled as purely viscous.

    Default parameter values were derived experimentally from a motor in
    Carnegie Mellon's undergraduate controls lab.

    Args:
        J:  moment of inertia of the rotor (kg.m^2)
        b:  motor viscous friction constant (N.m.s)
        Kb: electromotive force constant (V/rad/sec)
        Kt: motor torque constant (N.m/A)
        R:  electric resistance (Ohm)
        L:  electric inductance (H)

    Input:  V (voltage)
    Output: θ (angle)
    States: θ (angle), ω (angular velocity), i (current)

    Objective: For a 1-radian step in desired shaft angle θ_ref, track the
               commanded position with settling time under 0.040 s, overshoot
               under 16%, and zero steady-state error even in the presence of
               a step load-torque disturbance (reference tracking + disturbance
               rejection). Full-state feedback via pole placement (dominant
               poles at -100±100i, fast pole at -200; ζ=0.5, σ=100) meets the
               transient requirements. Integral action must be added by
               augmenting the state with w = ∫(θ - θ_ref) dt and placing the
               integrator pole at -300 to eliminate steady-state error under
               step disturbances.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=SystemModeling
        https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition&section=ControlStateSpace
    """

    def __init__(
        self,
        J: float = 3.2284e-6,
        b: float = 3.5077e-6,
        Kb: float = 0.0274,
        Kt: float = 0.0274,
        R: float = 4.0,
        L: float = 2.75e-6,
    ):
        self.J = J
        self.b = b
        self.Kb = Kb
        self.Kt = Kt
        self.R = R
        self.L = L
        A = [[0.0, 1.0, 0.0], [0.0, -b / J, Kt / J], [0.0, -Kb / L, -R / L]]
        B = [[0.0], [0.0], [1 / L]]
        C = [[1.0, 0.0, 0.0]]
        D = 0.0
        super().__init__(A, B, C, D, inputs=["V"], outputs=["theta"])


class InvertedPendulum(StateSpace):
    """Inverted Pendulum

    A pendulum mounted on a motorised cart that moves horizontally. The system
    is nonlinear and open-loop unstable — the pendulum will fall without
    active control. The nonlinear equations of motion are linearised about the
    vertically upward equilibrium θ = π using the small-angle approximations:

        cos(θ) ≈ -1,   sin(θ) ≈ -φ,   φ̇² ≈ 0

    where φ = θ - π is the deviation from the upright position. The linearised
    governing equations are:

        (I + m*l²)*φ̈ - m*g*l*φ = m*l*ẍ
        (M + m)*ẍ + b*ẋ - m*l*φ̈ = u

    The denominator p = I*(M+m) + M*m*l² appears in all A and B matrix entries
    and arises from eliminating the internal reaction forces between the cart
    and the pendulum.

    Args:
        M: mass of the cart (kg)
        m: mass of the pendulum (kg)
        b: coefficient of friction for cart (N/m/s)
        l: length to pendulum center of mass (m)
        I: moment of inertia of the pendulum (kg.m^2)
        g: acceleration due to gravity (m/s^2)

    Input:   F (force)
    Outputs: x (position), φ (angle)
    States:  x (position), v (velocity), φ (angle), ω (angular velocity)

    Objective: For a 0.2 m step in desired cart position x_ref, simultaneously
               drive x to x_ref (reference tracking) and stabilize φ to zero
               (keep the pendulum upright). The pendulum deflects transiently
               but must return to vertical, with settling time for both x and φ
               under 5 s, rise time for x under 0.5 s, |φ| never exceeding
               0.35 rad (20°) from vertical, and steady-state error under 2%
               for both outputs. Full-state feedback (LQR) with a
               precompensator Nbar is the recommended approach; a state
               estimator (observer) can be added when not all states are
               directly measured.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
        https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
    """

    def __init__(
        self,
        M: float = 0.5,
        m: float = 0.2,
        b: float = 0.1,
        l: float = 0.3,
        I: float = 0.006,
        g: float = 9.8,
    ):
        self.M = M
        self.m = m
        self.b = b
        self.l = l
        self.I = I
        self.g = g
        p = I * (M + m) + M * m * (l**2)
        A = [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -(I + m * l**2) * b / p, (m**2) * g * (l**2) / p, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -(m * l * b) / p, m * g * l * (M + m) / p, 0.0],
        ]
        B = [[0.0], [(I + m * l**2) / p], [0.0], [m * l / p]]
        C = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        D = [[0.0], [0.0]]
        super().__init__(A, B, C, D, inputs=["F"], outputs=["x", "phi"])


class BusSuspension(StateSpace):
    """Bus Suspension System (1/4 Bus Model)

    A quarter-car active suspension model with two masses: the bus body M1
    (sprung mass) connected via suspension spring K1 and damper b1 to the
    suspension mass M2 (unsprung mass), which in turn sits on the tyre spring
    K2 and damper b2 above the road surface W. An actuator generates control
    force U between the two masses. Applying Newton's second law to each mass
    gives the governing equations:

        M1*X1'' = -b1*(X1'-X2') - K1*(X1-X2) + U
        M2*X2'' =  b1*(X1'-X2') + K1*(X1-X2) + b2*(W'-X2') + K2*(W-X2) - U

    The suspension deflection Y1 = X1 - X2 is chosen as a state directly,
    reducing the system to four states. The road displacement W enters as a
    direct input (second column of B).

    Args:
        M1: 1/4 bus body mass (kg)
        M2: suspension mass (kg)
        K1: spring constant of suspension system (N/m)
        K2: spring constant of wheel and tire (N/m)
        b1: damping constant of suspension system (N.s/m)
        b2: damping constant of wheel and tire (N.s/m)

    Inputs:  U (control force), W (road displacement)
    Output:  Y1 = X1 - X2 (suspension deflection)
    States:  X1 (body position), X1' (body velocity),
             Y1 = X1-X2 (suspension deflection), Y1' (deflection rate)

    Objective: Regulate Y1 to zero, i.e. reject road disturbances so the bus
               body remains level despite bumps and surface irregularities
               (disturbance rejection). The design target is settling time
               under 5 s and overshoot under 5% for a 0.1 m step road input.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=Suspension&section=SystemModeling
        https://ctms.engin.umich.edu/CTMS/index.php?example=Suspension&section=ControlStateSpace
    """

    def __init__(
        self,
        M1: float = 2500.0,
        M2: float = 320.0,
        K1: float = 80000.0,
        K2: float = 500000.0,
        b1: float = 350.0,
        b2: float = 15020.0,
    ):
        self.M1 = M1
        self.M2 = M2
        self.K1 = K1
        self.K2 = K2
        self.b1 = b1
        self.b2 = b2
        A = [
            [0.0, 1.0, 0.0, 0.0],
            [
                -(b1 * b2) / (M1 * M2),
                0.0,
                (b1 / M1) * ((b1 / M1) + (b1 / M2) + (b2 / M2)) - (K1 / M1),
                -(b1 / M1),
            ],
            [b2 / M2, 0.0, -((b1 / M1) + (b1 / M2) + (b2 / M2)), 1.0],
            [K2 / M2, 0.0, -((K1 / M1) + (K1 / M2) + (K2 / M2)), 0.0],
        ]
        B = [
            [0.0, 0.0],
            [1.0 / M1, (b1 * b2) / (M1 * M2)],
            [0.0, -(b2 / M2)],
            [(1 / M1) + (1 / M2), -(K2 / M2)],
        ]
        C = [[0.0, 0.0, 1.0, 0.0]]
        D = [[0.0, 0.0]]
        super().__init__(A, B, C, D, inputs=["U", "W"], outputs=["Y1"])


class F1TenthCar(StateSpace):
    """F1/10 Car

    A kinematic bicycle model linearised about straight-line driving at
    constant longitudinal velocity v. Under the small-angle approximation
    (sin(x) ≈ tan(x) ≈ x), the lateral and heading dynamics decouple from
    the longitudinal dynamics and reduce to:

        ẏ = v * θ
        θ̇ = (v / L) * δ

    where y is the lateral displacement from the path, θ is the heading
    angle, δ is the front steering angle (input), and L is the wheelbase.
    This model is valid only for small steering angles and small deviations
    from the reference path; it is a standard basis for linear path-following
    controllers such as LQR and MPC on the F1/10 autonomous racing platform.

    Args:
        L: Wheelbase (m)
        v: Longitudinal velocity (m/s)

    Input:   δ (steering angle)
    Outputs: y (y-coordinate), θ (heading angle)
    States:  y (y-coordinate), θ (heading angle)

    Objective: Track a reference path by regulating lateral error y and
               heading error θ to zero (regulation / error-state tracking).

    References:
        https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf
    """

    def __init__(self, L: float = 0.33, v: float = 6.5):
        self.L = L
        self.v = v
        A = [[0.0, v], [0.0, 0.0]]
        B = [[0.0], [v / L]]
        C = [[1.0, 0.0], [0.0, 1.0]]
        D = [[0.0], [0.0]]
        super().__init__(A, B, C, D, inputs=["delta"], outputs=["y", "theta"])


class AircraftPitch(StateSpace):
    """Aircraft Pitch

    A linearised longitudinal model of an aircraft in steady-cruise at
    constant altitude and velocity. The full six-DOF nonlinear equations of
    motion are decoupled into longitudinal and lateral dynamics; only the
    longitudinal (pitch) channel is modelled here. Under the assumption that
    airspeed is unaffected by pitch-angle changes, the three governing
    equations are:

        α̇ = Zalpha*α + Zq*q + Zdelta*δ
        q̇ = Malpha*α + Mq*q + Mdelta*δ
        θ̇ = Omega*q

    where α is the angle of attack, q is the pitch rate, θ is the pitch
    angle, and δ is the elevator deflection angle (input).

    The parameters are aerodynamic stability derivatives and a kinematic
    frequency, each composed of deeper physical quantities defined as follows:

        μ = ρSc̄ / (4m)          [mass ratio]
        Ω = 2U / c̄              [pitch-kinematic frequency, rad/s]
        σ = 1 / (1 + μC_L)      [lift correction factor]
        η = μσC_M               [pitch-moment scaling constant]

    where ρ is air density, S is wing platform area, c̄ is mean chord length,
    m is aircraft mass, U is equilibrium flight speed, and C_L, C_D, C_M,
    C_W are the aerodynamic coefficients of lift, drag, pitch moment, and
    weight respectively. The stability derivatives are then:

        Zalpha = μΩσ(-(C_L + C_D))
        Zq     = μΩσ / (μ - C_L)
        Zdelta = μΩσ C_L
        Malpha = (μΩ / 2i_yy) * (C_M - η(C_L + C_D))
        Mq     = (μΩ / 2i_yy) * (C_M + σC_M(1 - μC_L))
        Mdelta = (μΩ / 2i_yy) * (ηC_W sin γ)
        Omega  = Ω = 2U / c̄

    where i_yy is the normalised moment of inertia and γ is the flight path
    angle. Default values are taken from Boeing commercial-aircraft flight data.

    Args:
        Zalpha: angle-of-attack stability derivative in α̇ equation = μΩσ(-(C_L+C_D)) (1/s)
        Zq:     pitch-rate stability derivative in α̇ equation = μΩσ/(μ-C_L) (1/s)
        Zdelta: elevator control derivative in α̇ equation = μΩσC_L (1/s)
        Malpha: angle-of-attack stability derivative in q̇ equation
                = μΩ(C_M-η(C_L+C_D))/(2i_yy) (1/s²)
        Mq:     pitch-rate stability derivative in q̇ equation = μΩC_M(1+σ(1-μC_L))/(2i_yy) (1/s)
        Mdelta: elevator control derivative in q̇ equation = μΩηC_W sin(γ)/(2i_yy) (1/s²)
        Omega:  pitch-kinematic frequency Ω = 2U/c̄; maps q into θ̇ (rad/s)

    Input:  δ (elevator deflection angle, rad)
    Output: θ (pitch angle, rad)
    States: α (angle of attack, rad), q (pitch rate, rad/s),
            θ (pitch angle, rad)

    Objective: For a 0.2 radian (11°) step in desired pitch angle θ_des,
               track the commanded pitch with overshoot under 10%, rise time
               under 2 s, settling time under 10 s, and steady-state error
               under 2% (reference tracking). The system is fully state
               controllable (controllability matrix has rank 3). LQR with
               Q = p*C'*C (p = 50) and R = 1 yields K = [-0.6435, 169.695,
               7.0711] and meets transient requirements. A precompensator
               Nbar = 7.0711 (computed via rscale) eliminates steady-state
               error; alternatively, integral control can be added for
               robustness to model uncertainty and step disturbances.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=SystemModeling
        https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlStateSpace
    """

    def __init__(
        self,
        Zalpha: float = -0.313,
        Zq: float = 56.7,
        Zdelta: float = 0.232,
        Malpha: float = -0.0139,
        Mq: float = -0.426,
        Mdelta: float = 0.0203,
        Omega: float = 56.7,
    ):
        self.Zalpha = Zalpha
        self.Zq = Zq
        self.Zdelta = Zdelta
        self.Malpha = Malpha
        self.Mq = Mq
        self.Mdelta = Mdelta
        self.Omega = Omega
        A = [[Zalpha, Zq, 0.0], [Malpha, Mq, 0.0], [0.0, Omega, 0.0]]
        B = [[Zdelta], [Mdelta], [0.0]]
        C = [[0.0, 0.0, 1.0]]
        D = 0.0
        super().__init__(A, B, C, D, inputs=["delta_e"], outputs=["theta"])


class BallBeam(StateSpace):
    """Ball and Beam (Torque Control)

    A ball is placed on a beam where it is free to roll along the beam's
    length. A motor applies torque directly at the centre of the beam,
    controlling the beam angle α and hence the ball position r. The ball is
    assumed to roll without slipping and friction is negligible.

    Applying the Lagrangian equation of motion for the ball and linearising
    about the horizontal equilibrium α = 0 gives:

        (J/R² + m) * r̈ = -m*g*α

    Rearranging:

        r̈ = H * α,   where H = -m*g / (J/R² + m)

    The beam angle dynamics are driven directly by the control input u = α̈
    (angular acceleration of the beam, produced by the motor torque). Writing
    the four first-order state equations:

        ṙ  = ṙ
        r̈  = H * α
        α̇  = α̇
        α̈  = u

    This is an open-loop marginally stable system (double integrator in r),
    making it a challenging control problem.

    Args:
        m: mass of the ball (kg)
        R: radius of the ball (m)
        g: gravitational acceleration (m/s²)
        J: ball's moment of inertia (kg.m²)

    Input:  u (beam angular acceleration α̈, rad/s²)
    Output: r (ball position, m)
    States: r (ball position, m), ṙ (ball velocity, m/s),
            α (beam angle, rad), α̇ (beam angular velocity, rad/s)

    Objective: For a 0.25 m step in desired ball position r_ref, track the
               commanded position with settling time under 3 s and overshoot
               under 5% (reference tracking). Full-state feedback via pole
               placement at (-2±2i, -20, -80), with ζ=0.7 for the dominant
               pair, meets both criteria. A precompensator Nbar (computed via
               rscale) is required to eliminate steady-state error.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=BallBeam&section=SystemModeling
        https://ctms.engin.umich.edu/CTMS/index.php?example=BallBeam&section=ControlStateSpace
    """

    def __init__(
        self,
        m: float = 0.111,
        R: float = 0.015,
        g: float = 9.8,
        J: float = 9.99e-6,
    ):
        self.m = m
        self.R = R
        self.g = g
        self.J = J
        H = -m * g / (J / R**2 + m)
        A = [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, H, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        B = [[0.0], [0.0], [0.0], [1.0]]
        C = [[1.0, 0.0, 0.0, 0.0]]
        D = 0.0
        super().__init__(A, B, C, D, inputs=["alpha_ddot"], outputs=["r"])


class MagSuspendedBall(StateSpace):
    """Magnetically Suspended Ball

    A ferromagnetic ball of mass m is suspended beneath an electromagnet.
    Current i through the coil induces an upward magnetic force K*i²/h that
    balances gravity, holding the ball at height h. The governing nonlinear
    equations are:

        m*ḧ = m*g - K*i²/h      (Newton's second law)
        V    = L*di/dt + R*i     (Kirchhoff's voltage law)

    The system is linearised about the equilibrium point (h_eq, 0, i_eq),
    where the ball is suspended in mid-air and ḧ = 0, giving the equilibrium
    current:

        i_eq = sqrt(m*g*h_eq / K)

    Defining state deviations Δh = h - h_eq, Δḣ, Δi = i - i_eq, and input
    ΔV = V - V_eq, the linearised state equations are:

        d/dt [Δh ]   [   0       1          0      ] [Δh ]   [   0   ]
             [Δḣ ] = [ g/h_eq   0   -2K·i_eq/      ] [Δḣ ] + [   0   ] ΔV
             [Δi ]   [   0       0       -R/L       ] [Δi ]   [ 1/L   ]
                                          (m·h_eq)

    The open-loop system has one unstable pole at +√(g/h_eq), making active
    control essential — without feedback the ball will fall or crash into the
    magnet.

    Args:
        m:    mass of the ball (kg)
        K:    magnetic force coefficient; force = K*i²/h (N·m/A²)
        L:    inductance of the electromagnet coil (H)
        R:    resistance of the electromagnet coil (Ohm)
        g:    gravitational acceleration (m/s²)
        h_eq: equilibrium height of the ball below the magnet (m)

    Input:  ΔV (deviation of applied voltage from equilibrium, V)
    Output: Δh (deviation of ball height from equilibrium, m)
    States: Δh (height deviation, m), Δḣ (velocity deviation, m/s),
            Δi (current deviation, A)

    Objective: Stabilise the ball at h_eq and reject disturbances (regulation).
               The open-loop system is unstable (poles at ±√(g/h_eq), −R/L).
               Full-state feedback via pole placement at (−20±20i, −100),
               with ζ=0.7 for the dominant pair, achieves settling time under
               0.5 s and overshoot under 5%. A precompensator Nbar (computed
               via rscale) is needed to track non-zero reference steps; an
               observer (L-gain via duality/place on (A', C')) can estimate
               ḣ and i when only h is measured directly.

    References:
        https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlStateSpace
    """

    def __init__(
        self,
        m: float = 0.05,
        K: float = 0.0001,
        L: float = 0.01,
        R: float = 1.0,
        g: float = 9.8,
        h_eq: float = 0.01,
    ):
        self.m = m
        self.K = K
        self.L = L
        self.R = R
        self.g = g
        self.h_eq = h_eq
        i_eq = math.sqrt(m * g * h_eq / K)
        self.i_eq = i_eq
        A = [
            [0.0, 1.0, 0.0],
            [g / h_eq, 0.0, -2.0 * K * i_eq / (m * h_eq)],
            [0.0, 0.0, -R / L],
        ]
        B = [[0.0], [0.0], [1.0 / L]]
        C = [[1.0, 0.0, 0.0]]
        D = 0.0
        super().__init__(A, B, C, D, inputs=["dV"], outputs=["dh"])
