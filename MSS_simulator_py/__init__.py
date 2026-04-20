"""MSS Simulator Python package.

This package implements a Python plant model aligned with the MSS
`VESSELS/models/osv.m` equations, using the same physical and hydrodynamic
parameters. It intentionally excludes DP controller and control allocation.

State vector `x` (12,), units and suggested operating ranges:
    x[0]  = u      [m/s]     surge velocity, typical [-8, 8]
    x[1]  = v      [m/s]     sway velocity, typical [-4, 4]
    x[2]  = w      [m/s]     heave velocity, typical [-2, 2]
    x[3]  = p      [rad/s]   roll rate, typical [-0.5, 0.5]
    x[4]  = q      [rad/s]   pitch rate, typical [-0.5, 0.5]
    x[5]  = r      [rad/s]   yaw rate, typical [-0.5, 0.5]
    x[6]  = north  [m]       NED north position
    x[7]  = east   [m]       NED east position
    x[8]  = down   [m]       NED down position
    x[9]  = phi    [rad]     roll angle, typical [-20 deg, 20 deg]
    x[10] = theta  [rad]     pitch angle, typical [-15 deg, 15 deg]
    x[11] = psi    [rad]     yaw angle, wrapped as needed

Control vector `u` (6,), RPM input only in stage 1:
    u[0] = n1      [rpm]     bow tunnel thruster 1, typical [-140, 140]
    u[1] = n2      [rpm]     bow tunnel thruster 2, typical [-140, 140]
    u[2] = n3      [rpm]     stern azimuth thruster 1, typical [-150, 150]
    u[3] = n4      [rpm]     stern azimuth thruster 2, typical [-200, 200]
    u[4] = alpha1  [rad]     azimuth angle 1, typical [-60 deg, 60 deg]
    u[5] = alpha2  [rad]     azimuth angle 2, typical [-60 deg, 60 deg]

Environment inputs:
    current_speed: float [m/s]
        Horizontal current speed magnitude `Vc`.
    current_direction: float [rad]
        Current direction `betaVc` in NED frame.
    tau_env_ned: array-like (6,) [N, N, N, Nm, Nm, Nm]
        External generalized disturbance in NED frame:
        [X_n, Y_e, Z_d, K_n, M_e, N_d].
        Internally rotated to BODY frame before being added to dynamics.

Main API:
    - OSVDynamics.derivatives(x, u, env): returns xdot
    - OSVDynamics.step_rk4(x, u, dt, env): returns one-step propagated state

Package layout:
    - MSS_simulator_py.osv: OSV dynamics submodule
    - demo/: interactive pygame demo (kept outside module package)

Notes:
    - This implementation follows `osv.m` modeling choices and constants.
    - Numerical trajectories may differ from MATLAB/Octave due to platform and
      solver details.
"""

from .osv.model import OSVDynamics, OSVEnvironment

__all__ = ["OSVDynamics", "OSVEnvironment"]
