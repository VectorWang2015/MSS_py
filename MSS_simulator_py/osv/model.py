from dataclasses import dataclass

import numpy as np

from .math_utils import (
    crossflow_drag,
    euler_jacobian,
    force_surge_damping,
    m2c,
    rbody,
    rzyx,
    smtrx,
    thr_config,
)
from .params import OSVParams, load_osv_params


@dataclass
class OSVEnvironment:
    current_speed: float = 0.0
    current_direction: float = 0.0
    tau_env_ned: np.ndarray | None = None

    def disturbance_ned(self) -> np.ndarray:
        if self.tau_env_ned is None:
            return np.zeros(6)
        tau = np.asarray(self.tau_env_ned, dtype=float).reshape(6)
        return tau


class OSVDynamics:
    def __init__(self, params: OSVParams | None = None):
        self.params = params if params is not None else load_osv_params()

    def _tau_env_body(self, eta: np.ndarray, env: OSVEnvironment) -> np.ndarray:
        tau_ned = env.disturbance_ned()
        phi, theta, psi = float(eta[3]), float(eta[4]), float(eta[5])
        r_nb = rzyx(phi, theta, psi)
        f_body = r_nb.T @ tau_ned[0:3]
        m_body = r_nb.T @ tau_ned[3:6]
        return np.concatenate((f_body, m_body))

    def _tau_thr(self, control: np.ndarray) -> np.ndarray:
        p = self.params
        control = np.asarray(control, dtype=float).reshape(6)
        n_rpm = control[0:4]
        alpha = control[4:6]

        t_thr = thr_config(alpha, p.l_x, p.l_y)
        u_thr = np.abs(n_rpm) * n_rpm
        tau_3dof = t_thr @ p.k_thr @ u_thr
        return np.array([tau_3dof[0], tau_3dof[1], 0.0, 0.0, 0.0, tau_3dof[2]])

    def derivatives(
        self, state: np.ndarray, control: np.ndarray, env: OSVEnvironment | None = None
    ) -> np.ndarray:
        p = self.params
        x = np.asarray(state, dtype=float).reshape(12)
        env = env if env is not None else OSVEnvironment()

        nu = x[0:6]
        eta = x[6:12]

        psi = float(eta[5])
        v_c = np.array(
            [
                env.current_speed * np.cos(env.current_direction - psi),
                env.current_speed * np.sin(env.current_direction - psi),
                0.0,
            ]
        )
        nu_c = np.array([v_c[0], v_c[1], 0.0, 0.0, 0.0, 0.0])
        nu_c_dot = np.concatenate((-smtrx(nu[3:6]) @ v_c, np.zeros(3)))
        nu_r = nu - nu_c

        _, crb = rbody(
            p.mass,
            0.35 * p.b,
            0.25 * p.l,
            0.25 * p.l,
            nu[3:6],
            p.r_bg,
        )
        ca = m2c(p.ma, nu)

        x_surge, _, _ = force_surge_damping(
            nu_r[0],
            p.mass,
            p.s,
            p.l,
            100.0,
            p.rho,
            p.u_max,
            p.thrust_max,
        )
        tau_drag = np.array([x_surge, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_cross = crossflow_drag(p.l, p.b, p.t, nu_r)
        tau_thr = self._tau_thr(control)
        tau_env_body = self._tau_env_body(eta, env)

        d_eff = p.d.copy()
        d_eff[0, 0] = 0.0

        phi, theta = float(eta[3]), float(eta[4])
        j = euler_jacobian(phi, theta, psi)
        eta_dot = j @ nu
        nu_dot = nu_c_dot + p.minv @ (
            tau_thr
            + tau_drag
            + tau_cross
            + tau_env_body
            - (crb + ca + d_eff) @ nu_r
            - p.g_matrix @ eta
        )

        return np.concatenate((nu_dot, eta_dot))

    def step_rk4(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
        env: OSVEnvironment | None = None,
    ) -> np.ndarray:
        x = np.asarray(state, dtype=float).reshape(12)
        k1 = self.derivatives(x, control, env)
        k2 = self.derivatives(x + 0.5 * dt * k1, control, env)
        k3 = self.derivatives(x + 0.5 * dt * k2, control, env)
        k4 = self.derivatives(x + dt * k3, control, env)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
