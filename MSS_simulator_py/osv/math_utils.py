import numpy as np


def smtrx(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )


def hmtrx(r: np.ndarray) -> np.ndarray:
    s = smtrx(r)
    top = np.hstack((np.eye(3), s.T))
    bottom = np.hstack((np.zeros((3, 3)), np.eye(3)))
    return np.vstack((top, bottom))


def rzyx(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [
                cpsi * cth,
                -spsi * cphi + cpsi * sth * sphi,
                spsi * sphi + cpsi * cphi * sth,
            ],
            [
                spsi * cth,
                cpsi * cphi + sphi * sth * spsi,
                -cpsi * sphi + sth * spsi * cphi,
            ],
            [-sth, cth * sphi, cth * cphi],
        ]
    )


def tzyx(phi: float, theta: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    eps = 1e-9
    if abs(cth) < eps:
        cth = eps if cth >= 0 else -eps
    return np.array(
        [
            [1.0, sphi * sth / cth, cphi * sth / cth],
            [0.0, cphi, -sphi],
            [0.0, sphi / cth, cphi / cth],
        ]
    )


def euler_jacobian(phi: float, theta: float, psi: float) -> np.ndarray:
    r = rzyx(phi, theta, psi)
    t = tzyx(phi, theta)
    top = np.hstack((r, np.zeros((3, 3))))
    bottom = np.hstack((np.zeros((3, 3)), t))
    return np.vstack((top, bottom))


def m2c(m: np.ndarray, nu: np.ndarray) -> np.ndarray:
    m = 0.5 * (m + m.T)
    nu = np.asarray(nu, dtype=float).reshape(-1)
    if nu.size == 6:
        m11 = m[0:3, 0:3]
        m12 = m[0:3, 3:6]
        m21 = m12.T
        m22 = m[3:6, 3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        nu1_dot = m11 @ nu1 + m12 @ nu2
        nu2_dot = m21 @ nu1 + m22 @ nu2

        top = np.hstack((np.zeros((3, 3)), -smtrx(nu1_dot)))
        bottom = np.hstack((-smtrx(nu1_dot), -smtrx(nu2_dot)))
        return np.vstack((top, bottom))

    if nu.size == 3:
        return np.array(
            [
                [0.0, 0.0, -m[1, 1] * nu[1] - m[1, 2] * nu[2]],
                [0.0, 0.0, m[0, 0] * nu[0]],
                [m[1, 1] * nu[1] + m[1, 2] * nu[2], -m[0, 0] * nu[0], 0.0],
            ]
        )

    raise ValueError("nu must be length 3 or 6")


def rbody(
    mass: float, r44: float, r55: float, r66: float, nu2: np.ndarray, r_bg: np.ndarray
):
    i3 = np.eye(3)
    o3 = np.zeros((3, 3))
    i_g = mass * np.diag([r44**2, r55**2, r66**2])

    mrb_cg = np.block([[mass * i3, o3], [o3, i_g]])
    crb_cg = np.block([[mass * smtrx(nu2), o3], [o3, -smtrx(i_g @ nu2)]])

    h = hmtrx(r_bg)
    mrb = h.T @ mrb_cg @ h
    crb = h.T @ crb_cg @ h
    return mrb, crb


def hoerner_coeff(b: float, t: float) -> float:
    table = np.array(
        [
            [0.0108623, 1.96608],
            [0.176606, 1.96573],
            [0.353025, 1.89756],
            [0.451863, 1.78718],
            [0.472838, 1.58374],
            [0.492877, 1.27862],
            [0.493252, 1.21082],
            [0.558473, 1.08356],
            [0.646401, 0.998631],
            [0.833589, 0.87959],
            [0.988002, 0.828415],
            [1.30807, 0.759941],
            [1.63918, 0.691442],
            [1.85998, 0.657076],
            [2.31288, 0.630693],
            [2.59998, 0.596186],
            [3.00877, 0.586846],
            [3.45075, 0.585909],
            [3.7379, 0.559877],
            [4.00309, 0.559315],
        ]
    )
    ratio = b / (2.0 * t)
    if ratio <= table[-1, 0]:
        return float(np.interp(ratio, table[:, 0], table[:, 1]))
    return float(table[-1, 1])


def crossflow_drag(l: float, b: float, t: float, nu_r: np.ndarray) -> np.ndarray:
    rho = 1025.0
    dx = l / 20.0
    cd2d = hoerner_coeff(b, t)

    y_h = 0.0
    z_h = 0.0
    m_h = 0.0
    n_h = 0.0

    v_r = float(nu_r[1])
    w_r = float(nu_r[2])
    q = float(nu_r[4])
    r = float(nu_r[5])

    x_samples = np.arange(-l / 2.0, l / 2.0 + 0.5 * dx, dx)
    for x_l in x_samples:
        u_h = abs(v_r + x_l * r) * (v_r + x_l * r)
        u_v = abs(w_r + x_l * q) * (w_r + x_l * q)
        y_h += -0.5 * rho * t * cd2d * u_h * dx
        z_h += -0.5 * rho * t * cd2d * u_v * dx
        m_h += -0.5 * rho * t * cd2d * x_l * u_v * dx
        n_h += -0.5 * rho * t * cd2d * x_l * u_h * dx

    return np.array([0.0, y_h, z_h, 0.0, m_h, n_h])


def added_mass_surge(mass: float, l: float, rho: float = 1025.0) -> float:
    nabla = mass / rho
    return 2.7 * rho * nabla ** (5.0 / 3.0) / l**2


def force_surge_damping(
    u_r: float,
    mass: float,
    wetted_surface: float,
    l: float,
    t1: float,
    rho: float,
    u_max: float,
    thrust_max: float,
):
    del wetted_surface
    x_udot = -added_mass_surge(mass, l, rho)
    x_u = -(mass - x_udot) / t1
    x_uu = -thrust_max / (u_max**2)
    u_cross = 2.0
    sigma = 1.0 - np.tanh(u_r / u_cross)
    x = sigma * x_u * u_r + (1.0 - sigma) * x_uu * abs(u_r) * u_r
    return x, x_uu, x_u


def thr_config(alpha: np.ndarray, l_x: np.ndarray, l_y: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    l_x = np.asarray(l_x, dtype=float).reshape(-1)
    l_y = np.asarray(l_y, dtype=float).reshape(-1)
    t_thr = np.zeros((3, 4))
    t_thr[:, 0] = np.array([0.0, 1.0, l_x[0]])
    t_thr[:, 1] = np.array([0.0, 1.0, l_x[1]])
    a1 = alpha[0]
    a2 = alpha[1]
    t_thr[:, 2] = np.array(
        [np.cos(a1), np.sin(a1), l_x[2] * np.sin(a1) - l_y[2] * np.cos(a1)]
    )
    t_thr[:, 3] = np.array(
        [np.cos(a2), np.sin(a2), l_x[3] * np.sin(a2) - l_y[3] * np.cos(a2)]
    )
    return t_thr
