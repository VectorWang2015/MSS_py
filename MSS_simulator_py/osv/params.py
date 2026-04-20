from dataclasses import dataclass, replace

import numpy as np

from .math_utils import rbody


@dataclass(frozen=True)
class OSVParams:
    l: float
    b: float
    t: float
    rho: float
    c_b: float
    s: float
    k_max: np.ndarray
    n_max: np.ndarray
    k_thr: np.ndarray
    l_x: np.ndarray
    l_y: np.ndarray
    thrust_max: float
    u_max: float
    nabla: float
    mass: float
    r_bg: np.ndarray
    g_matrix: np.ndarray
    mrb: np.ndarray
    ma: np.ndarray
    m: np.ndarray
    minv: np.ndarray
    d: np.ndarray


def load_osv_params() -> OSVParams:
    l = 83.0
    b = 18.0
    t = 5.0
    rho = 1025.0
    c_b = 0.65
    s = l * b + 2.0 * t * b

    k_max = np.array([300e3, 300e3, 420e3, 655e3], dtype=float)
    n_max = np.array([140.0, 140.0, 150.0, 200.0], dtype=float)
    k_thr = np.diag(k_max / (n_max**2))
    l_x = np.array([37.0, 35.0, -l / 2.0, -l / 2.0], dtype=float)
    l_y = np.array([0.0, 0.0, 7.0, -7.0], dtype=float)

    thrust_max = float(k_max[2] + k_max[3])
    u_max = 7.7

    nabla = c_b * l * b * t
    mass = rho * nabla
    r_bg = np.array([-4.5, 0.0, -1.2], dtype=float)

    c_w = 0.8
    awp = c_w * b * l
    kb = (1.0 / 3.0) * (5.0 * t / 2.0 - nabla / awp)
    k_munro_smith = (6.0 * c_w**3) / ((1.0 + c_w) * (1.0 + 2.0 * c_w))
    r_bb = np.array([-4.5, 0.0, t - kb], dtype=float)
    bg = r_bb[2] - r_bg[2]

    i_t = k_munro_smith * (b**3 * l) / 12.0
    i_l = 0.7 * (l**3 * b) / 12.0
    bm_t = i_t / nabla
    bm_l = i_l / nabla
    gm_t = bm_t - bg
    gm_l = bm_l - bg

    lcf = -0.5
    g33 = rho * 9.81 * awp
    g44 = rho * 9.81 * nabla * gm_t
    g55 = rho * 9.81 * nabla * gm_l
    g_cf = np.diag([0.0, 0.0, g33, g44, g55, 0.0])

    def hmtrx(r: np.ndarray) -> np.ndarray:
        from .math_utils import hmtrx as _hmtrx

        return _hmtrx(r)

    g_co = hmtrx(np.array([lcf, 0.0, 0.0])).T @ g_cf @ hmtrx(np.array([lcf, 0.0, 0.0]))
    g_matrix = (
        hmtrx(np.array([0.0, 0.0, 0.0])).T @ g_co @ hmtrx(np.array([0.0, 0.0, 0.0]))
    )

    r44 = 0.35 * b
    r55 = 0.25 * l
    r66 = 0.25 * l
    mrb, _ = rbody(mass, r44, r55, r66, np.zeros(3), r_bg)

    ma = 1e9 * np.array(
        [
            [0.0006, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0020, 0.0, 0.0031, 0.0, -0.0091],
            [0.0, 0.0, 0.0083, 0.0, 0.0907, 0.0],
            [0.0, 0.0031, 0.0, 0.0748, 0.0, -0.1127],
            [0.0, 0.0, 0.0907, 0.0, 3.9875, 0.0],
            [0.0, -0.0091, 0.0, -0.1127, 0.0, 1.2416],
        ],
        dtype=float,
    )

    m = mrb + ma
    minv = np.linalg.inv(m)

    t1 = 100.0
    t2 = 100.0
    t6 = 1.0
    zeta4 = 0.15
    zeta5 = 0.3
    w3 = np.sqrt(g_matrix[2, 2] / m[2, 2])
    w4 = np.sqrt(g_matrix[3, 3] / m[3, 3])
    w5 = np.sqrt(g_matrix[4, 4] / m[4, 4])
    zeta3 = 0.2
    d = np.diag(
        [
            m[0, 0] / t1,
            m[1, 1] / t2,
            m[2, 2] * 2.0 * zeta3 * w3,
            m[3, 3] * 2.0 * zeta4 * w4,
            m[4, 4] * 2.0 * zeta5 * w5,
            m[5, 5] / t6,
        ]
    )

    return OSVParams(
        l=l,
        b=b,
        t=t,
        rho=rho,
        c_b=c_b,
        s=s,
        k_max=k_max,
        n_max=n_max,
        k_thr=k_thr,
        l_x=l_x,
        l_y=l_y,
        thrust_max=thrust_max,
        u_max=u_max,
        nabla=nabla,
        mass=mass,
        r_bg=r_bg,
        g_matrix=g_matrix,
        mrb=mrb,
        ma=ma,
        m=m,
        minv=minv,
        d=d,
    )


def load_osv_custom_params() -> OSVParams:
    """Load custom OSV parameters for interactive demo use.

    Customization rule:
    - Keep all baseline parameters unchanged except stern thruster limits.
    - Force both stern thrusters (n3, n4) to use the current right-side config.
      That means n3/n4 share n_max and k_max from the baseline n4 channel.
    """

    base = load_osv_params()
    k_max = base.k_max.copy()
    n_max = base.n_max.copy()

    k_max[2] = k_max[3]
    n_max[2] = n_max[3]

    k_thr = np.diag(k_max / (n_max**2))
    thrust_max = float(k_max[2] + k_max[3])

    return replace(
        base,
        k_max=k_max,
        n_max=n_max,
        k_thr=k_thr,
        thrust_max=thrust_max,
    )
