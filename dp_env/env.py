import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from MSS_simulator_py.osv.model import OSVDynamics, OSVEnvironment
from MSS_simulator_py.osv.params import load_osv_custom_params

from .config import EnvConfig
from .reward import compute_dp_reward


def wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def control_point_ned(state, ship_cfg):
    x = np.asarray(state, dtype=float).reshape(12)
    north, east = float(x[6]), float(x[7])
    psi = float(x[11])
    dx = ship_cfg.control_point_x_m
    dy = ship_cfg.control_point_y_m
    cp_north = north + dx * math.cos(psi) - dy * math.sin(psi)
    cp_east = east + dx * math.sin(psi) + dy * math.cos(psi)
    return cp_north, cp_east


def cg_from_control_point(cp_north, cp_east, psi, ship_cfg):
    dx = ship_cfg.control_point_x_m
    dy = ship_cfg.control_point_y_m
    north = cp_north - dx * math.cos(psi) + dy * math.sin(psi)
    east = cp_east - dx * math.sin(psi) - dy * math.cos(psi)
    return north, east


def compute_tracking_geometry(state, target_north_m, target_east_m, final_target_yaw_rad, ship_cfg):
    x = np.asarray(state, dtype=float).reshape(12)
    psi = float(x[11])
    cp_north, cp_east = control_point_ned(x, ship_cfg)
    dn = float(target_north_m - cp_north)
    de = float(target_east_m - cp_east)
    d_m = float(math.hypot(dn, de))
    bearing_ned = float(math.atan2(de, dn))
    eta = float(wrap_pi(bearing_ned - psi))
    dpsi = float(wrap_pi(final_target_yaw_rad - psi))

    ship_length = max(1e-6, float(getattr(ship_cfg, "ship_length_m", 0.0)) or 0.0)
    if ship_length <= 0.0:
        ship_length = 1.0
    d_clip = min(d_m, ship_cfg.distance_clip_ship_lengths * ship_length)
    d_norm_L = d_clip / ship_length

    return {
        "cp_north": cp_north,
        "cp_east": cp_east,
        "dn": dn,
        "de": de,
        "d_m": d_m,
        "d_norm_L": d_norm_L,
        "bearing_ned": bearing_ned,
        "eta": eta,
        "dpsi": dpsi,
    }


def build_observation(state, du, dv, dr, geometry, ship_cfg):
    x = np.asarray(state, dtype=float).reshape(12)
    u, v, r = float(x[0]), float(x[1]), float(x[5])
    obs = np.array(
        [
            float(geometry["d_norm_L"]),
            math.cos(float(geometry["dpsi"])),
            math.sin(float(geometry["dpsi"])),
            math.cos(float(geometry["eta"])),
            math.sin(float(geometry["eta"])),
            np.clip(u, -ship_cfg.u_clip_mps, ship_cfg.u_clip_mps),
            np.clip(v, -ship_cfg.v_clip_mps, ship_cfg.v_clip_mps),
            np.clip(r, -ship_cfg.r_clip_radps, ship_cfg.r_clip_radps),
            np.clip(du, -ship_cfg.du_clip_mps2, ship_cfg.du_clip_mps2),
            np.clip(dv, -ship_cfg.dv_clip_mps2, ship_cfg.dv_clip_mps2),
            np.clip(dr, -ship_cfg.dr_clip_radps2, ship_cfg.dr_clip_radps2),
        ],
        dtype=np.float32,
    )
    return obs


class VesselDPEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.ship_cfg = cfg.ship
        self.task_cfg = cfg.task

        params = load_osv_custom_params()
        self.params = params
        self.model = OSVDynamics(params=params)
        self.ship_length_m = float(params.l)
        self.ship_cfg.ship_length_m = float(params.l)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        o = self.ship_cfg
        obs_high = np.array(
            [
                float(self.ship_cfg.distance_clip_ship_lengths),
                1.0, 1.0,
                1.0, 1.0,
                o.u_clip_mps, o.v_clip_mps, o.r_clip_radps,
                o.du_clip_mps2, o.dv_clip_mps2, o.dr_clip_radps2,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.plant_env = OSVEnvironment(
            current_speed=self.ship_cfg.current_speed,
            current_direction=math.radians(self.ship_cfg.current_direction_deg),
            wind_speed=self.ship_cfg.wind_speed,
            wind_direction=math.radians(self.ship_cfg.wind_direction_deg),
        )

        self.state = np.zeros(12, dtype=float)
        self.applied_action = np.zeros(4, dtype=float)
        self.prev_applied_action = np.zeros(4, dtype=float)
        self.current_control = np.zeros(6, dtype=float)
        self.time_s = 0.0

        self.target_north_m = 0.0
        self.target_east_m = 0.0
        self.target_yaw_rad = 0.0

    def _rng_uniform(self, low, high):
        return float(self.np_random.uniform(low, high))

    def _sample_initial_yaw(self):
        t = self.task_cfg
        if t.randomize_initial_yaw:
            return math.radians(self._rng_uniform(t.initial_yaw_min_deg, t.initial_yaw_max_deg))
        return math.radians(t.start_yaw_deg)

    def _station_initial_state(self):
        t = self.task_cfg
        psi = self._sample_initial_yaw()
        if t.randomize_start:
            radius = self._rng_uniform(t.start_radius_min_m, t.start_radius_max_m)
            bearing = self._rng_uniform(-math.pi, math.pi)
            cp_n = t.target_north_m + radius * math.cos(bearing)
            cp_e = t.target_east_m + radius * math.sin(bearing)
            north, east = cg_from_control_point(cp_n, cp_e, psi, self.ship_cfg)
        else:
            north = float(t.start_north_m)
            east = float(t.start_east_m)
        x = np.zeros(12, dtype=float)
        x[6] = north
        x[7] = east
        x[11] = psi
        return x

    def _update_active_target(self):
        t = self.task_cfg
        self.target_north_m = float(t.target_north_m)
        self.target_east_m = float(t.target_east_m)
        self.target_yaw_rad = math.radians(float(t.target_yaw_deg))

    def _action_to_control(self, applied_action):
        rpm = np.asarray(applied_action, dtype=float).reshape(4) * self.params.n_max
        a1 = math.radians(self.ship_cfg.fixed_azimuth_deg[0])
        a2 = math.radians(self.ship_cfg.fixed_azimuth_deg[1])
        return np.array([rpm[0], rpm[1], rpm[2], rpm[3], a1, a2], dtype=float)

    def _integrate_state(self, control):
        substep_dt = float(self.task_cfg.control_dt) / int(self.task_cfg.integration_substeps)
        x = self.state.copy()
        for _ in range(int(self.task_cfg.integration_substeps)):
            x = self.model.step_rk4(x, control, substep_dt, self.plant_env)
            x[11] = float(wrap_pi(x[11]))
        return x

    def _planar_accels(self, state, control):
        xdot = self.model.derivatives(state, control, self.plant_env)
        return float(xdot[0]), float(xdot[1]), float(xdot[5])

    def _build_info(self, obs, geom, du, dv, dr, reward, reward_terms):
        return {
            "time_s": float(self.time_s),
            "obs": np.asarray(obs, dtype=float),
            "state": self.state.copy(),
            "north": float(self.state[6]),
            "east": float(self.state[7]),
            "psi_rad": float(self.state[11]),
            "psi_deg": math.degrees(float(self.state[11])),
            "u": float(self.state[0]),
            "v": float(self.state[1]),
            "r": float(self.state[5]),
            "du": float(du),
            "dv": float(dv),
            "dr": float(dr),
            "target_north_m": float(self.target_north_m),
            "target_east_m": float(self.target_east_m),
            "target_yaw_rad": float(self.target_yaw_rad),
            "target_yaw_deg": math.degrees(float(self.target_yaw_rad)),
            "distance_m": float(geom["d_m"]),
            "distance_ship_lengths": float(geom["d_norm_L"]),
            "eta_rad": float(geom["eta"]),
            "eta_deg": math.degrees(float(geom["eta"])),
            "dpsi_rad": float(geom["dpsi"]),
            "dpsi_deg": math.degrees(float(geom["dpsi"])),
            "applied_action": self.applied_action.copy(),
            "current_control": self.current_control.copy(),
            "reward": float(reward),
            "reward_terms": dict(reward_terms),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time_s = 0.0
        self.applied_action.fill(0.0)
        self.prev_applied_action.fill(0.0)

        self.state = self._station_initial_state()
        self._update_active_target()

        self.current_control = self._action_to_control(self.applied_action)
        du, dv, dr = self._planar_accels(self.state, self.current_control)
        geom = compute_tracking_geometry(
            self.state,
            self.target_north_m,
            self.target_east_m,
            self.target_yaw_rad,
            self.ship_cfg,
        )
        obs = build_observation(self.state, du, dv, dr, geom, self.ship_cfg)
        info = self._build_info(obs, geom, du, dv, dr, 0.0, {})
        return obs, info

    def step(self, action):
        self.prev_applied_action = self.applied_action.copy()
        self.applied_action = np.clip(np.asarray(action, dtype=float).reshape(4), -1.0, 1.0)
        self.current_control = self._action_to_control(self.applied_action)

        self.state = self._integrate_state(self.current_control)
        self.time_s += float(self.task_cfg.control_dt)

        du, dv, dr = self._planar_accels(self.state, self.current_control)

        geom = compute_tracking_geometry(
            self.state,
            self.target_north_m,
            self.target_east_m,
            self.target_yaw_rad,
            self.ship_cfg,
        )

        truncated = self.time_s >= self.task_cfg.timeout_s

        reward, terms = compute_dp_reward(
            d_norm_L=float(geom["d_norm_L"]),
            dpsi=float(geom["dpsi"]),
            eta=float(geom["eta"]),
            u=float(self.state[0]),
            v=float(self.state[1]),
            r=float(self.state[5]),
            du=du, dv=dv, dr=dr,
            applied_action=self.applied_action.copy(),
            prev_applied_action=self.prev_applied_action.copy(),
            dt=float(self.task_cfg.control_dt),
            timeout=truncated,
        )

        obs = build_observation(self.state, du, dv, dr, geom, self.ship_cfg)
        info = self._build_info(obs, geom, du, dv, dr, reward, terms)
        return obs, float(reward), False, truncated, info
