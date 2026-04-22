from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DemoConfig:
    rpm_limits: np.ndarray = field(
        default_factory=lambda: np.array([140.0, 140.0, 150.0, 200.0])
    )
    rpm_step: float = 5.0
    azimuth_limit: float = np.deg2rad(60.0)
    azimuth_step: float = np.deg2rad(2.0)
    current_speed_step: float = 0.05
    current_dir_step: float = np.deg2rad(5.0)
    tau_force_step: float = 500.0
    tau_moment_step: float = 2_000.0


@dataclass(frozen=True)
class DemoControlState:
    rpm: np.ndarray = field(default_factory=lambda: np.zeros(4))
    azimuth: np.ndarray = field(default_factory=lambda: np.zeros(2))
    current_speed: float = 0.0
    current_direction: float = 0.0
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    tau_env_ned: np.ndarray = field(default_factory=lambda: np.zeros(6))
    paused: bool = False

    def to_control_vector(self, cfg: DemoConfig) -> np.ndarray:
        rpm = np.clip(self.rpm, -cfg.rpm_limits, cfg.rpm_limits)
        az = np.clip(self.azimuth, -cfg.azimuth_limit, cfg.azimuth_limit)
        return np.concatenate((rpm, az))


def _clip_state(s: DemoControlState, cfg: DemoConfig) -> DemoControlState:
    rpm = np.clip(s.rpm, -cfg.rpm_limits, cfg.rpm_limits)
    azimuth = np.clip(s.azimuth, -cfg.azimuth_limit, cfg.azimuth_limit)
    return DemoControlState(
        rpm=rpm,
        azimuth=azimuth,
        current_speed=max(0.0, s.current_speed),
        current_direction=s.current_direction,
        wind_speed=max(0.0, s.wind_speed),
        wind_direction=s.wind_direction,
        tau_env_ned=s.tau_env_ned,
        paused=s.paused,
    )


def apply_action(
    state: DemoControlState, cfg: DemoConfig, action: str
) -> DemoControlState:
    rpm = state.rpm.copy()
    az = state.azimuth.copy()
    tau = state.tau_env_ned.copy()
    current_speed = float(state.current_speed)
    current_dir = float(state.current_direction)
    wind_speed = float(state.wind_speed)
    wind_dir = float(state.wind_direction)
    paused = bool(state.paused)

    if action == "n1_up":
        rpm[0] += cfg.rpm_step
    elif action == "n1_down":
        rpm[0] -= cfg.rpm_step
    elif action == "n2_up":
        rpm[1] += cfg.rpm_step
    elif action == "n2_down":
        rpm[1] -= cfg.rpm_step
    elif action == "n3_up":
        rpm[2] += cfg.rpm_step
    elif action == "n3_down":
        rpm[2] -= cfg.rpm_step
    elif action == "n4_up":
        rpm[3] += cfg.rpm_step
    elif action == "n4_down":
        rpm[3] -= cfg.rpm_step
    elif action == "a1_left":
        az[0] -= cfg.azimuth_step
    elif action == "a1_right":
        az[0] += cfg.azimuth_step
    elif action == "a2_left":
        az[1] -= cfg.azimuth_step
    elif action == "a2_right":
        az[1] += cfg.azimuth_step
    elif action == "current_speed_up":
        current_speed += cfg.current_speed_step
    elif action == "current_speed_down":
        current_speed -= cfg.current_speed_step
    elif action == "current_dir_left":
        current_dir += cfg.current_dir_step
    elif action == "current_dir_right":
        current_dir -= cfg.current_dir_step
    elif action == "wind_speed_up":
        wind_speed += cfg.current_speed_step
    elif action == "wind_speed_down":
        wind_speed -= cfg.current_speed_step
    elif action == "wind_dir_plus":
        wind_dir += cfg.current_dir_step
    elif action == "wind_dir_minus":
        wind_dir -= cfg.current_dir_step
    elif action == "tau_x_plus":
        tau[0] += cfg.tau_force_step
    elif action == "tau_x_minus":
        tau[0] -= cfg.tau_force_step
    elif action == "tau_y_plus":
        tau[1] += cfg.tau_force_step
    elif action == "tau_y_minus":
        tau[1] -= cfg.tau_force_step
    elif action == "tau_n_plus":
        tau[5] += cfg.tau_moment_step
    elif action == "tau_n_minus":
        tau[5] -= cfg.tau_moment_step
    elif action == "toggle_pause":
        paused = not paused
    elif action == "zero_controls":
        rpm[:] = 0.0
        az[:] = 0.0
        tau[:] = 0.0
        current_speed = 0.0
        current_dir = 0.0
        wind_speed = 0.0
        wind_dir = 0.0

    return _clip_state(
        DemoControlState(
            rpm=rpm,
            azimuth=az,
            current_speed=current_speed,
            current_direction=current_dir,
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            tau_env_ned=tau,
            paused=paused,
        ),
        cfg,
    )
