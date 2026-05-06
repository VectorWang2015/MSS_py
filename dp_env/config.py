from dataclasses import dataclass, field


@dataclass
class ShipConfig:
    control_point_x_m: float = 0.0
    control_point_y_m: float = 0.0
    distance_clip_ship_lengths: float = 1.5
    fixed_azimuth_deg: tuple[float, float] = (0.0, 0.0)
    current_speed: float = 0.0
    current_direction_deg: float = 0.0
    wind_speed: float = 0.0
    wind_direction_deg: float = 0.0
    u_clip_mps: float = 5.0
    v_clip_mps: float = 3.0
    r_clip_radps: float = 1.0
    du_clip_mps2: float = 1.0
    dv_clip_mps2: float = 1.0
    dr_clip_radps2: float = 1.0


@dataclass
class TaskConfig:
    target_north_m: float = 0.0
    target_east_m: float = 0.0
    target_yaw_deg: float = 45.0
    randomize_start: bool = True
    start_radius_min_m: float = 50.0
    start_radius_max_m: float = 150.0
    randomize_initial_yaw: bool = True
    initial_yaw_min_deg: float = -90.0
    initial_yaw_max_deg: float = 90.0
    start_north_m: float = 0.0
    start_east_m: float = 0.0
    start_yaw_deg: float = 0.0
    timeout_s: float = 1000.0
    control_dt: float = 1.0
    integration_substeps: int = 8


@dataclass
class EnvConfig:
    ship: ShipConfig = field(default_factory=ShipConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
