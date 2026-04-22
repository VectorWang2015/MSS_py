from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OSVWindSimpleParams:
    rho_air: float
    afw: float
    alw: float
    l_ref: float
    c_x: float
    c_y: float
    c_n: float


def load_osv_wind_simple_params() -> OSVWindSimpleParams:
    return OSVWindSimpleParams(
        rho_air=1.225,
        afw=180.0,
        alw=850.0,
        l_ref=83.0,
        c_x=0.9,
        c_y=0.85,
        c_n=0.10,
    )


def wind_velocity_ned(wind_speed: float, wind_direction: float) -> np.ndarray:
    return np.array(
        [
            wind_speed * np.cos(wind_direction),
            wind_speed * np.sin(wind_direction),
            0.0,
        ]
    )
