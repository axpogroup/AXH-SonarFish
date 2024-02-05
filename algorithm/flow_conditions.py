from math import atan, cos, sin

import numpy as np
import pandas as pd

def rot_mat_from_river_velocity(conf: dict) -> np.ndarray:
    vx = conf["river_pixel_velocity"][0]
    vy = conf["river_pixel_velocity"][1]
    
    theta = - (np.pi * 1.5 - atan(vx / vy))
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def rotate_velocity_vectors(velocities_df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    rot = rot_mat_from_river_velocity(conf)
    return pd.DataFrame(np.dot(rot, velocities_df.T).T, columns=["v_xr", "v_yr"])