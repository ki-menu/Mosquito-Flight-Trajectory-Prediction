"""
Data augmentation functions for mosquito 3D flight trajectory.

Each function takes a numpy array of shape (T, 3) representing (x, y, z)
coordinates over T timesteps, and returns an augmented array of the same shape.

Usage example:
    import numpy as np
    import pandas as pd
    from augmentation import translate_last_to_origin

    df = pd.read_csv("data/train/TRAIN_00001.csv")
    coords = df[["x", "y", "z"]].values  # shape (T, 3)

    augmented = translate_last_to_origin(coords)
"""

import numpy as np


def translate_last_to_origin(coords: np.ndarray) -> np.ndarray:
    """Translate trajectory so that the last coordinate becomes (0, 0, 0).

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].

    Returns:
        Translated array of the same shape.
    """
    offset = coords[-1].copy()
    return coords - offset


def reverse_trajectory(coords: np.ndarray) -> np.ndarray:
    """Augment by reversing the time order of the trajectory.

    Simulates a mosquito travelling the same path in the opposite direction,
    i.e. from the original endpoint back to the original start point.

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].

    Returns:
        Time-reversed array of the same shape.
    """
    return coords[::-1].copy()


def get_rotation_matrix(coords: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix that aligns the last step direction with the x-axis.

    This is the matrix R used internally by normalize_rotation. Exposing it allows
    callers to apply the same rotation to paired data (e.g. target labels).

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].

    Returns:
        Rotation matrix of shape (3, 3). Returns identity if direction is degenerate.
    """
    direction = coords[-1] - coords[-2]
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)

    v = direction / norm
    target = np.array([1.0, 0.0, 0.0])

    axis = np.cross(v, target)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        R = np.eye(3) if np.dot(v, target) > 0 else np.diag([-1.0, -1.0, 1.0])
        return R.astype(np.float32)

    axis = axis / axis_norm
    cos_a = np.dot(v, target)
    sin_a = axis_norm

    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    return R.astype(np.float32)


def to_displacement_vectors(coords: np.ndarray) -> np.ndarray:
    """Convert a coordinate sequence to consecutive displacement vectors.

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].

    Returns:
        Array of shape (T-1, 3) where row t is coords[t+1] - coords[t].
    """
    return np.diff(coords, axis=0).astype(coords.dtype)


def normalize_rotation(coords: np.ndarray) -> np.ndarray:
    """Rotate trajectory so that the last step direction aligns with the x-axis.

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].

    Returns:
        Rotated array of the same shape.
    """
    return coords @ get_rotation_matrix(coords).T


def compute_velocity_acceleration(
    coords: np.ndarray, dt: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration features from coordinate trajectory.

    Uses forward differences. Note that differentiation reduces sequence length.

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].
        dt: Time step between consecutive samples in seconds (default 1.0).

    Returns:
        Tuple of (velocity, acceleration):
            - velocity: shape (T-1, 3)
            - acceleration: shape (T-2, 3)
    """
    velocity = np.diff(coords, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt

    return velocity, acceleration


def remove_speed_outliers(
    coords: np.ndarray, threshold: float = 3.0, dt: float = 1.0
) -> np.ndarray:
    """Replace positions where instantaneous speed exceeds threshold via linear interpolation.

    Speed is estimated as ||coords[t+1] - coords[t]|| / dt. Any point t+1 whose
    incoming speed exceeds the threshold is replaced by linearly interpolating between
    the nearest valid (non-outlier) neighbors. The first and last points are never
    replaced to preserve trajectory bounds.

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].
        threshold: Speed threshold in m/s; steps above this are replaced (default 3.0).
        dt: Time step between consecutive samples in seconds (default 1.0).

    Returns:
        Cleaned array of the same shape.
    """
    result = coords.copy()
    T = len(coords)

    speeds = np.linalg.norm(np.diff(coords, axis=0), axis=1) / dt  # shape (T-1,)

    outlier = np.zeros(T, dtype=bool)
    outlier[1:] = speeds > threshold
    outlier[0] = False
    outlier[-1] = False

    for idx in np.where(outlier)[0]:
        prev = idx - 1
        while prev > 0 and outlier[prev]:
            prev -= 1

        nxt = idx + 1
        while nxt < T - 1 and outlier[nxt]:
            nxt += 1

        alpha = (idx - prev) / (nxt - prev)
        result[idx] = coords[prev] * (1 - alpha) + coords[nxt] * alpha

    return result


def normalize_speed_scale(coords: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Normalize trajectory coordinates by the median instantaneous speed.

    Computes median speed across all timesteps as the characteristic speed scale,
    then divides all coordinates by that scale so trajectories of different absolute
    speeds become comparable.

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].
        dt: Time step between consecutive samples in seconds (default 1.0).

    Returns:
        Normalized array of the same shape (units: original_unit / (m/s)).
    """
    speeds = np.linalg.norm(np.diff(coords, axis=0), axis=1) / dt
    scale = np.median(speeds)
    if scale < 1e-8:
        return coords.copy()
    return coords / scale
