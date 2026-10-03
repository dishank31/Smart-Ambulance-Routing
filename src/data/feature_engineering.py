from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.geo_utils import bearing_vectorized, haversine_distance as base_haversine_distance
from ..utils.geo_utils import haversine_vectorized, manhattan_distance as base_manhattan_distance


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return base_haversine_distance(lat1, lon1, lat2, lon2)


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(bearing_vectorized(np.array([lat1]), np.array([lon1]), np.array([lat2]), np.array([lon2]))[0])


def manhattan_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return base_manhattan_distance(lat1, lon1, lat2, lon2)


def _add_cyclical_columns(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / period)
    df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / period)
    return df


def create_temporal_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    frame = df.copy()
    frame[datetime_col] = pd.to_datetime(frame[datetime_col])
    frame["hour"] = frame[datetime_col].dt.hour
    frame["day_of_week"] = frame[datetime_col].dt.dayofweek
    frame["month"] = frame[datetime_col].dt.month
    frame["year"] = frame[datetime_col].dt.year
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["is_rush_hour"] = (
        frame["hour"].between(7, 9, inclusive="both")
        | frame["hour"].between(17, 19, inclusive="both")
    ).astype(int)
    frame = _add_cyclical_columns(frame, "hour", 24)
    frame = _add_cyclical_columns(frame, "day_of_week", 7)
    frame = _add_cyclical_columns(frame, "month", 12)
    return frame


def create_geospatial_features(df: pd.DataFrame, pickup_lat_col: str, pickup_lon_col: str, dropoff_lat_col: str, dropoff_lon_col: str) -> pd.DataFrame:
    frame = df.copy()
    frame["distance_km"] = haversine_vectorized(
        frame[pickup_lat_col].to_numpy(),
        frame[pickup_lon_col].to_numpy(),
        frame[dropoff_lat_col].to_numpy(),
        frame[dropoff_lon_col].to_numpy(),
    )
    frame["bearing"] = bearing_vectorized(
        frame[pickup_lat_col].to_numpy(),
        frame[pickup_lon_col].to_numpy(),
        frame[dropoff_lat_col].to_numpy(),
        frame[dropoff_lon_col].to_numpy(),
    )
    frame["manhattan_dist"] = [
        base_manhattan_distance(a, b, c, d)
        for a, b, c, d in zip(
            frame[pickup_lat_col],
            frame[pickup_lon_col],
            frame[dropoff_lat_col],
            frame[dropoff_lon_col],
        )
    ]
    return frame


def create_interaction_features(df: pd.DataFrame, feature_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    frame = df.copy()
    for left, right in feature_pairs:
        if left in frame.columns and right in frame.columns:
            frame[f"{left}_x_{right}"] = frame[left] * frame[right]
    return frame


def create_rolling_features(df: pd.DataFrame, group_col: str, value_col: str, windows: list[int] | tuple[int, ...] = (6, 24)) -> pd.DataFrame:
    frame = df.copy()
    if "timestamp" in frame.columns:
        frame = frame.sort_values([group_col, "timestamp"]).reset_index(drop=True)
    for window in windows:
        rolling = frame.groupby(group_col)[value_col].rolling(window=window, min_periods=1)
        frame[f"{value_col}_rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        frame[f"{value_col}_rolling_sum_{window}"] = rolling.sum().reset_index(level=0, drop=True)
        frame[f"{value_col}_rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
    return frame
