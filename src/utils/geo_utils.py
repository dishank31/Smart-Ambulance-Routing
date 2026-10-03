"""Geospatial utilities used by feature engineering and routing."""

from __future__ import annotations

from math import asin, atan2, cos, degrees, radians, sin, sqrt

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    x = sin(dlon) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360


def bearing_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def get_bounding_box(center_lat: float, center_lon: float, radius_km: float) -> tuple[float, float, float, float]:
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / max(111.320 * cos(radians(center_lat)), 0.1)
    return (
        center_lat - lat_delta,
        center_lat + lat_delta,
        center_lon - lon_delta,
        center_lon + lon_delta,
    )


def manhattan_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    north_south = haversine_distance(lat1, lon1, lat2, lon1)
    east_west = haversine_distance(lat2, lon1, lat2, lon2)
    return north_south + east_west


def point_in_radius(point_lat: float, point_lon: float, center_lat: float, center_lon: float, radius_km: float) -> bool:
    return haversine_distance(point_lat, point_lon, center_lat, center_lon) <= radius_km


def filter_hospitals_by_radius(hospitals_df: pd.DataFrame, emergency_lat: float, emergency_lon: float, radius_km: float) -> pd.DataFrame:
    if hospitals_df.empty:
        return hospitals_df.copy()
    frame = hospitals_df.copy()
    frame["distance_km"] = haversine_vectorized(
        np.full(len(frame), emergency_lat),
        np.full(len(frame), emergency_lon),
        frame["latitude"].to_numpy(),
        frame["longitude"].to_numpy(),
    )
    return frame.loc[frame["distance_km"] <= radius_km].sort_values("distance_km").reset_index(drop=True)
