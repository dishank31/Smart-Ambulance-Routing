"""
Geospatial utility functions for the Smart Ambulance ML system.
Haversine distance, bearing calculation, manhattan distance.
"""

import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of point 1 (degrees)
        lat2, lon2: Coordinates of point 2 (degrees)
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine for pandas Series / numpy arrays.
    
    Args:
        lat1, lon1, lat2, lon2: Arrays of coordinates (degrees)
    
    Returns:
        Array of distances in kilometers
    """
    R = 6371
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Coordinates of point 1 (degrees)
        lat2, lon2: Coordinates of point 2 (degrees)
    
    Returns:
        Bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    bearing = atan2(x, y)
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing


def bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized bearing calculation for pandas/numpy."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Manhattan (taxicab) distance — useful for grid-based cities.
    
    Args:
        lat1, lon1, lat2, lon2: Coordinates (degrees)
    
    Returns:
        Approximate distance in kilometers
    """
    KM_PER_DEGREE = 111.0  # approximate
    return (abs(lat2 - lat1) + abs(lon2 - lon1)) * KM_PER_DEGREE
