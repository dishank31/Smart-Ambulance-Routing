import numpy as np
import pandas as pd
from ..utils.geo_utils import haversine_vectorized, bearing_vectorized, manhattan_distance

def add_cyclical_features(df, col, max_val):
    """Add sin and cos features to capture cyclical nature of time variables."""
    df = df.copy()
    if col in df.columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def create_eta_features(df):
    """Generate richer features for ETA predictions."""
    df = df.copy()
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['month'] = df['pickup_datetime'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    if set(['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']).issubset(df.columns):
        if 'distance_km' not in df.columns:
            df['distance_km'] = haversine_vectorized(
                df['pickup_latitude'], df['pickup_longitude'],
                df['dropoff_latitude'], df['dropoff_longitude']
            )
        df['bearing'] = bearing_vectorized(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        df['manhattan_dist'] = manhattan_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )

    if 'hour' in df.columns:
        df = add_cyclical_features(df, 'hour', 24)
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                              ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)

    if 'day_of_week' in df.columns:
        df = add_cyclical_features(df, 'day_of_week', 7)

    if 'trip_duration' in df.columns and 'duration_minutes' not in df.columns:
        df['duration_minutes'] = df['trip_duration'] / 60.0

    return df

def create_bed_features(df):
    """Generate richer features for Bed availability predictions."""
    df = df.copy()
    
    if 'hour' in df.columns:
        df = add_cyclical_features(df, 'hour', 24)
    if 'day_of_week' in df.columns:
        df = add_cyclical_features(df, 'day_of_week', 7)
    
    # Department encoding
    if 'department' in df.columns and not any(c.startswith('dept_') for c in df.columns):
        df = pd.get_dummies(df, columns=['department'], prefix='dept')
        
    return df

def create_severity_features(df):
    """Generate features suitable for severity models."""
    df = df.copy()
    
    # Optional vitals combinations or interactions could be placed here
    
    return df
