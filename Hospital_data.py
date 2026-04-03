import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_bed_availability_data(n_days=365, n_hospitals=10):
    """
    Generate realistic hospital bed availability data
    Based on published patterns:
    - ICU occupancy: 70-95% (higher at night, winter)
    - Emergency: fluctuates with time of day
    - General: more predictable
    """
    records = []
    
    hospitals = [
        {'id': i, 'name': f'Hospital_{i}', 
         'icu_total': np.random.choice([20, 30, 40, 50]),
         'emergency_total': np.random.choice([30, 50, 70]),
         'general_total': np.random.choice([100, 150, 200, 300])}
        for i in range(n_hospitals)
    ]
    
    start_date = datetime(2023, 1, 1)
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        month = current_date.month
        day_of_week = current_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Seasonal factor (higher occupancy in winter)
        season_factor = 1.0 + 0.15 * np.cos(2 * np.pi * (month - 1) / 12)
        
        for hour in range(24):
            for hospital in hospitals:
                # Time-of-day pattern
                # Peak admissions: 10AM-2PM and 6PM-10PM
                hour_factor = (
                    0.7 + 0.3 * np.sin(np.pi * (hour - 6) / 12) 
                    if 6 <= hour <= 22 else 0.5
                )
                
                for dept, total_key in [('ICU', 'icu_total'), 
                                         ('Emergency', 'emergency_total'),
                                         ('General', 'general_total')]:
                    total_beds = hospital[total_key]
                    
                    # Base occupancy rate by department
                    if dept == 'ICU':
                        base_occupancy = 0.82  # ICUs run at ~80-85%
                        noise_std = 0.05
                    elif dept == 'Emergency':
                        base_occupancy = 0.65
                        noise_std = 0.15  # More variable
                    else:
                        base_occupancy = 0.75
                        noise_std = 0.08
                    
                    # Calculate occupied beds
                    occupancy_rate = (
                        base_occupancy * season_factor * hour_factor 
                        + np.random.normal(0, noise_std)
                    )
                    occupancy_rate = np.clip(occupancy_rate, 0.1, 0.98)
                    
                    occupied = int(total_beds * occupancy_rate)
                    available = total_beds - occupied
                    
                    # Recent admissions/discharges (correlated with time)
                    admissions_1h = max(0, int(np.random.poisson(
                        2 if dept == 'Emergency' else 1) * hour_factor))
                    discharges_1h = max(0, int(np.random.poisson(
                        1.5 if 8 <= hour <= 16 else 0.3)))
                    
                    records.append({
                        'timestamp': current_date + timedelta(hours=hour),
                        'hospital_id': hospital['id'],
                        'hospital_name': hospital['name'],
                        'department': dept,
                        'total_beds': total_beds,
                        'occupied_beds': occupied,
                        'available_beds': available,
                        'occupancy_rate': occupancy_rate,
                        'admissions_last_1h': admissions_1h,
                        'discharges_last_1h': discharges_1h,
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'month': month,
                        'is_weekend': is_weekend,
                        'is_holiday': 1 if (month == 12 and current_date.day in [25, 31]) else 0,
                        'season': ['Winter','Winter','Spring','Spring','Spring',
                                  'Summer','Summer','Summer','Fall','Fall','Fall','Winter'][month-1]
                    })
    
    df = pd.DataFrame(records)
    
    # Add rolling features
    for hospital_id in df['hospital_id'].unique():
        for dept in df['department'].unique():
            mask = (df['hospital_id'] == hospital_id) & (df['department'] == dept)
            idx = df[mask].index
            df.loc[idx, 'admissions_rolling_6h'] = (
                df.loc[idx, 'admissions_last_1h'].rolling(6, min_periods=1).sum()
            )
            df.loc[idx, 'discharges_rolling_6h'] = (
                df.loc[idx, 'discharges_last_1h'].rolling(6, min_periods=1).sum()
            )
            df.loc[idx, 'occupancy_rolling_avg_24h'] = (
                df.loc[idx, 'occupancy_rate'].rolling(24, min_periods=1).mean()
            )
    
    return df

# Generate
bed_data = generate_bed_availability_data(n_days=365, n_hospitals=10)
import os
output_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'bed_availability')
os.makedirs(output_dir, exist_ok=True)
bed_data.to_csv(os.path.join(output_dir, 'hospital_beds.csv'), index=False)
print(f"Generated {len(bed_data)} bed availability records")
print(bed_data.head())