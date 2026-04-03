import numpy as np
import pandas as pd
from src.utils.geo_utils import haversine_distance

class HospitalService:
    def __init__(self):
        # Realistic NYC-area hospitals with varied capacities
        np.random.seed(42)
        lats = np.random.uniform(40.6, 40.9, size=10)
        lons = np.random.uniform(-74.05, -73.7, size=10)

        names = [
            "Mount Sinai Medical Center",
            "NYU Langone Health",
            "NewYork-Presbyterian Hospital",
            "Bellevue Hospital Center",
            "Lenox Hill Hospital",
            "St. Luke's Roosevelt Hospital",
            "Brooklyn Methodist Hospital",
            "Queens General Hospital",
            "Bronx-Lebanon Hospital",
            "Staten Island University Hospital",
        ]

        # Varied capacities per hospital
        icu_caps   = [45, 60, 55, 40, 30, 25, 35, 20, 28, 15]
        er_caps    = [100, 120, 110, 90, 70, 65, 80, 50, 60, 40]
        gen_caps   = [250, 300, 280, 220, 160, 140, 180, 100, 130, 80]

        records = []
        for i in range(10):
            records.append({
                'id': i + 1,
                'name': names[i],
                'latitude': lats[i],
                'longitude': lons[i],
                'icu_total': icu_caps[i],
                'emergency_total': er_caps[i],
                'general_total': gen_caps[i],
            })

        self.hospitals = pd.DataFrame(records)

    def get_nearby_hospitals(self, lat, lon, radius_km=50000.0):
        nearby = []
        for idx, h in self.hospitals.iterrows():
            dist = haversine_distance(lat, lon, h['latitude'], h['longitude'])
            if dist <= radius_km:
                h_dict = h.to_dict()
                h_dict['distance_km'] = dist
                nearby.append(h_dict)
        # Sort by distance so nearest hospitals come first
        nearby.sort(key=lambda x: x['distance_km'])
        return nearby
