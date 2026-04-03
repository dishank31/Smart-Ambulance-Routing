import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

feats = joblib.load('models/eta/eta_features.joblib')
scaler = joblib.load('models/eta/scaler_eta.joblib')
model = joblib.load('models/eta/eta_stacking_v1.joblib')

lines = []
lines.append(f"Features ({len(feats)}): {feats}")

# Test with different distances
for dist in [1.0, 5.0, 10.0, 20.0, 30.0]:
    X = np.zeros((1, len(feats)))
    X[0, feats.index('distance_km')] = dist
    X[0, feats.index('hour')] = 12
    X[0, feats.index('month')] = 4
    X[0, feats.index('day_of_week')] = 2
    X[0, feats.index('pickup_latitude')] = 40.71
    X[0, feats.index('pickup_longitude')] = -74.0
    X[0, feats.index('dropoff_latitude')] = 40.71 + dist * 0.01
    X[0, feats.index('dropoff_longitude')] = -74.0 + dist * 0.01
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    lines.append(f"  distance={dist:5.1f} km -> ETA = {pred:.2f} min")

# Test bed models too
xgb = joblib.load('models/bed_xgboost.pkl')
lgb_model = joblib.load('models/bed_lightgbm.pkl')

lines.append("")
lines.append("Bed predictions (XGB + LGB avg):")
test_cases = [
    [25, 0, 8, 70, 110, 36.5],
    [25, 0, 8, 130, 180, 39.0],
    [70, 1, 20, 130, 180, 39.0],
    [70, 1, 3,  65, 100, 37.0],
]
for case in test_cases:
    X = np.array([case])
    xp = xgb.predict(X)[0]
    lp = lgb_model.predict(X)[0]
    avg = (xp + lp) / 2
    lines.append(f"  age={case[0]:3d} hr={case[3]:3d} sbp={case[4]:3d} temp={case[5]:.1f} -> beds={avg:.1f} (xgb={xp:.1f} lgb={lp:.1f})")

with open('test_output.txt', 'w') as f:
    f.write('\n'.join(lines))
print("Done. See test_output.txt")
