import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# LOAD DATA
df = pd.read_csv("datasets/processed/eta_processed.csv")
print("Dataset loaded:", df.shape)
print(df.columns)

# FEATURES (FIXED)
features = [
    'passenger_count',
    'pickup_latitude',
    'pickup_longitude',
    'dropoff_latitude',
    'dropoff_longitude',
    'hour',
    'day_of_week',
    'month',
    'distance_km'
]

# TARGET (FIXED)
target = 'duration_min'

X = df[features]
y = df[target]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model trained")

# EVALUATION
y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# SAVE MODEL (IMPORTANT PATH)
joblib.dump(model, "models/eta/rf_eta_v1.joblib")

print("Model saved in models/eta/")