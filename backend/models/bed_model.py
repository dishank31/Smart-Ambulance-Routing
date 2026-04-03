import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("datasets/processed/bed_availability_processed.csv")
print("Dataset loaded:", df.shape)

print("\nColumns:")
print(df.columns)

# =========================
# TARGET
# =========================
target = 'available_beds'

# =========================
# FEATURES
# =========================
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# =========================
# ENCODE CATEGORICAL
# =========================
X = pd.get_dummies(X, drop_first=True)

print("After encoding:", X.shape)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("\nModel trained")

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMAE:", mae)
print("RMSE:", rmse)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/bed_availability/rf_beds_v1.joblib")

print("\nModel saved in models/bed_availability/")