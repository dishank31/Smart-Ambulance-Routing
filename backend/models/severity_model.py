import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("datasets/processed/severity_cleaned.csv")
print("Dataset loaded:", df.shape)

print("\nColumns:")
print(df.columns)

# =========================
# TARGET (UPDATE IF NEEDED)
# =========================
target = 'esi'  # change if different

# =========================
# FEATURES
# =========================
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

print("\nUsing features:", features)

# =========================
# HANDLE CLASS IMBALANCE
# =========================
classes = np.unique(y)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y
)

class_weights = dict(zip(classes, weights))
print("\nClass Weights:", class_weights)

# =========================
# ENCODE CATEGORICAL DATA
# =========================
X = pd.get_dummies(X, drop_first=True)

print("After encoding:", X.shape)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("\nModel trained")

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/severity/rf_severity_v1.joblib")

print("\nModel saved in models/severity/")