from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..utils.logger import setup_logger


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.imputer = KNNImputer(n_neighbors=5)
        self.logger = setup_logger("preprocessor")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        before = int(frame.isna().sum().sum())
        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if numeric_cols:
            frame[numeric_cols] = self.imputer.fit_transform(frame[numeric_cols])
        for col in categorical_cols:
            mode = frame[col].mode(dropna=True)
            frame[col] = frame[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

        after = int(frame.isna().sum().sum())
        self.logger.info("Missing values handled: before=%s after=%s", before, after)
        return frame

    def remove_outliers(self, df: pd.DataFrame, columns: list[str], method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        if method != "iqr":
            raise ValueError("Only 'iqr' outlier removal is supported.")
        frame = df.copy()
        original_rows = len(frame)
        for col in columns:
            if col not in frame.columns:
                continue
            q1 = frame[col].quantile(0.25)
            q3 = frame[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            frame = frame.loc[frame[col].between(lower, upper)]
        self.logger.info("Outlier removal dropped %s rows", original_rows - len(frame))
        return frame.reset_index(drop=True)

    def encode_categorical(self, df: pd.DataFrame, columns: list[str], method: str = "label") -> pd.DataFrame:
        frame = df.copy()
        for col in columns:
            if col not in frame.columns:
                continue
            if method == "label":
                encoder = LabelEncoder()
                frame[col] = encoder.fit_transform(frame[col].astype(str))
                self.label_encoders[col] = encoder
            elif method == "onehot":
                dummies = pd.get_dummies(frame[col].astype(str), prefix=col, drop_first=False)
                frame = pd.concat([frame.drop(columns=[col]), dummies], axis=1)
            else:
                raise ValueError("method must be 'label' or 'onehot'")
        return frame

    def scale_features(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        frame = df.copy()
        valid = [col for col in columns if col in frame.columns]
        if valid:
            frame[valid] = self.scaler.fit_transform(frame[valid])
        return frame

    def add_cyclical_features(self, df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
        frame = df.copy()
        if col in frame.columns:
            frame[f"{col}_sin"] = np.sin(2 * np.pi * frame[col] / max_val)
            frame[f"{col}_cos"] = np.cos(2 * np.pi * frame[col] / max_val)
        return frame

    def handle_class_imbalance(self, X, y, method: str = "smote"):
        if method != "smote":
            return X, y
        before = pd.Series(y).value_counts().to_dict()
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        after = pd.Series(y_resampled).value_counts().to_dict()
        self.logger.info("Class imbalance handled with SMOTE: before=%s after=%s", before, after)
        return X_resampled, y_resampled

    def split_data(self, X, y, test_size: float = 0.15, val_size: float = 0.15, temporal: bool = False):
        if temporal:
            n = len(X)
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))
            return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]

        stratify_target = y if len(np.unique(y)) <= min(20, len(y)) else None
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_target)
        val_relative = val_size / (1 - test_size)
        stratify_temp = y_temp if len(np.unique(y_temp)) <= min(20, len(y_temp)) else None
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative, random_state=42, stratify=stratify_temp)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_preprocessor(self, filepath: str) -> None:
        joblib.dump({"scaler": self.scaler, "label_encoders": self.label_encoders, "imputer": self.imputer}, filepath)

    def load_preprocessor(self, filepath: str) -> "DataPreprocessor":
        payload = joblib.load(filepath)
        self.scaler = payload["scaler"]
        self.label_encoders = payload["label_encoders"]
        self.imputer = payload["imputer"]
        return self

    def save_scaler(self, path: str) -> None:
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        self.scaler = joblib.load(path)
