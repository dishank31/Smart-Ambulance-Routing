import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = KNNImputer(n_neighbors=5)
        
    def handle_missing_values(self, df):
        """Handle missing values using KNN for numerical and mode for categorical."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
        return df
    
    def remove_outliers(self, df, columns, threshold=1.5):
        """Remove outliers using IQR method for specific columns."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df
    
    def encode_categorical(self, df, columns, method='label'):
        """Encode categorical variables."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                if method == 'label':
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                elif method == 'onehot':
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
        return df

    def scale_features(self, df, columns):
        """Scale specific numeric features using standard scaler."""
        df = df.copy()
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            df[valid_cols] = self.scaler.fit_transform(df[valid_cols])
        return df

    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance using SMOTE."""
        if method == 'smote':
            smote = SMOTE(random_state=42, k_neighbors=3)
            return smote.fit_transform(X, y)
        return X, y

    def split_data(self, X, y, test_size=0.15, val_size=0.15, temporal=False):
        """Split data into train, val and test sets."""
        if temporal:
            n = len(X)
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            val_relative = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative, random_state=42, stratify=y_temp
            )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)
