"""
Model Registry — tracks model versions with metrics and metadata.
Stores versioning info in a JSON file for easy retrieval.
"""

import json
import os
import joblib
from datetime import datetime


class ModelRegistry:
    """JSON-based model version tracking."""
    
    def __init__(self, registry_path="models/model_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load existing registry or create new one."""
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                if "models" not in data:
                    data["models"] = []
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {"models": []}
    
    def register_model(self, model, model_name, metrics, version=None, save=True):
        """
        Register a trained model with metrics.
        
        Args:
            model: Trained model object
            model_name: Name like 'severity_rf', 'eta_stacking'
            metrics: Dict of performance metrics
            version: Version string (auto-incremented if None)
            save: Whether to save the model to disk
        
        Returns:
            Registry entry dict
        """
        if version is None:
            existing = [m for m in self.registry["models"] if m["name"] == model_name]
            version = f"v{len(existing) + 1}"
        
        # Determine save path
        category = model_name.split("_")[0]  # severity, eta, bed
        model_dir = os.path.join("models", category)
        os.makedirs(model_dir, exist_ok=True)
        
        filepath = os.path.join(model_dir, f"{model_name}_{version}.joblib")
        
        if save:
            joblib.dump(model, filepath)
        
        # Create entry
        entry = {
            "name": model_name,
            "version": version,
            "filepath": filepath,
            "metrics": {k: round(v, 6) if isinstance(v, float) else v
                       for k, v in metrics.items()},
            "created_at": datetime.now().isoformat(),
            "is_active": True,
        }
        
        # Deactivate previous versions of same model
        for m in self.registry["models"]:
            if m["name"] == model_name:
                m["is_active"] = False
        
        self.registry["models"].append(entry)
        self._save_registry()
        
        print(f"✅ Registered {model_name} {version}")
        return entry
    
    def get_active_model(self, model_name):
        """Load the currently active version of a model."""
        for m in self.registry["models"]:
            if m["name"] == model_name and m["is_active"]:
                return joblib.load(m["filepath"])
        raise ValueError(f"No active model found for '{model_name}'")
    
    def get_model_info(self, model_name):
        """Get metadata for the active version of a model."""
        for m in self.registry["models"]:
            if m["name"] == model_name and m["is_active"]:
                return m
        return None
    
    def list_models(self):
        """List all registered models."""
        return self.registry["models"]
    
    def list_active_models(self):
        """List only active (current) model versions."""
        return [m for m in self.registry["models"] if m["is_active"]]
    
    def _save_registry(self):
        """Persist registry to disk."""
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)
