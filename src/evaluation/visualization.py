import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from ..utils.config import Config

def explain_prediction(model, X_train, X_instance, feature_names, model_name):
    """Generate SHAP explanation for a single prediction and feature importances."""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    try:
        explainer = shap.TreeExplainer(model)
        
        # summary plot
        X_sample = X_train[:500] if len(X_train) > 500 else X_train
        shap_values_all = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_all, X_sample, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'shap_summary_{model_name.replace(" ", "_").lower()}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        shap_values_single = explainer.shap_values(X_instance.reshape(1, -1))
        
        return shap_values_single
    except Exception as e:
        print(f"SHAP explanation failed for {model_name}: {e}")
        return None

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:] # top 15
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances — {model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'feature_importance_{model_name.replace(" ", "_").lower()}.png'), dpi=150)
        plt.close()
