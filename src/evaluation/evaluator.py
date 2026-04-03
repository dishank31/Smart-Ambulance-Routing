import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, learning_curve
import time
import os
from ..utils.config import Config

class ModelEvaluator:
    def __init__(self, task='classification'):
        self.task = task
        self.results = {}
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    def evaluate_classification(self, model, X_test, y_test, model_name):
        start = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start
        
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'Model': model_name,
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'Precision (Weighted)': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'Recall (Weighted)': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'F1 Score (Weighted)': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'F1 Score (Macro)': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'Inference Time (ms)': float(inference_time * 1000 / len(X_test) if len(X_test) > 0 else 0)
        }
        
        if y_proba is not None:
            try:
                metrics['ROC-AUC (OVR)'] = float(roc_auc_score(
                    y_test, y_proba, multi_class='ovr', average='weighted'
                ))
            except Exception:
                metrics['ROC-AUC (OVR)'] = np.nan
                
        self.results[model_name] = metrics
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return metrics

    def evaluate_regression(self, model, X_test, y_test, model_name):
        start = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start
        
        metrics = {
            'Model': model_name,
            'MAE': float(mean_absolute_error(y_test, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'MAPE': float(mean_absolute_percentage_error(y_test, y_pred) * 100),
            'R² Score': float(r2_score(y_test, y_pred)),
            'Inference Time (ms)': float(inference_time * 1000 / len(X_test) if len(X_test) > 0 else 0)
        }
        
        self.results[model_name] = metrics
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"MAE: {metrics['MAE']:.4f} | R²: {metrics['R² Score']:.4f}")
        return metrics

    def plot_confusion_matrix(self, model, X_test, y_test, model_name, labels=None):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix — {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'), dpi=150)
        plt.close()

    def plot_roc_curves(self, models_dict, X_test, y_test):
        plt.figure(figsize=(12, 8))
        
        for name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_test)
                if len(classes) > 2:
                    y_test_bin = label_binarize(y_test, classes=classes)
                    try:
                        auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
                        
                        # Plot macro-average ROC
                        fpr_dict = dict()
                        tpr_dict = dict()
                        for i in range(len(classes)):
                            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        
                        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(classes))]))
                        mean_tpr = np.zeros_like(all_fpr)
                        for i in range(len(classes)):
                            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
                        mean_tpr /= len(classes)
                        
                        plt.plot(all_fpr, mean_tpr, label=f'{name} (Macro AUC={auc:.3f})', linewidth=2)
                    except Exception:
                        pass
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves — Model Comparison', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'roc_curves_comparison_{self.task}.png'), dpi=150)
        plt.close()

    def plot_model_comparison_bar(self):
        if not self.results:
            return
            
        df = pd.DataFrame(self.results).T
        metrics_to_plot = [col for col in df.columns if col not in ['Model', 'Inference Time (ms)'] and 'AUC' not in col]
        
        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            return
            
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            if df[metric].dtype in [np.float64, np.int64, float, int]:
                ax = axes[i] if n_metrics > 1 else axes
                vals = df[metric].astype(float)
                bars = ax.barh(df.index, vals, color=plt.cm.Set2.colors)
                ax.set_xlabel(metric)
                ax.set_title(metric)
                
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'model_comparison_{self.task}.png'), dpi=150)
        plt.close()

    def plot_learning_curve(self, model, X, y, model_name, cv=3):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 4),
            scoring='accuracy' if self.task == 'classification' else 'neg_mean_absolute_error'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        if self.task != 'classification':
            train_mean = -train_mean
            test_mean = -test_mean
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('Error' if self.task != 'classification' else 'Accuracy')
        plt.title(f'Learning Curve — {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, f'learning_curve_{model_name.replace(" ", "_").lower()}.png'), dpi=150)
        plt.close()
