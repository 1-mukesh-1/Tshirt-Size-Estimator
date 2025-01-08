import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import config as cfg

class Visualizer:
    def plot_training_history(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.title('Training History'), plt.legend()
        plt.show()
        
        plt.figure(figsize=(15, 8))
        [plt.plot(history['val_errors'][m], label=m) for m in cfg.TARGET_MEASUREMENTS]
        plt.xlabel('Epoch'), plt.ylabel('Mean Absolute Error (cm)')
        plt.title('Validation Errors by Measurement')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(), plt.show()
    
    def plot_evaluation_results(self, results, dataset_name):
        metrics_df = results['metrics']
        plt.figure(figsize=(12, 10))
        sns.heatmap(metrics_df[['MAE', 'RMSE', 'R2']], annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title(f'Performance Metrics Heatmap - {dataset_name}')
        plt.tight_layout(), plt.show()
        
        n_measurements = len(cfg.TARGET_MEASUREMENTS)
        rows = int(np.ceil(n_measurements/2))
        
        # Scatter plots
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for idx, m in enumerate(cfg.TARGET_MEASUREMENTS):
            pred, actual = results['predictions'][m], results['actuals'][m]
            axes[idx].scatter(actual, pred, alpha=0.5)
            axes[idx].plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
            axes[idx].set(xlabel='Actual (cm)', ylabel='Predicted (cm)', 
                         title=f'{m} - Predictions vs Actual')
            axes[idx].text(0.05, 0.95, f'R² = {results["metrics"].loc[m, "R2"]:.3f}',
                         transform=axes[idx].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        if n_measurements % 2: fig.delaxes(axes[-1])
        plt.tight_layout(), plt.show()
        
        # Error distributions
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for idx, m in enumerate(cfg.TARGET_MEASUREMENTS):
            errors = np.array(results['predictions'][m]) - np.array(results['actuals'][m])
            sns.histplot(errors, kde=True, ax=axes[idx])
            axes[idx].axvline(x=0, color='r', linestyle='--')
            axes[idx].set(xlabel='Error (cm)', ylabel='Count', title=f'{m} - Error Distribution')
            axes[idx].text(0.05, 0.95, f'Mean = {np.mean(errors):.2f}\nStd = {np.std(errors):.2f}',
                         transform=axes[idx].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        if n_measurements % 2: fig.delaxes(axes[-1])
        plt.tight_layout(), plt.show()