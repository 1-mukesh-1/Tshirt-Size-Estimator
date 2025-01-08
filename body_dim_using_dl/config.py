import torch
import pandas as pd

def load_measurements(csv_path): return ['shoulder-breadth', 'chest']

MODEL_CONFIG = {
    'pose_input_dim': 67, 'hidden_dim': 128, 'dropout_rate': 0.5,
    'target_measurements': ['shoulder-breadth', 'chest']
}

TRAIN_CONFIG = {
    'batch_size': 8, 'num_epochs': 150, 'learning_rate': 0.0001,
    'weight_decay': 1e-4, 'train_split': 0.8, 'val_split': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'early_stopping_patience': 20
}

PREPROCESS_CONFIG = {
    'min_detection_confidence': 0.3, 'model_complexity': 2,
    'image_size': (512, 512), 'morph_kernel_size': 5,
    'is_silhouette': True
}

FILE_CONFIG = {
    'model_save_path': 'best_model.pth', 'train_output_path': 'processed_train.csv',
    'val_output_path': 'processed_val.csv', 'test_output_path': 'processed_test.csv',
    'metrics_output_path': 'evaluation_metrics.csv'
}

VIZ_CONFIG = {
    'plot_size': (12, 8), 'font_size': 10, 'dpi': 100,
    'save_plots': False, 'plots_dir': 'plots'
}

def get_config(): return {'model': MODEL_CONFIG, 'train': TRAIN_CONFIG, 
                         'preprocess': PREPROCESS_CONFIG, 'file': FILE_CONFIG, 'viz': VIZ_CONFIG}

def validate_config():
    assert MODEL_CONFIG['pose_input_dim'] > 0 and MODEL_CONFIG['hidden_dim'] > 0
    assert 0 <= MODEL_CONFIG['dropout_rate'] <= 1
    assert len(MODEL_CONFIG['target_measurements']) > 0
    assert TRAIN_CONFIG['batch_size'] > 0 and TRAIN_CONFIG['num_epochs'] > 0
    assert TRAIN_CONFIG['learning_rate'] > 0
    assert 0 <= TRAIN_CONFIG['train_split'] <= 1 and 0 <= TRAIN_CONFIG['val_split'] <= 1
    assert TRAIN_CONFIG['train_split'] + TRAIN_CONFIG['val_split'] == 1
    assert PREPROCESS_CONFIG['min_detection_confidence'] > 0
    assert PREPROCESS_CONFIG['model_complexity'] in [0, 1, 2]
    assert all(x > 0 for x in PREPROCESS_CONFIG['image_size'])
    assert PREPROCESS_CONFIG['morph_kernel_size'] > 0
    assert isinstance(PREPROCESS_CONFIG['is_silhouette'], bool)

if __name__ == "__main__":
    validate_config()
    print("Configuration validation passed!")
    print("\nCurrent Configuration:")
    for category, config in get_config().items():
        print(f"\n{category.upper()} Configuration:")
        for key, value in config.items(): print(f"{key}: {value}")