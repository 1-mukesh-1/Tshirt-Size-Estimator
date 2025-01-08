import torch
import torch.nn as nn
from config import MODEL_CONFIG

class BodyMeasurementModel(nn.Module):
    def __init__(self, measurements=None):
        super().__init__()
        self.measurements = measurements or MODEL_CONFIG['target_measurements']
        
        self.landmark_encoder = nn.Sequential(
            nn.Linear(MODEL_CONFIG['pose_input_dim'], 64), nn.ReLU(),
            nn.BatchNorm1d(64), nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(64, MODEL_CONFIG['hidden_dim']), nn.ReLU(),
            nn.BatchNorm1d(MODEL_CONFIG['hidden_dim'])
        )
        
        self.measurement_heads = nn.ModuleDict({
            measurement: nn.Sequential(
                nn.Linear(MODEL_CONFIG['hidden_dim'], 32), nn.ReLU(),
                nn.Dropout(MODEL_CONFIG['dropout_rate']),
                nn.Linear(32, 1)
            ) for measurement in self.measurements
        })
    
    def forward(self, landmarks, height):
        features = self.landmark_encoder(torch.cat([landmarks, height], dim=1))
        return {m: self.measurement_heads[m](features) for m in self.measurements}