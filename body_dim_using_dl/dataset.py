import torch
from torch.utils.data import Dataset
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import config as cfg

class BodyMeasurementDataset(Dataset):
    def __init__(self, df, scaler=None, augment=True):
        self.df = df
        self.measurements = cfg.MODEL_CONFIG['target_measurements']
        self.augment = augment
        if scaler is None:
            self.scaler = StandardScaler()
            measurement_values = np.array([self.df[m].values for m in self.measurements]).T
            self.scaler.fit(measurement_values)
        else: self.scaler = scaler
    
    def augment_data(self, landmarks, height_cm):
        landmarks = np.array(landmarks).reshape(-1, 2)
        scale = np.random.uniform(0.95, 1.05)
        landmarks = landmarks * scale
        height_cm = height_cm * scale
        tx, ty = np.random.uniform(-0.02, 0.02, 2)
        landmarks[:, 0] += tx; landmarks[:, 1] += ty
        return landmarks.flatten(), height_cm
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        landmarks = json.loads(row['Landmarks']) if isinstance(row['Landmarks'], str) else row['Landmarks']
        height_cm = row['height_cm']
        
        if self.augment and np.random.random() > 0.5:
            landmarks, height_cm = self.augment_data(landmarks, height_cm)
        else: landmarks = np.array(landmarks).flatten()
        
        measurements = self.scaler.transform(
            np.array([row[m] for m in self.measurements]).reshape(1, -1)
        ).flatten()
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'height': torch.FloatTensor([height_cm/200.0]),
            'measurements': torch.FloatTensor(measurements),
            'file_name': str(row['photo_id'])
        }