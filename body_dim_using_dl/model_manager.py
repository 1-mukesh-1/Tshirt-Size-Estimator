import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config as cfg
from model import BodyMeasurementModel
from dataset import BodyMeasurementDataset

class ModelManager:
    def __init__(self):
        print(cfg)
        self.model = self.scaler = None
        self.history = {'train_loss': [], 'val_loss': [], 
                       'val_errors': {m: [] for m in cfg.MODEL_CONFIG['target_measurements']}}

    def train(self, train_data, save_path='best_model.pth'):
        dataset = BodyMeasurementDataset(train_data)
        self.scaler = dataset.scaler
        train_size = int(cfg.TRAIN_CONFIG['train_split'] * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN_CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN_CONFIG['batch_size'])
        
        self.model = BodyMeasurementModel().to(cfg.TRAIN_CONFIG['device'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=cfg.TRAIN_CONFIG['learning_rate'],
                             weight_decay=cfg.TRAIN_CONFIG['weight_decay'])
        
        best_val_loss, patience_counter = float('inf'), 0
        
        for epoch in range(cfg.TRAIN_CONFIG['num_epochs']):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.TRAIN_CONFIG["num_epochs"]}'):
                landmarks = batch['landmarks'].to(cfg.TRAIN_CONFIG['device'])
                height = batch['height'].to(cfg.TRAIN_CONFIG['device'])
                target = batch['measurements'].to(cfg.TRAIN_CONFIG['device'])
                
                optimizer.zero_grad()
                outputs = self.model(landmarks, height)
                loss = sum(criterion(outputs[m], target[:, i:i+1]) 
                          for i, m in enumerate(cfg.MODEL_CONFIG['target_measurements']))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            measurement_errors = {m: [] for m in cfg.MODEL_CONFIG['target_measurements']}
            
            with torch.no_grad():
                for batch in val_loader:
                    landmarks = batch['landmarks'].to(cfg.TRAIN_CONFIG['device'])
                    height = batch['height'].to(cfg.TRAIN_CONFIG['device'])
                    target = batch['measurements'].to(cfg.TRAIN_CONFIG['device'])
                    outputs = self.model(landmarks, height)
                    batch_loss = sum(criterion(outputs[m], target[:, i:i+1]) 
                                   for i, m in enumerate(cfg.MODEL_CONFIG['target_measurements']))
                    val_loss += batch_loss.item()
                    for i, m in enumerate(cfg.MODEL_CONFIG['target_measurements']):
                        measurement_errors[m].append(torch.abs(outputs[m] - target[:, i:i+1]).mean().item())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for m in cfg.MODEL_CONFIG['target_measurements']:
                self.history['val_errors'][m].append(np.mean(measurement_errors[m]))
            
            print(f'\nEpoch {epoch+1}/{cfg.TRAIN_CONFIG["num_epochs"]}:')
            print(f'Training Loss: {train_loss:.4f}\nValidation Loss: {val_loss:.4f}\n\nValidation Errors:')
            for m in cfg.MODEL_CONFIG['target_measurements']:
                print(f'{m}: {np.mean(measurement_errors[m]):.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(), 'scaler': self.scaler,
                           'val_loss': val_loss, 'history': self.history}, save_path)
            elif (patience_counter := patience_counter + 1) >= cfg.TRAIN_CONFIG['early_stopping_patience']:
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                break
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=cfg.TRAIN_CONFIG['device'])
        self.model = BodyMeasurementModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.history = checkpoint.get('history', self.history)
        self.model.to(cfg.TRAIN_CONFIG['device']).eval()
    
    def predict(self, landmarks, height):
        if self.model is None: raise ValueError("Model not loaded. Call load_model() first.")
        self.model.eval()
        height *= 1.25
        with torch.no_grad():
            landmarks_tensor = torch.FloatTensor(np.array(landmarks, dtype=np.float32).flatten()).unsqueeze(0)
            height_tensor = torch.FloatTensor([height/200.0]).unsqueeze(0)
            outputs = self.model(landmarks_tensor.to(cfg.TRAIN_CONFIG['device']), 
                               height_tensor.to(cfg.TRAIN_CONFIG['device']))
            predictions = self.scaler.inverse_transform(np.array([[outputs[m].cpu().numpy()[0, 0] 
                         for m in cfg.MODEL_CONFIG['target_measurements']]]))
            return {m: float(predictions[0, i]) for i, m in enumerate(cfg.MODEL_CONFIG['target_measurements'])}
    
    def evaluate(self, test_data):
        if self.model is None: raise ValueError("Model not loaded. Call load_model() first.")
        self.model.eval()
        predictions = {m: [] for m in cfg.MODEL_CONFIG['target_measurements']}
        actuals = {m: [] for m in cfg.MODEL_CONFIG['target_measurements']}
        successful_rows = 0
        
        for idx, row in tqdm(test_data.iterrows(), desc="Evaluating"):
            try:
                landmarks = json.loads(row['Landmarks']) if isinstance(row['Landmarks'], str) else row['Landmarks']
                pred = self.predict(landmarks, row['height_cm'])
                for measurement in cfg.MODEL_CONFIG['target_measurements']:
                    predictions[measurement].append(pred[measurement])
                    actuals[measurement].append(row[measurement])
                successful_rows += 1
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        if successful_rows == 0: raise ValueError("No successful predictions were made.")
        
        metrics = [{'Measurement': m, 
                   'MAE': mean_absolute_error(actuals[m], predictions[m]),
                   'RMSE': np.sqrt(mean_squared_error(actuals[m], predictions[m])),
                   'R2': r2_score(actuals[m], predictions[m])} 
                  for m in cfg.MODEL_CONFIG['target_measurements']]
        
        return {'metrics': pd.DataFrame(metrics).set_index('Measurement'),
                'predictions': predictions, 'actuals': actuals}