import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import config as cfg

class DataProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, 
            model_complexity=cfg.PREPROCESS_CONFIG['model_complexity'],
            min_detection_confidence=cfg.PREPROCESS_CONFIG['min_detection_confidence'],
            enable_segmentation=True)
    
    def _process_silhouette(self, image):
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if np.mean(image) < 127: image = cv2.bitwise_not(image)
        kernel = np.ones((cfg.PREPROCESS_CONFIG['morph_kernel_size'], 
                         cfg.PREPROCESS_CONFIG['morph_kernel_size']), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def _process_regular_photo(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (192, 192, 192)
            image = np.where(mask[..., np.newaxis], image, bg_image)
        return image
    
    def _visualize_landmarks(self, image, landmarks):
        viz_image = image.copy()
        for idx, landmark in enumerate(landmarks):
            x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
            cv2.circle(viz_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(viz_image, str(idx), (x+5, y+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return viz_image
    
    def process_image(self, image_path, visualize=False):
        image = cv2.imread(str(image_path))
        if image is None: raise ValueError(f"Could not read image: {image_path}")
        image = cv2.resize(image, cfg.PREPROCESS_CONFIG['image_size'])
        processed_image = self._process_silhouette(image) if cfg.PREPROCESS_CONFIG['is_silhouette'] \
            else self._process_regular_photo(image)
        results = self.pose.process(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks: return None
        landmarks = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
        return (landmarks, self._visualize_landmarks(processed_image, landmarks)) if visualize else landmarks
    
    def process_dataset(self, data_path, output_path=None):
        df = pd.read_csv(data_path)
        if 'Landmarks' not in df.columns: df['Landmarks'] = None
        successful = failed = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            try:
                landmarks = self.process_image(row['image_path'])
                if landmarks is not None:
                    df.at[idx, 'Landmarks'] = json.dumps(landmarks)
                    successful += 1
                else: failed += 1
            except Exception as e:
                print(f"\nError processing {row['photo_id']}: {str(e)}")
                failed += 1
        print(f"\nProcessing complete!\nSuccessfully processed: {successful} images\nFailed: {failed} images")
        if output_path: df.to_csv(output_path, index=False)
        return df