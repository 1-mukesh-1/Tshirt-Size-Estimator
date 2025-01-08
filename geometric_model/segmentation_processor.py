import numpy as np, cv2
from ultralytics import YOLO
from config import MODEL_CONFIG as MC

class SegmentationProcessor:
   def load_yolo_model():
       return YOLO(MC['yolo']['model_path'])

   def perform_segmentation(image, model):
       results = model(image, conf=MC['yolo']['confidence_threshold'], classes=[MC['yolo']['person_class_id']])
       mask = np.zeros(image.shape[:2], dtype=np.uint8)
       
       if len(results[0].masks) > 0:
           seg_mask = cv2.resize(results[0].masks[0].data.cpu().numpy()[0], (image.shape[1], image.shape[0]))
           mask = (seg_mask > 0.5).astype(np.uint8) * 255
           
       return mask