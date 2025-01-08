import cv2, mediapipe as mp, numpy as np
from config import MODEL_CONFIG, BODY_PARTS, ANATOMY_CONFIG as AC

class PoseDetector:
    def load_pose_model():
        return mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2, 
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    def detect_pose(pose_model, image, original_width, original_height, is_silhouette=False):
        if is_silhouette:
            pose_model.min_detection_confidence = 0.3
            
        points = [None] * 33
        results = pose_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            height, width = image.shape[:2]
            landmarks = results.pose_landmarks.landmark
            
            for i in range(33):
                try:
                    lm = landmarks[i]
                    points[i] = (int(lm.x * width), int(lm.y * height))
                except IndexError:
                    points[i] = None
                    
        return points

    def filter_anatomical_outliers(points):
        if points is None or len(points) < max(BODY_PARTS.values()) + 1:
            return points
        
        filtered = points.copy()
        
        try:
            r_shoulder = points[BODY_PARTS['RShoulder']]
            l_shoulder = points[BODY_PARTS['LShoulder']]
            
            if r_shoulder and l_shoulder:
                shoulder_dist = np.sqrt((r_shoulder[0] - l_shoulder[0])**2 + (r_shoulder[1] - l_shoulder[1])**2)
                if shoulder_dist > abs(r_shoulder[0] - l_shoulder[0]) * AC['shoulder_symmetry_threshold']:
                    filtered[BODY_PARTS['RShoulder']] = filtered[BODY_PARTS['LShoulder']] = None
        except (IndexError, TypeError):
            pass
        
        return filtered