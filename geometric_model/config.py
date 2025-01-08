import cv2, numpy as np, mediapipe as mp

MODEL_CONFIG = {
    'pose': {'min_detection_confidence': 0.3, 'model_complexity': 2, 'static_image_mode': True, 'enable_segmentation': True},
    'yolo': {'model_path': 'yolov8x-seg.pt', 'confidence_threshold': 0.5, 'person_class_id': 0}
}

IMAGE_PROCESSING = {
    'enhancement': {
        'clahe': {'clip_limit': 3.0, 'tile_grid_size': (8, 8)},
        'denoising': {'h_luminance': 10, 'photo_render': 10, 'search_window': 7, 'block_size': 21}
    },
    'scaling': {
        'factors': [0.8, 1.0, 1.2],
        'blob': {'target_size': (368, 368), 'scale_factor': 1.0/255, 'mean': (0, 0, 0), 'swap_rb': False, 'crop': False}
    }
}

VIS_CONFIG = {
    'colors': {
        'skeleton_line': (255, 255, 0), 'skeleton_point': (0, 0, 255), 'original_point': (0, 0, 255),
        'true_point': (0, 255, 0), 'connection_line': (255, 0, 0), 'text': (0, 165, 255),
        'height_point': (128, 0, 128), 'height_line': (0, 0, 0)
    },
    'sizes': {'line_thickness': 15, 'point_radius': 7, 'text_scale': 1, 'text_thickness': 8},
    'plot': {'figure_size': (20, 10), 'font': cv2.FONT_HERSHEY_SIMPLEX},
    'overlay': {'alpha': 0.7, 'green_tint': np.array([0, 255, 0], dtype=np.uint8)}
}

ANATOMY_CONFIG = {'shoulder_symmetry_threshold': 0.5, 'shoulder_detection': {'step_size': 1}}
MEASUREMENT_CONFIG = {'decimal_precision': 1, 'text_offset': {'x': 10, 'y': -10}}

mp_pose = mp.solutions.pose.PoseLandmark
BODY_PARTS = {
    "Nose": mp_pose.NOSE.value, "RShoulder": mp_pose.RIGHT_SHOULDER.value, "RElbow": mp_pose.RIGHT_ELBOW.value,
    "RWrist": mp_pose.RIGHT_WRIST.value, "LShoulder": mp_pose.LEFT_SHOULDER.value, "LElbow": mp_pose.LEFT_ELBOW.value,
    "LWrist": mp_pose.LEFT_WRIST.value, "RHip": mp_pose.RIGHT_HIP.value, "RKnee": mp_pose.RIGHT_KNEE.value,
    "RAnkle": mp_pose.RIGHT_ANKLE.value, "LHip": mp_pose.LEFT_HIP.value, "LKnee": mp_pose.LEFT_KNEE.value,
    "LAnkle": mp_pose.LEFT_ANKLE.value, "REye": mp_pose.RIGHT_EYE.value, "LEye": mp_pose.LEFT_EYE.value,
    "REar": mp_pose.RIGHT_EAR.value, "LEar": mp_pose.LEFT_EAR.value
}

POSE_PAIRS = [
    ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["RShoulder", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LShoulder", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Nose", "REye"], ["REye", "REar"], 
    ["Nose", "LEye"], ["LEye", "LEar"]
]