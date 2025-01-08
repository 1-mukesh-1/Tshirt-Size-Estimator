import cv2, mediapipe as mp, numpy as np, matplotlib.pyplot as plt

class BodyMeasurement:
    def __init__(self, image_path, height_cm, is_silhouette=True):
        self.mp_pose, self.mp_draw = mp.solutions.pose, mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.height_px, self.width_px = self.image.shape[:2]
        self.height_cm, self.is_silhouette = height_cm, is_silhouette
        self.processed_image = self.preprocess_silhouette() if is_silhouette else self.image
        self.results = self.pose.process(self.processed_image)
        if not self.results.pose_landmarks: raise ValueError("No pose landmarks detected. Please check the image.")
        landmarks = self.results.pose_landmarks.landmark
        self.scaling_factor = height_cm / ((landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y - landmarks[self.mp_pose.PoseLandmark.NOSE].y) * self.height_px)

    def get_point_coordinates(self, point): return (point.x, point.y) if hasattr(point, 'x') and hasattr(point, 'y') else point

    def get_midpoint(self, point1, point2):
        x1, y1 = self.get_point_coordinates(point1)
        x2, y2 = self.get_point_coordinates(point2)
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def preprocess_silhouette(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 127, 255, cv2.THRESH_BINARY)
        mask = np.zeros_like(gray)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours: cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        masked = cv2.bitwise_and(self.image, self.image, mask=mask)
        lab = cv2.cvtColor(masked, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge((cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l), a, b)), cv2.COLOR_LAB2RGB)

    def execute(self):
        landmarks = self.results.pose_landmarks.landmark
        shoulder_width_px = abs((landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x - landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * self.width_px)
        shoulder_midpoint = self.get_midpoint(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        upper_midpoint = self.get_midpoint(landmarks[self.mp_pose.PoseLandmark.NOSE], shoulder_midpoint)
        lower_midpoint = self.get_midpoint(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP], landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        torso_height_px = abs((upper_midpoint[1] - lower_midpoint[1]) * self.height_px)
        measurements = {
            'shoulder_width': round(shoulder_width_px * self.scaling_factor, 1),
            'torso_height': round(torso_height_px * self.scaling_factor, 1),
            'chest_width': round(shoulder_width_px * 0.9 * self.scaling_factor, 1)
        }
        print(f"\nBody Measurements:\nShoulder Width: {measurements['shoulder_width']} cm\nTorso Height: {measurements['torso_height']} cm\nChest Width: {measurements['chest_width']} cm")
        return measurements

    def visualize_measurements(self):
        image, landmarks = self.image.copy(), self.results.pose_landmarks.landmark
        measurements = self.execute()
        shoulder_midpoint = self.get_midpoint(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        upper_midpoint = self.get_midpoint(landmarks[self.mp_pose.PoseLandmark.NOSE], shoulder_midpoint)
        lower_midpoint = self.get_midpoint(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP], landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        points = {
            'upper': (int(upper_midpoint[0] * self.width_px), int(upper_midpoint[1] * self.height_px)),
            'lower': (int(lower_midpoint[0] * self.width_px), int(lower_midpoint[1] * self.height_px)),
            'left_shoulder': (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * self.width_px), int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * self.height_px)),
            'right_shoulder': (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * self.width_px), int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * self.height_px))
        }
        cv2.line(image, points['left_shoulder'], points['right_shoulder'], (0, 255, 0), 2)
        cv2.line(image, points['upper'], points['lower'], (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (key, value) in enumerate(measurements.items()): cv2.putText(image, f"{key.replace('_', ' ').title()}: {value:.1f} cm", (10, 30 + 40 * i), font, 1, (0, 255, 0), 2)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        plt.close()
        return image