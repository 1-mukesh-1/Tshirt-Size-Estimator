import cv2
import mediapipe as mp
from image_processor import ImageProcessor
from pose_detector import PoseDetector
from segmentation_processor import SegmentationProcessor
from measurement_calculator import MeasurementCalculator
from visualizer import Visualizer
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

class BodyMeasurement:
    def __init__(self, image_path, height, is_silhouette=False, generate_visualizations=True, generate_measurements=True):
        self.image_path, self.height, self.is_silhouette = image_path, height, is_silhouette
        self.generate_visualizations, self.generate_measurements = generate_visualizations, generate_measurements
        self.data = {'image': None, 'enhanced_image': None, 'mask': None, 'points': None,
                    'true_shoulders': {'left': None, 'right': None}, 'height_points': {},
                    'measurements': {}, 'real_measurements': {}, 'visualizations': {}}
        self.pose_detector = self.yolo_model = None

    def _initialize_models(self):
        self.pose_detector, self.yolo_model = PoseDetector.load_pose_model(), SegmentationProcessor.load_yolo_model()

    def _process_image(self):
        self.data['image'] = cv2.imread(self.image_path)
        if self.is_silhouette:
            tmp = self.data['image']
            rgb_image = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            if np.mean(rgb_image) < 127: tmp = cv2.bitwise_not(tmp)
            kernel = np.ones((5,5), np.uint8)
            tmp = cv2.morphologyEx(cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
            edges = cv2.Canny(tmp, 100, 200)
            self.data['enhanced_image'] = cv2.addWeighted(tmp, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)
            self.data['mask'] = cv2.cvtColor(self.data['enhanced_image'], cv2.COLOR_BGR2GRAY)
        else:
            self.data['enhanced_image'] = ImageProcessor.enhance_image(self.data['image'])
        return self.data['enhanced_image'], self.data['image'].shape[1], self.data['image'].shape[0]

    def _detect_features(self, image, width, height):
        self.data['points'] = PoseDetector.detect_pose(self.pose_detector, image, width, height, self.is_silhouette)
        self.data['mask'] = SegmentationProcessor.perform_segmentation(self.data['image'], self.yolo_model)

    def _calculate_measurements(self):
        self.data['true_shoulders'] = MeasurementCalculator.calculate_true_shoulders(self.data['mask'], self.data['points'])
        self.data['height_points'] = MeasurementCalculator.find_height_points(self.data['mask'], self.data['points'])
        self.data['measurements'] = MeasurementCalculator.calculate_measurements(self.data['height_points'], self.data['true_shoulders'])
        scaling_factor = MeasurementCalculator.calculate_scaling_factor(self.data['measurements'].get('total_height'), self.height)
        self.data['real_measurements'] = MeasurementCalculator.convert_to_real_measurements(self.data['measurements'], scaling_factor)
        self.data['real_measurements']["torso_height"] = self.get_garment_length(self.data["image"]) * scaling_factor

    def _generate_visualizations(self):
        self.data['visualizations'] = {
            'pose': Visualizer.draw_skeleton(self.data['image'], self.data['points']),
            'segmentation': Visualizer.create_mask_overlay(self.data['image'], self.data['mask']),
            'mask_skeleton': Visualizer.draw_skeleton(cv2.cvtColor(self.data['mask'], cv2.COLOR_GRAY2BGR), self.data['points']),
            'shoulder_landmarks': Visualizer.draw_shoulder_landmarks(self.data['image'], self.data['points'], self.data['true_shoulders']),
            'shoulder_measurements': Visualizer.draw_shoulder_measurements(self.data['image'], self.data['true_shoulders']),
            'measurement_points': Visualizer.draw_measurement_points(self.data['mask'], self.data['height_points']),
            'measurement_distances': Visualizer.draw_measurement_distances(self.data['mask'], self.data['height_points'], self.data['measurements'])
        }
        Visualizer.display_results(self.data)
        self.data['visualizations'] = None

    def execute(self):
        self._initialize_models()
        image, width, height = self._process_image()
        self._detect_features(image, width, height)
        self._calculate_measurements()
        if self.generate_visualizations: self._generate_visualizations()
        if self.generate_measurements: Visualizer.print_measurements(self.data['real_measurements'])
        return {'pixel_measurements': self.data['measurements'], 'real_measurements': self.data['real_measurements']}
    
    def get_garment_length(self, img):
        model = YOLO('geometric_model/segmentation_training/best_model/best.onnx', task='segment')
        results = model(img, conf=0.25, classes=[0, 4, 7])
        binary_mask = np.any(results[0].masks.data.cpu().numpy(), axis=0).astype(np.uint8) * 255
        y_coords = np.nonzero(binary_mask)[0]
        top_y = np.min(y_coords)
        bottom_y = np.max(y_coords)
        vertical_length = bottom_y - top_y
        
        if (self.generate_visualizations):
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(binary_mask, cmap='gray')
            plt.axhline(y=top_y, color='r', linestyle='-', linewidth=1)
            plt.axhline(y=bottom_y, color='b', linestyle='-', linewidth=1)
            plt.text(10, top_y-10, f'Length: {vertical_length} pixels', color='red')
            plt.axis('off')
            plt.show()
        
        return vertical_length