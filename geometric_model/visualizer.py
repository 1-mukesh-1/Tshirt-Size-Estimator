import cv2, matplotlib.pyplot as plt, numpy as np
from config import VIS_CONFIG as VC, POSE_PAIRS, BODY_PARTS as BP, MEASUREMENT_CONFIG as MCONFIG
from measurement_calculator import MeasurementCalculator as MCalc

class Visualizer:
    @staticmethod
    def draw_skeleton(image, points):
        if points is None or len(points) < 33: 
            return image
            
        result = image.copy()
        for pair in POSE_PAIRS:
            try:
                idA, idB = BP.get(pair[0]), BP.get(pair[1])
                if (idA is not None and idB is not None and 
                    idA < len(points) and idB < len(points) and 
                    points[idA] and points[idB]):
                    
                    cv2.line(result, points[idA], points[idB], 
                            VC['colors']['skeleton_line'], 
                            thickness=VC['sizes']['line_thickness'])
                    
                    for point in [points[idA], points[idB]]:
                        cv2.circle(result, point, VC['sizes']['point_radius'],
                                 VC['colors']['skeleton_point'],
                                 thickness=VC['sizes']['line_thickness'],
                                 lineType=cv2.FILLED)
            except Exception:
                continue
        return result

    @staticmethod
    def create_mask_overlay(image, mask):
        overlay = image.copy()
        overlay[mask > 0] = (overlay[mask > 0] * VC['overlay']['alpha'] + 
                           VC['overlay']['green_tint'] * (1 - VC['overlay']['alpha']))
        return overlay

    @staticmethod
    def draw_shoulder_landmarks(image, points, true_shoulders):
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        for side, idx in [('L', BP['LShoulder']), ('R', BP['RShoulder'])]:
            if points[idx]:
                cv2.circle(result, points[idx], VC['sizes']['point_radius'],
                          VC['colors']['original_point'], thickness=VC['sizes']['line_thickness'])
                label_pos = (points[idx][0] + (10 if side == 'L' else -80), points[idx][1])
                cv2.putText(result, f"Original {side}", label_pos, VC['plot']['font'],
                           VC['sizes']['text_scale'], VC['colors']['text'],
                           VC['sizes']['text_thickness'])

        for side in ['left', 'right']:
            if true_shoulders[side]:
                cv2.circle(result, true_shoulders[side], VC['sizes']['point_radius'],
                          VC['colors']['true_point'], thickness=VC['sizes']['line_thickness'])
                
                label_pos = (true_shoulders[side][0] + (10 if side == 'left' else -80),
                            true_shoulders[side][1])
                cv2.putText(result, f"True {'L' if side == 'left' else 'R'}", label_pos,
                           VC['plot']['font'], VC['sizes']['text_scale'], 
                           VC['colors']['text'], VC['sizes']['text_thickness'])
                
                orig_idx = BP['LShoulder' if side == 'left' else 'RShoulder']
                if points[orig_idx]:
                    cv2.line(result, points[orig_idx], true_shoulders[side],
                            VC['colors']['connection_line'],
                            thickness=VC['sizes']['line_thickness'])
        return result

    @staticmethod
    def draw_shoulder_measurements(image, true_shoulders):
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        if all(true_shoulders.values()):
            for point in true_shoulders.values():
                cv2.circle(result, point, VC['sizes']['point_radius'],
                          VC['colors']['true_point'], thickness=VC['sizes']['line_thickness'])
            
            cv2.line(result, true_shoulders['left'], true_shoulders['right'],
                    VC['colors']['height_line'], thickness=VC['sizes']['line_thickness'])
            
            distance = MCalc.calculate_distance(true_shoulders['left'], true_shoulders['right'])
            if distance:
                mid_point = MCalc.calculate_midpoint(true_shoulders['left'], true_shoulders['right'])
                if mid_point:
                    cv2.putText(result, f"Width: {distance:.1f}px",
                              (mid_point[0] - 60, mid_point[1] - 20),
                              VC['plot']['font'], VC['sizes']['text_scale'],
                              VC['colors']['text'], VC['sizes']['text_thickness'])
        return result

    @staticmethod
    def draw_measurement_points(mask, height_points):
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        point_labels = {'top':'Top', 'hip_mid':'Hip Mid', 'left_knee':'L Knee',
                       'right_knee':'R Knee', 'left_ankle':'L Ankle', 'right_ankle':'R Ankle'}
        
        for key, label in point_labels.items():
            if point := height_points.get(key):
                cv2.circle(result, point, VC['sizes']['point_radius'],
                          VC['colors']['height_point'], thickness=VC['sizes']['line_thickness'])
                cv2.putText(result, label, 
                           (point[0] + MCONFIG['text_offset']['x'], 
                            point[1] + MCONFIG['text_offset']['y']),
                           VC['plot']['font'], VC['sizes']['text_scale'],
                           VC['colors']['text'], VC['sizes']['text_thickness'])
        return result

    @staticmethod
    def draw_measurement_distances(mask, height_points, measurements):
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        pairs = [('torso','top','hip_mid'), ('left_hip_knee','hip_mid','left_knee'),
                ('right_hip_knee','hip_mid','right_knee'), ('left_knee_ankle','left_knee','left_ankle'),
                ('right_knee_ankle','right_knee','right_ankle')]

        for mkey, skey, ekey in pairs:
            if all((measurement := measurements.get(mkey), 
                   start := height_points.get(skey),
                   end := height_points.get(ekey))):
                cv2.line(result, start, end, VC['colors']['height_line'],
                        thickness=VC['sizes']['line_thickness'])
                
                if mid := MCalc.calculate_midpoint(start, end):
                    cv2.putText(result, f"{measurement:.1f}px",
                              (mid[0] + MCONFIG['text_offset']['x'], 
                               mid[1] + MCONFIG['text_offset']['y']),
                              VC['plot']['font'], VC['sizes']['text_scale'],
                              VC['colors']['text'], VC['sizes']['text_thickness'])
        return result

    @staticmethod
    def display_results(data):
        plt.figure(figsize=VC['plot']['figure_size'])
        plots = [('Original Image', data['image']),
                ('Detected Pose', data['visualizations']['pose']),
                ('Instance Segmentation', data['visualizations']['segmentation']),
                ('Binary Mask with Pose', data['visualizations']['mask_skeleton']),
                ('Shoulder Landmarks', data['visualizations']['shoulder_landmarks']),
                ('Shoulder Width', data['visualizations']['shoulder_measurements']),
                ('Measurement Points', data['visualizations']['measurement_points']),
                ('Measurement Distances', data['visualizations']['measurement_distances'])]
        
        for idx, (title, img) in enumerate(plots, 1):
            plt.subplot(2, 4, idx)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_measurements(m):
        print("\nIn Centimeters:")
        print(f"Shoulder width: {m.get('shoulder_width', 'N/A'):.1f} cm")
        print(f"Torso height: {m.get('torso', 'N/A'):.1f} cm")
        print("\nLeft leg measurements:")
        print(f"Hip to knee: {m.get('left_hip_knee', 'N/A'):.1f} cm")
        print(f"Knee to ankle: {m.get('left_knee_ankle', 'N/A'):.1f} cm") 
        print("\nRight leg measurements:")
        print(f"Hip to knee: {m.get('right_hip_knee', 'N/A'):.1f} cm")
        print(f"Knee to ankle: {m.get('right_knee_ankle', 'N/A'):.1f} cm")
        print(f"\nTotal height: {m.get('total_height', 'N/A'):.1f} cm")