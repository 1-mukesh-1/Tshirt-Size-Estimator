import numpy as np
from config import BODY_PARTS, ANATOMY_CONFIG as AC

class MeasurementCalculator:
   def calculate_scaling_factor(pixel_height, true_height_cm):
       return None if not pixel_height else true_height_cm / pixel_height

   def convert_to_real_measurements(measurements, scaling_factor):
       return {k: v * scaling_factor for k,v in measurements.items() if v is not None} if scaling_factor else {}

   def find_true_shoulder_point(mask, shoulder_point, direction='left'):
       if not shoulder_point: return None
       x, y = shoulder_point
       h, w = mask.shape
       step = AC['shoulder_detection']['step_size']
       x_step = step if direction == 'left' else -step
       
       while 0 <= x < w and 0 <= y < h and mask[y,x] > 0:
           x += x_step
           y -= step
       return (x - x_step, y + step)

   def calculate_true_shoulders(mask, points):
       if mask is None or points is None:
           return {'left': None, 'right': None}
           
       l_shoulder = points[BODY_PARTS['LShoulder']] if points else None
       r_shoulder = points[BODY_PARTS['RShoulder']] if points else None
       
       if l_shoulder and r_shoulder and l_shoulder[0] < r_shoulder[0]:
           l_shoulder, r_shoulder = r_shoulder, l_shoulder
       
       return {
           'left': MeasurementCalculator.find_true_shoulder_point(mask, l_shoulder, 'left'),
           'right': MeasurementCalculator.find_true_shoulder_point(mask, r_shoulder, 'right')
       }

   def find_height_points(mask, points):
       default = {k: None for k in ['top', 'hip_mid', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']}
       if mask is None or points is None: return default
       
       try:
           nz = np.nonzero(mask)
           top_point = None
           if len(nz[0]) > 0:
               top_y = np.min(nz[0])
               x_coords = np.where(mask[top_y] > 0)[0]
               if len(x_coords) > 0:
                   top_point = (int(np.mean(x_coords)), top_y)

           lh = points[BODY_PARTS['LHip']] if BODY_PARTS['LHip'] < len(points) else None
           rh = points[BODY_PARTS['RHip']] if BODY_PARTS['RHip'] < len(points) else None
           hip_mid = MeasurementCalculator.calculate_midpoint(lh, rh)

           points = {
               'top': top_point,
               'hip_mid': hip_mid,
               'left_knee': points[BODY_PARTS['LKnee']] if BODY_PARTS['LKnee'] < len(points) else None,
               'right_knee': points[BODY_PARTS['RKnee']] if BODY_PARTS['RKnee'] < len(points) else None, 
               'left_ankle': points[BODY_PARTS['LAnkle']] if BODY_PARTS['LAnkle'] < len(points) else None,
               'right_ankle': points[BODY_PARTS['RAnkle']] if BODY_PARTS['RAnkle'] < len(points) else None
           }

           if hip_mid:
               if not points['left_ankle']:
                   points['left_ankle'] = MeasurementCalculator.find_lowest_point_relative_to_midpoint(mask, hip_mid[0], True)
               if not points['right_ankle']:
                   points['right_ankle'] = MeasurementCalculator.find_lowest_point_relative_to_midpoint(mask, hip_mid[0], False)
               
               if not points['left_knee'] and points['left_ankle']:
                   points['left_knee'] = MeasurementCalculator.calculate_midpoint(hip_mid, points['left_ankle'])
               if not points['right_knee'] and points['right_ankle']:
                   points['right_knee'] = MeasurementCalculator.calculate_midpoint(hip_mid, points['right_ankle'])

           return points

       except Exception as e:
           print(f"Error in find_height_points: {str(e)}")
           return default

   def calculate_measurements(height_points, true_shoulders):
       if not height_points: height_points = {}
       measurements = {}
       
       if true_shoulders and all(true_shoulders.get(s) for s in ['left', 'right']):
           measurements['shoulder_width'] = MeasurementCalculator.calculate_distance(
               true_shoulders['left'], true_shoulders['right'])

       if all(height_points.get(p) for p in ['top', 'hip_mid']):
           measurements['torso'] = MeasurementCalculator.calculate_distance(
               height_points['top'], height_points['hip_mid'])

       leg_pairs = [
           ('left_hip_knee', 'hip_mid', 'left_knee'),
           ('right_hip_knee', 'hip_mid', 'right_knee'),
           ('left_knee_ankle', 'left_knee', 'left_ankle'),
           ('right_knee_ankle', 'right_knee', 'right_ankle')
       ]
       
       for key, start, end in leg_pairs:
           if height_points.get(start) and height_points.get(end):
               measurements[key] = MeasurementCalculator.calculate_distance(height_points[start], height_points[end])

       if all(k in measurements for k in ['torso', 'left_hip_knee', 'left_knee_ankle']):
           measurements['total_height'] = (
               measurements['torso'] + 
               max(
                   measurements.get('left_hip_knee', 0) + measurements.get('left_knee_ankle', 0),
                   measurements.get('right_hip_knee', 0) + measurements.get('right_knee_ankle', 0)
               )
           )

       return measurements

   def calculate_midpoint(p1, p2):
       return None if not p1 or not p2 else (int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2))

   def calculate_distance(p1, p2):
       return None if not p1 or not p2 else np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

   def find_lowest_point_relative_to_midpoint(mask, mid_x, is_left=True):
       if mask is None or mid_x is None: return None
       h, w = mask.shape
       search_range = range(0, mid_x) if is_left else range(mid_x, w)
       
       for y in range(h-1, -1, -1):
           for x in search_range:
               if mask[y,x] > 0:
                   return (x, y)
       return None