#!/bin/bash

# Function to check if a file exists
file_exists() {
    if [ -e "$1" ]; then
        return 0
    else
        return 1
    fi
}

# Prompt the user for the model to use
echo "Choose the model you want to use:"
echo "1. Baseline Model"
echo "2. Deep Learning Model"
echo "3. Geometric Model"
read -p "Enter your choice (1, 2, or 3): " model_choice

# Prompt the user for the image path
read -p "Enter the path to the input image: " image_path

# Check if the file exists
while ! file_exists "$image_path"; do
    echo "File not found: $image_path"
    read -p "Enter the path to the input image: " image_path
done

# Prompt the user for their height
read -p "Enter your height in centimeters: " height_cm

# Execute the selected model
case $model_choice in
    1)
        echo "Running Baseline Model..."
        python -c """
import sys
import os
sys.path.append(os.path.dirname('baseline_model/'))
from baseline_model.body_measurement import BodyMeasurement as BL_Body_Measurement
measurements = BL_Body_Measurement('$image_path', height_cm=$height_cm, is_silhouette=False)
results = measurements.execute()
print('\nShoulder Width: ' + str(results['shoulder_width']) +' cm')
print(f'Torso Height: ' + str(results['torso_height']) + ' cm')
print(f'Chest Width: ' + str(results['chest_width']) + ' cm')
"""
        ;;
    2)
        echo "Running Deep Learning Model..."
        python -c "
import sys
import os
sys.path.append(os.path.dirname('body_dim_using_dl/'))
from body_dim_using_dl.model_manager import ModelManager
from body_dim_using_dl.data_processor import DataProcessor
model_manager = ModelManager()
data_processor = DataProcessor()
model_path = './body_dim_using_dl/best_model.pth'
model_manager.load_model(model_path)
landmarks = data_processor.process_image('$image_path')
predictions = model_manager.predict(landmarks, $height_cm)
print(f'\nShoulder-Breadth: {{predictions['shoulder-breadth']:.2f}} cm')
"
        ;;
    3)
        echo "Running Geometric Model..."
        python -c "
import sys
import os
sys.path.append(os.path.dirname('geometric_model/'))
from geometric_model.body_measurement import BodyMeasurement as GM_Body_Measurement
measurements = GM_Body_Measurement('$image_path', $height_cm, False, False, False).execute()
print(f'\nShoulder Width: {{measurements['real_measurements']['shoulder_width']:.2f}} cm')
"
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac