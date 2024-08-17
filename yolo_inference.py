from ultralytics import YOLO
import numpy as np

def safe_box_coordinates(box):
    """Filter out boxes with NaN values."""
    box_array = np.array(box.xyxy)
    if np.any(np.isnan(box_array)):
        print("Warning: Detected NaN values in box coordinates")
        return None  # Skip boxes with NaN values
    return box_array

model = YOLO('yolov8x')

try:
    results = model.predict('input_videos/08fd33_4.mp4', save=True)

    print(results[0])
    print('=====================================')

    for box in results[0].boxes:
        safe_box = safe_box_coordinates(box)
        if safe_box is not None:
            print(box)
        else:
            print("Skipped box due to NaN values")

except Exception as e:
    print(f"An error occurred: {e}")
