import yaml, os
from path.root import ROOT_DIR

# Available model types/sizes
AVAILABLE_MODELS = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8m-seg",
    "yolov8l-seg",
    "yolov8x-seg",
    "yolov8n-cls",
    "yolov8s-cls",
    "yolov8m-cls",
    "yolov8l-cls",
    "yolov8x-cls",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
    "yolo11n-seg",
    "yolo11s-seg",
    "yolo11m-seg",
    "yolo11l-seg",
    "yolo11x-seg",
]

# Training parameters
with open(os.path.join(ROOT_DIR, 'configuration', 'training.yaml'), 'r') as _File:
    TRAINING_PARAMETERS = yaml.safe_load(_File)

    print(TRAINING_PARAMETERS)