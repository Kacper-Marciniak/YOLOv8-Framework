import yaml, os
from path.root import ROOT_DIR

# Available model types/sizes
AVAILABLE_MODELS = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
]

# Training parameters
with open(os.path.join(ROOT_DIR, 'configuration', 'training.yaml'), 'r') as _File:
    TRAINING_PARAMETERS = yaml.safe_load(_File)

    print(TRAINING_PARAMETERS)