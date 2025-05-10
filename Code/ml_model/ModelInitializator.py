from ultralytics import YOLO, RTDETR


def get_model_class(model: str, task: str = None, verbose: bool = False, model_type: str = 'yolo') -> YOLO | RTDETR:
    """
    Initialize YOLO or RTDETR model
    """
    if model_type == 'yolo':
        return YOLO(model, task=task, verbose=verbose)
    else:
        return RTDETR(model)