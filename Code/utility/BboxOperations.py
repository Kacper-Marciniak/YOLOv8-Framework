import numpy as np


def convertXYWHToXYXY(a_Bbox: np.ndarray | list):
    """
    Converts bounding box from [X Y W H] to [X1 Y1 X2 Y2] format
    """
    a_Bbox = np.array(a_Bbox, dtype=float)
    x1, y1 = a_Bbox[0]-a_Bbox[2]/2, a_Bbox[1]-a_Bbox[3]/2
    x2, y2 = a_Bbox[0]+a_Bbox[2]/2, a_Bbox[1]+a_Bbox[3]/2
    return np.array(
        [x1, y1, x2, y2], dtype=float
    )

def convertXYXYToXYWH(a_Bbox: np.ndarray | list):
    """
    Converts bounding box from [X1 Y1 X2 Y2] to [X Y W H] format
    """
    a_Bbox = np.array(a_Bbox, dtype=float)
    w, h = a_Bbox[2]-a_Bbox[0], a_Bbox[3]-a_Bbox[1]
    x, y = a_Bbox[0]+w/2, a_Bbox[1]+h/2
    return np.array(
        [x, y, w, h], dtype=float
    )

def convertAbsToRel(a_Bbox: np.ndarray | list, t_ImageShape: tuple):
    """
    Converts bounding box from absolute to relative size format
    """
    a_Bbox = np.array(a_Bbox, dtype=float)
    a_Bbox[0] /= float(t_ImageShape[1])
    a_Bbox[2] /= float(t_ImageShape[1])
    a_Bbox[1] /= float(t_ImageShape[0])
    a_Bbox[3] /= float(t_ImageShape[0])
    return a_Bbox

def scale(a_Bbox: np.ndarray | list, f_ResizeX: float, f_ResizeY: float):
    """
    Scale bounding box
    """
    a_Bbox = np.array(a_Bbox, dtype=float)
    return np.array(
        [a_Bbox[0]*f_ResizeX,a_Bbox[1]*f_ResizeY,a_Bbox[2]*f_ResizeX,a_Bbox[3]*f_ResizeY], dtype=float
    )
