import cv2 as cv
import numpy as np
from ml_model.CResults import ImageResults

COLOURS = [
    (0, 0, 255),        # Red
    (255, 0, 0),        # Blue
    (0, 255, 0),        # Green
    (0, 255, 255),      # Yellow
    (0, 165, 255),      # Orange
    (128, 0, 128),      # Purple
    (255, 192, 203),    # Pink
    (255, 255, 0),      # Cyan
    (255, 0, 255),      # Magenta
    (139, 69, 19),      # Brown
    (0, 128, 128),      # Teal
    (0, 255, 0),        # Lime
    (75, 0, 130),       # Indigo
    (128, 0, 0),        # Maroon
    (128, 128, 0),      # Olive
    (128, 0, 0),        # Navy
    (64, 224, 208),     # Turquoise
    (255, 218, 185),    # Peach
    (230, 230, 250),    # Lavender
    (112, 128, 144)     # Slate Gray
]

def getColour(i_Index: int):
    return COLOURS[i_Index % len(COLOURS)]

def putText(a_Img: np.ndarray, s_Text: str, t_Loc: tuple, f_Scale: float = 1.0, i_Thickness: int = 1, t_Colour: tuple = (255,255,255)):
    cv.putText(a_Img, s_Text, t_Loc, cv.FONT_HERSHEY_DUPLEX, f_Scale, (0,0,0), 2*i_Thickness, cv.LINE_AA)
    cv.putText(a_Img, s_Text, t_Loc, cv.FONT_HERSHEY_DUPLEX, f_Scale, t_Colour, i_Thickness, cv.LINE_AA)
    return a_Img

def drawResults(a_Img: np.ndarray, c_ImageResults: ImageResults, _Size: tuple|int = None, b_DrawInferenceTime: bool = False):
    
    # Resize input image
    if isinstance(_Size,tuple):
        f_ResizeX = _Size[0] / a_Img.shape[1]
        f_ResizeY = _Size[1] / a_Img.shape[0]
    elif isinstance(_Size,int):
        f_ResizeX = _Size / max(a_Img.shape[:2])
        f_ResizeY = f_ResizeX
    else:
        f_ResizeX = 1.0
        f_ResizeY = 1.0

    a_Img = cv.resize(a_Img, (int(f_ResizeX*a_Img.shape[1]), int(f_ResizeY*a_Img.shape[0])))
    a_OverlayBboxes = np.zeros_like(a_Img)
    a_OverlayMasks = np.zeros_like(a_Img)
    
    # Draw objects
    for _Pred in c_ImageResults.get_predictions():
        
        t_Colour = getColour(_Pred.iClass)
        
        # Draw masks for instance segmentation
        if _Pred.Polygon.exists():
            a_Polygon = _Pred.get_polygon().scale_by(f_ResizeX,f_ResizeY).round().get_array()
            a_OverlayMasks = cv.drawContours(a_OverlayMasks, [a_Polygon], 0, t_Colour, -1, cv.LINE_AA)

        
        # Draw BBOX
        a_Bbox = _Pred.get_bbox().scale_by(f_ResizeX, f_ResizeY).round().get_xyxy()
        cv.rectangle(a_OverlayBboxes, (a_Bbox[0],a_Bbox[1]), (a_Bbox[2],a_Bbox[3]), t_Colour, 2, cv.LINE_AA) 
        
        # Put text
        s_Text = f"{_Pred.sClass}: {_Pred.fScore:.2f}"
        f_TextScale, i_TextThickness = 0.5, 1
        (w, h), _ = cv.getTextSize(s_Text, cv.FONT_HERSHEY_DUPLEX, f_TextScale, i_TextThickness)
        h += 8
        w += 4        
        cv.rectangle(a_OverlayBboxes, (a_Bbox[0],a_Bbox[1]), (a_Bbox[0]+w,a_Bbox[1]+h), t_Colour, -1)        
        a_OverlayBboxes = putText(a_OverlayBboxes, s_Text, (a_Bbox[0]+2,a_Bbox[1]+h-3), f_TextScale, i_TextThickness, (255,255,255))

    # Create final preview image
    a_Img = cv.addWeighted(
        a_Img, 1,
        a_OverlayMasks, .5, 
        0
    )
    a_Img = cv.addWeighted(
        a_Img, 1,
        a_OverlayBboxes, 1, 
        0
    ) 

    if b_DrawInferenceTime:
        a_Img = putText(a_Img, f"Processing time: {c_ImageResults.get_inference_time():.1f} ms", (0, a_Img.shape[0]-10), .75, 1, (255,255,255))
    
    return a_Img