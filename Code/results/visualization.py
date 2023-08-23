import cv2 as cv
import numpy as np
from utility.BboxOperations import *

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

def drawResults(a_Img: np.ndarray, dc_Results: dict, _Size: tuple|int = None, b_DrawInferenceTime: bool = False):
    
    # Resize input image
    if isinstance(_Size,tuple):
        f_ResizeX = _Size[0] / a_Img.shape[1]
        f_ResizeY = _Size[1] / a_Img.shape[0]
    elif isinstance(_Size,int):
        f_ResizeX = _Size / max(a_Img.shape[:2])
        f_ResizeY = f_ResizeX
    else:
        f_ResizeX = 1
        f_ResizeY = 1

    a_Img = cv.resize(a_Img, (int(f_ResizeX*a_Img.shape[1]), int(f_ResizeY*a_Img.shape[0])))
    a_OverlayBboxes = np.zeros_like(a_Img)
    a_OverlayMasks = np.zeros_like(a_Img)
        
    
    # Draw objects
    for i, (i_ClassID, f_Score) in enumerate(zip(dc_Results['class'], dc_Results['score'])):
        
        t_Colour = getColour(i)
        
        # Draw masks for instance segmentation
        if dc_Results['task'] == 'segment':
            a_Polygon = dc_Results['polygon'][i].copy()

            a_Polygon[:,0], a_Polygon[:,1] = a_Polygon[:,0]*f_ResizeX, a_Polygon[:,1]*f_ResizeY

            a_Mask = cv.drawContours(np.zeros((a_OverlayMasks.shape[0],a_OverlayMasks.shape[1],1),dtype=np.uint8), [a_Polygon], 0, 255, -1, cv.LINE_AA)
            a_TmpMask = np.zeros_like(a_OverlayMasks)
            a_TmpMask[a_Mask[:,:,0]>0] = t_Colour
            
            a_Indices1, a_Indices2 = np.sum(a_OverlayMasks,2)>0, np.sum(a_TmpMask,2)>0
            a_Indices = np.logical_and(a_Indices1,a_Indices2)
            a_OverlayMasks[a_Indices] = np.clip(np.round(a_TmpMask[a_Indices]*0.50+a_OverlayMasks[a_Indices]*0.50).astype(np.uint8),0,255)
            a_Indices = np.logical_and(np.logical_not(a_Indices1),a_Indices2)
            a_OverlayMasks[a_Indices] = a_TmpMask[a_Indices]
            
            del(a_Indices1)
            del(a_Indices2)
            del(a_Indices)
            del(a_TmpMask)

        
        # Draw BBOX
        a_TmpMask = np.zeros_like(a_OverlayMasks)
        a_Bbox = np.round(scale(dc_Results['bbox'][i], f_ResizeX, f_ResizeY)).astype(int)
        cv.rectangle(a_TmpMask, (a_Bbox[0],a_Bbox[1]), (a_Bbox[2],a_Bbox[3]), t_Colour, 2, cv.LINE_AA) 
        
        # Put text
        s_Text = f"{dc_Results['names'][i_ClassID]}: {f_Score:.2f}"
        f_TextScale, i_TextThickness = 0.5, 1
        (w, h), _ = cv.getTextSize(s_Text, cv.FONT_HERSHEY_DUPLEX, f_TextScale, i_TextThickness)
        h += 6
        w += 4        
        cv.rectangle(a_TmpMask, (a_Bbox[0],a_Bbox[1]), (a_Bbox[0]+w,a_Bbox[1]+h), t_Colour, -1)
        cv.putText(a_TmpMask, s_Text, (a_Bbox[0]+2,a_Bbox[1]+h-3), cv.FONT_HERSHEY_DUPLEX, f_TextScale, (1,1,1), i_TextThickness*2, cv.LINE_AA)
        cv.putText(a_TmpMask, s_Text, (a_Bbox[0]+2,a_Bbox[1]+h-3), cv.FONT_HERSHEY_DUPLEX, f_TextScale, (255,255,255), i_TextThickness, cv.LINE_AA)

        a_Indices1, a_Indices2 = np.sum(a_OverlayBboxes,2)>0, np.sum(a_TmpMask,2)>0
        a_Indices = np.logical_and(a_Indices1,a_Indices2)
        a_OverlayBboxes[a_Indices] = np.clip(np.round(a_TmpMask[a_Indices]*0.50+a_OverlayBboxes[a_Indices]*0.50).astype(np.uint8),0,255)
        a_Indices = np.logical_and(np.logical_not(a_Indices1),a_Indices2)
        a_OverlayBboxes[a_Indices] = a_TmpMask[a_Indices]
            
        del(a_Indices1)
        del(a_Indices2)
        del(a_Indices)
        del(a_TmpMask)

    # Create final preview image
    a_Indices = np.sum(a_OverlayMasks,axis=2)>0  
    a_Img[a_Indices] = np.clip(np.round(a_Img[a_Indices]*0.50+a_OverlayMasks[a_Indices]*0.50).astype(np.uint8),0,255)

    a_Indices = np.sum(a_OverlayBboxes,axis=2)>0  
    a_Img[a_Indices] = np.clip(np.round(a_Img[a_Indices]*0.35+a_OverlayBboxes[a_Indices]*0.65).astype(np.uint8),0,255)

    if b_DrawInferenceTime:
        cv.putText(a_Img, f"Processing time: {dc_Results['time']:.1f}", (0,a_Img.shape[0]), cv.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0), 2, cv.LINE_AA)
        cv.putText(a_Img, f"Processing time: {dc_Results['time']:.1f}", (0,a_Img.shape[0]), cv.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 1, cv.LINE_AA)
    
    return a_Img