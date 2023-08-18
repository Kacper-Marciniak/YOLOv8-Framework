import cv2 as cv
import numpy as np
from utility.BboxOperations import *


def drawResults(a_Img: np.ndarray, dc_Results: dict, t_Size: tuple = (None,None)):
    if t_Size != (None,None):
        f_ResizeX = t_Size[0] / a_Img.shape[1]
        f_ResizeY = t_Size[1] / a_Img.shape[0]

        a_Img = cv.resize(a_Img, (int(f_ResizeX*a_Img.shape[1]), int(f_ResizeY*a_Img.shape[0])))
    
    else:
        f_ResizeX = 1
        f_ResizeY = 1
    
    for a_Bbox, i_ClassID, f_Score in zip(dc_Results['bbox'],dc_Results['class'], dc_Results['score']):

        a_Bbox = np.round(scale(a_Bbox, f_ResizeX, f_ResizeY)).astype(int)

        cv.rectangle(a_Img, (a_Bbox[0],a_Bbox[1]), (a_Bbox[2],a_Bbox[3]), (0,0,0), 2) 
        cv.rectangle(a_Img, (a_Bbox[0],a_Bbox[1]), (a_Bbox[2],a_Bbox[3]), (255,255,255), 1)

        s_Text = f"{i_ClassID}: {f_Score:.2f}"
        f_TextScale, i_TextThickness = 0.5, 1
        (w, h), _ = cv.getTextSize(s_Text, cv.FONT_HERSHEY_COMPLEX, f_TextScale, i_TextThickness)
        h += 6
        w += 4
        
        cv.rectangle(a_Img, (a_Bbox[0],a_Bbox[1]-2), (a_Bbox[0]+w,a_Bbox[1]-h-2), (0,0,0), -1)
        cv.putText(a_Img, s_Text, (a_Bbox[0]+2,a_Bbox[1]-3), cv.FONT_HERSHEY_COMPLEX, f_TextScale, (255,255,255), i_TextThickness)

    return a_Img