"""
Use this script to run the SAM model on a camera feed and segment objects by drawing bounding boxes on the screen.
"""

import cv2 as cv
import numpy as np

from ml_model.CModelSAM import CModelSAM as ModelSAM
from results.visualization import drawResults
from camera.CCamera import CCamera as Camera

# Target FPS value
i_TargetFPS = 10
# Confidence threshold value
f_Thresh = 0.25

global_bbox = []
global_mouse = [0,0]

def get_click(event,x,y,flags,param):
    global global_bbox, global_mouse
    if event == cv.EVENT_LBUTTONUP:
        global_bbox += [x,y]
    global_mouse = [x,y]
    

if __name__ == "__main__":
    # Initialize camera
    CCamera = Camera()    

    # Initialize model
    c_Model = ModelSAM(
        f_Thresh = f_Thresh, # Confidence threshold value
    )

    # Initialize window
    cv.namedWindow('Camera')
    cv.setMouseCallback('Camera', get_click)

    # Get first frame
    a_Img = CCamera.grabFrame()
    c_Model.setImage(a_Img)

    while(cv.getWindowProperty("Camera", cv.WND_PROP_VISIBLE)):

        # Get bounding box
        while len(global_bbox)<4:
            a_Preview = a_Img.copy()
            
            if len(global_bbox)==2: 
                cv.circle(a_Preview, (global_bbox[0], global_bbox[1]), 3, (0,255,0), -1)
                cv.line(a_Preview, (0,global_bbox[1]), (a_Img.shape[1],global_bbox[1]), (0,255,0), 1)
                cv.line(a_Preview, (global_bbox[0],0), (global_bbox[0],a_Img.shape[0]), (0,255,0), 1)
            cv.circle(a_Preview, global_mouse, 3, (255,255,255), -1)
            cv.line(a_Preview, (0,global_mouse[1]), (a_Img.shape[1],global_mouse[1]), (255,255,255), 1)
            cv.line(a_Preview, (global_mouse[0],0), (global_mouse[0],a_Img.shape[0]), (255,255,255), 1)

            cv.imshow("Camera", a_Preview)
            sKey = cv.waitKey(50)

            if not cv.getWindowProperty("Camera", cv.WND_PROP_VISIBLE) or sKey != -1: break
        else:
            # Inference - object detection        
            c_ImageResults = c_Model.Segment(a_Img, l_Bbox=global_bbox, b_PrintOutput=False)

            # Visualize results with opencv GUI
            a_Preview = drawResults(a_Img.copy(), c_ImageResults, b_NoLabels=True)

            # Segment object
            if c_ImageResults.get_n_predictions():
                a_Polygon = c_ImageResults.get_predictions()[0].get_polygon().round().get_array()
                x,y,w,h = cv.boundingRect(a_Polygon)
                # Segment
                a_Mask = np.zeros((a_Img.shape[0],a_Img.shape[1]))
                a_Mask = cv.drawContours(a_Mask, [a_Polygon], -1, 255, -1)
                a_Segment = a_Img.copy()
                a_Segment[a_Mask==0] = 0
                a_Segment = a_Segment[y:y+h, x:x+w]
                
                cv.imshow("Segment", a_Segment)

            cv.imshow("Camera", a_Preview)        
            sKey = cv.waitKey(0)

        # Break from loop if ESC is pressed
        if sKey == 27: 
            break
        # Get new frame if SPACE is pressed
        elif sKey == ord(' '):
            c_Model.resetImage()       
            a_Img = CCamera.grabFrame()
            c_Model.setImage(a_Img)
            
        global_bbox = []
        if cv.getWindowProperty("Segment", cv.WND_PROP_VISIBLE):
            cv.destroyWindow("Segment")