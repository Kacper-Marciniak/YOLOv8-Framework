"""
Preview inference from camera
"""

import cv2 as cv
import time
import numpy as np

from ml_model.CModelSAM import CModelSAM as Model
from results.visualization import drawResultsSAM
from camera.CCamera import CCamera as Camera

# Target FPS value
i_TargetFPS = 10
# Confidence threshold value
f_Thresh = 0.75

def on_change(value): 
    global i_InputRadius
    i_InputRadius = max(1,value)

if __name__ == "__main__":
    # Initialize camera
    CCamera = Camera()    

    # Initialize model
    c_Model = Model(
        f_Thresh = f_Thresh, # Confidence threshold value
    )

    while(True):
        # Count frame time
        f_Time = time.time()
        # Get new frame
        a_Img = CCamera.grabFrame()

        l_Points = [
            [a_Img.shape[1]//2,a_Img.shape[0]//2]
        ]

        # Inference - object detection        
        c_ImageResults = c_Model.Detect(a_Img, l_Points=l_Points, b_PrintOutput=False)

        # Visualize results with opencv GUI
        a_Preview = drawResultsSAM(a_Img.copy(), c_ImageResults, l_Points, _Size=1000, b_DrawInferenceTime=True)
        cv.imshow("Camera", a_Preview)
        
        f_Time = (time.time() - f_Time)*1000.0
        f_Time = max(1,int(1000.0/i_TargetFPS-f_Time))
        
        cv.waitKey(f_Time)

        # Break from loop when openCV window is closed
        if not cv.getWindowProperty("Camera", cv.WND_PROP_VISIBLE): break
    
    try: 
        CCamera.close()
    except:
        pass
    CCamera = None