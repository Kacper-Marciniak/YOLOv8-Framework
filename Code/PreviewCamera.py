"""
Preview inference from camera
"""

import os
import cv2 as cv
import time

from ml_model.CModelML import CModelML as Model
from path.root import ROOT_DIR
from results.visualization import drawResults
from camera.CCamera import CCamera as Camera

# Target FPS value
i_TargetFPS = 10
# Confidence threshold value
f_Thresh = 0.25

if __name__ == "__main__":
    # Initialize camera
    CCamera = Camera()    

    # Initialize model
    c_Model = Model(
        s_PathWeights = os.path.join(ROOT_DIR,'models','model.pt'), # Load custom model from 'models' directory
        f_Thresh = f_Thresh # Confidence threshold value
    )

    while(True):
        # Count frame time
        f_Time = time.time()
        # Get new frame
        a_Img = CCamera.grabFrame()

        # Inference - object detection
        dc_Results = c_Model.Detect(a_Img)

        # Visualize results with opencv GUI
        a_Preview = drawResults(a_Img.copy(), dc_Results, _Size=1000, b_DrawInferenceTime=True)
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