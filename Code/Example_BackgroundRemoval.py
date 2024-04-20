"""
Example: bluring background from webcam feed
"""

import cv2 as cv
import time

from ml_model.CModelML import CModelML as Model
from camera.CCamera import CCamera as Camera
import numpy as np

# Target FPS
i_TargetFPS = 30
# Confidence threshold value
f_Thresh = 0.50

if __name__ == "__main__":
    # Initialize camera
    CCamera = Camera(0)

    # Initialize model
    c_Model = Model(
        s_PathWeights = 'yolov8l-seg.pt', # YOLOv8-Large detection model trained on COCO dataset
        f_Thresh = f_Thresh, # Confidence threshold value
    )

    while(True):
        # Count frame time
        f_Time = time.time()
        # Get new frame
        a_Img = CCamera.grabFrame()

        # Inference - object detection
        c_Results = c_Model.Detect(a_Img, b_PrintOutput = False)

        c_Results.get_predictions_by_class("person")

        # Get all instances of 'person' class
        l_PersonContours = [_pred.get_polygon() for _pred in c_Results.get_predictions_by_class("person")]

        # Create mask
        a_Mask = np.zeros((a_Img.shape[0],a_Img.shape[1],1))
        a_Mask = cv.drawContours(a_Mask, l_PersonContours, -1, 255, -1)
        a_Mask = cv.morphologyEx(a_Mask, cv.MORPH_CLOSE, (15,15))
       
        # Blur image
        t_KernelImage = (31,31)
        a_ImgBlur = cv.blur(a_Img.copy(), t_KernelImage)
        a_ImgBlur = cv.cvtColor(cv.cvtColor(a_ImgBlur, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

        # Blur mask
        t_KernelMask = (9,9)
        a_Mask = np.expand_dims((255-cv.blur(a_Mask, t_KernelMask)).astype(float)/255.0,-1)

        # Combine blurred background and foregorund
        a_Img = np.clip(a_ImgBlur*a_Mask+a_Img*(1.0-a_Mask), 0, 255).astype(np.uint8)

        # Display frame
        cv.imshow("Camera", a_Img)        

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