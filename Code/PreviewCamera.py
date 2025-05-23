"""
Preview inference from camera and segment objects using YOLOv8n-seg model
"""

import cv2 as cv
import time

from ml_model.CModelML import CModelML as Model
from results.visualization import drawResults
from camera.CCamera import CCamera as Camera

# Target FPS value
i_TargetFPS = 10
# Confidence threshold value
f_Thresh = 0.25

if __name__ == "__main__":
    # Initialize camera
    CCamera = Camera()
    CCamera.setResolution(1280,720)

    # Initialize YOLO for instance segmentation
    c_Model = Model(
        s_PathWeights = 'yolo11n-seg.pt', # Load YOLOv8n segmentation model
        f_Thresh = f_Thresh, # Confidence threshold value
        b_MobileSAMPostProcess = False, # Disable SAM post-processing and segmentation
    )
    # or
    # Initialize RT-DETR for object detection and SAM for segmentation
    """c_Model = Model(
        s_PathWeights = 'rtdetr-l.pt', # Load YOLOv8n segmentation model
        f_Thresh = f_Thresh, # Confidence threshold value
        b_MobileSAMPostProcess = True, # Enable SAM post-processing and segmentation
        s_ModelArchitectureType = 'rtdetr', # Set model architecture type to RT-DETR
    )"""

    while(True):
        # Count frame time
        f_Time = time.time()
        # Get new frame
        a_Img = CCamera.grabFrame()

        # Inference - object detection
        c_ImageResults = c_Model.Detect(a_Img)

        # Visualize results with opencv GUI
        a_Preview = drawResults(a_Img.copy(), c_ImageResults, _Size=1000, b_DrawInferenceTime=True)
        cv.imshow("Camera", a_Preview)
                
        sKey = cv.waitKey(max(1,int(1000.0/i_TargetFPS-((time.time() - f_Time)*1000.0))))

        # Break from loop when OpenCV window is closed or ESC is pressed
        if not cv.getWindowProperty("Camera", cv.WND_PROP_VISIBLE) or sKey == 27: break
    
    # Release camera and close OpenCV windows
    CCamera.close()
    cv.destroyAllWindows()