"""
Preview inference and save output in YOLO and COCO format
"""

import os
import cv2 as cv

from ml_model.CModelML import CModelML as Model
from path.root import ROOT_DIR
from parameters.parameters import ALLOWED_INPUT_FILES
from utility.GUI import askdirectory
from results.visualization import drawResults
from results.output import saveResultsYolo, saveResultsCoco

# Confidence threshold value
f_Thresh = 0.75

if __name__ == "__main__":
    # Select directory with input images
    s_PathImages = askdirectory("Select folder with images", initialdir=os.path.join(ROOT_DIR,'datasets'))
    # Load list of images
    l_ImageNames = [s_FileName for s_FileName in os.listdir(s_PathImages) if s_FileName.lower().split('.')[-1] in ALLOWED_INPUT_FILES]
    print(f"{len(l_ImageNames)} images in folder. Starting preview inference...")

    # Get output path
    s_OutputPath = os.path.join(ROOT_DIR, 'inference_output')

    # Initialize model
    c_Model = Model(
        s_PathWeights = os.path.join(ROOT_DIR,'models','model.pt'), # Default model path
        f_Thresh = f_Thresh # Confidence threshold value
    )

    # Iterate through files
    for i, s_FileName in enumerate(l_ImageNames):
        # Load image
        a_Img = cv.imread(os.path.join(s_PathImages,s_FileName))

        # Inference - object detection
        dc_Results = c_Model.Detect(a_Img)

        # Visualize results with opencv GUI
        a_Preview = drawResults(a_Img.copy(), dc_Results, (800,800))
        cv.imshow(os.path.basename(s_FileName), a_Preview)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Save output as YOLO .txt file
        saveResultsYolo(dc_Results, os.path.join(s_OutputPath, s_FileName.split('.')[0]+'.txt'))
        # Save output as COCO .json file
        saveResultsCoco(dc_Results, os.path.join(s_OutputPath, s_FileName.split('.')[0]+'.json'), i_ImageID = i)