"""
Preview inference and save output in YOLO and COCO format. Use your custom model (located in /models) and images from a chosen folder.
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
f_Thresh = 0.50

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
        s_PathWeights = os.path.join(ROOT_DIR,'models','model.pt'), # Load custom model from 'models' directory
        f_Thresh = f_Thresh, # Confidence threshold value
    )

    # Iterate through files
    for i, s_FileName in enumerate(l_ImageNames):
        # Load image
        a_Img = cv.imread(os.path.join(s_PathImages,s_FileName))

        # Inference - object detection
        c_ImageResults = c_Model.Detect(a_Img, s_ImageID=str(i))

        # Visualize results with opencv GUI
        a_Preview = drawResults(a_Img.copy(), c_ImageResults, _Size=1000)
        cv.imshow(os.path.basename(s_FileName), a_Preview)
        iKey = cv.waitKey(0)
        cv.destroyAllWindows()

        # Save output as YOLO .txt file
        saveResultsYolo(c_ImageResults, os.path.join(s_OutputPath, s_FileName.split('.')[0]+'.txt'))
        # Save output as COCO .json file
        saveResultsCoco(c_ImageResults, os.path.join(s_OutputPath, s_FileName.split('.')[0]+'.json'))

        if iKey == 27: 
            # Press 'ESC' to exit
            break
        elif iKey == 115: 
            # Press 's' to save preview image
            cv.imwrite(os.path.join(s_OutputPath, s_FileName.split('.')[0]+'.jpg'), a_Preview)
            print("Preview image saved.")