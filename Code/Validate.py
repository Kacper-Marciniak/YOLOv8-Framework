"""
Run system validation
"""

from validation.validateSystem import Validate
from utility.GUI import askdirectory, askopenfilename
import os
from path.root import ROOT_DIR

# Confidence threshold value
f_Thresh = 0.25

if __name__ == "__main__":
    # Set path to test images
    s_DatasetPath = askdirectory("Select dataset folder", initialdir=os.path.join(ROOT_DIR,'datasets'))
    s_ModelPath = askopenfilename("Select model weights file", initialdir=os.path.join(ROOT_DIR,'training_output'))
    # Run validation 
    dc_Results = Validate(  
        s_DatasetPath,
        f_Thresh = f_Thresh,
        b_SaveOutput = True,
        sCustomWeightsPath = s_ModelPath
    )