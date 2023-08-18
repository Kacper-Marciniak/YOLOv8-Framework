"""
Run system validation
"""

from validation.validateSystem import Validate
from utility.GUI import askdirectory
import os
from path.root import ROOT_DIR

if __name__ == "__main__":
    # Set path to test images
    s_DatasetPath = askdirectory("Select dataset folder", initialdir=os.path.join(ROOT_DIR,'datasets'))
    # Run validation 
    dc_Results = Validate(  
        s_DatasetPath,
        f_Thresh = 0.75,
        b_SaveOutput = True
    )