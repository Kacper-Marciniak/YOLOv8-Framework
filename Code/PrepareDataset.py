"""
Create the dataset needed for model training from labeled set of images
"""

import os

from dataset_prep.createDataset import prepareDataset
from utility.GUI import askdirectory
from path.root import ROOT_DIR

if __name__ == "__main__":
    # Select input folder with images and labels
    s_InputPath = askdirectory("Select input folder with images and labels" , initialdir=os.path.join(ROOT_DIR,'datasets'))
    # Select output dataset folder, f.e: 'DAT-1'
    s_OutputDatasetPath = askdirectory("Select output dataset folder", initialdir=os.path.join(ROOT_DIR,'datasets'))

    if '' in [s_InputPath, s_OutputDatasetPath]: raise Exception("Not a valid path")

    # Run dataset preparation script
    prepareDataset(s_InputPath, s_OutputDatasetPath)