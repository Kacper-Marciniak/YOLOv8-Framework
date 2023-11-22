"""
1. Create the dataset needed for model training from labeled set of images.
2. Split into N segments.
3. Train YOLOv8 object detection models.
4. Evaluate.
"""

import os, json
from dataset_prep.createDataset import prepareCrossEvalData
from path.root import ROOT_DIR
from utility.GUI import askdirectory, choosefromlist
from training.CTrainer import CTrainer
from parameters.TrainingParameters import AVAILABLE_MODELS
from validation.validateSystem import Validate
from datetime import datetime
import numpy as np
import gc

# Data preparation
iNSegments = 5

# Training
i_Epochs = 25
f_ConfThreshTest = 0.25
i_BatchSize = 8

# Eval
f_EvalThresh = .25

if __name__ == "__main__":

    # Select input folder with images and labels
    s_InputPath = askdirectory("Select input folder with images and labels" , initialdir=os.path.join(ROOT_DIR,'datasets'))
    # Select output dataset folder, f.e: 'DAT-1'
    s_OutputDatasetPath = askdirectory("Select output dataset folder", initialdir=os.path.join(ROOT_DIR,'datasets'))
    # Choose model type
    s_ModelName = choosefromlist(AVAILABLE_MODELS, title="Choose model size", width=75)
    # Set output directory
    sResultsFolder = "CrossEval_"+str(datetime.now().strftime("%Y%m%d_%H%M%S")).replace(' ','_')
    os.makedirs(os.path.join(ROOT_DIR, 'validation_results', sResultsFolder))

    if '' in [s_InputPath, s_OutputDatasetPath]: raise Exception("Not a valid path")

    # Run dataset preparation script
    prepareCrossEvalData(s_InputPath, s_OutputDatasetPath, iNSegments)

    lDatasets = [f"dataset_{i}" for i in range(iNSegments)]

    for i,sData in enumerate(lDatasets):
        s_DatasetDir = os.path.join(s_OutputDatasetPath,sData)
        try:
            # Initialize trainer class
            c_Trainer = CTrainer(
                s_ModelName = s_ModelName,
                s_DatasetDirectory = s_DatasetDir
            )
            # Train model
            c_Trainer.Train(i_Epochs, i_BatchSize)
            # Copy data configuration 
            c_Trainer._CopyDataConfig()

            # Run validation 
            dc_Results = Validate(  
                s_DatasetDir,
                f_Thresh = f_EvalThresh,
                b_SaveOutput = False,
                sCustomWeightsPath = os.path.join(c_Trainer.s_TrainingOutputPath,'weights','best.pt')
            )

            with open(os.path.join(ROOT_DIR, 'validation_results', sResultsFolder, f'results_{i}.json'),'w') as _File:
                json.dump(dc_Results, _File, indent=4)
                _File.close()

            del(c_Trainer)
            gc.collect()

            print(f"All tasks finished for training {i+1}/{iNSegments}!")
        except Exception as e:
            print(f"\n\nTraining {i+1}/{iNSegments} stopped. Error message:\n{e}")

    dc_Results = {}
    for i in range(iNSegments):
        with open(os.path.join(ROOT_DIR, 'validation_results', sResultsFolder, f'results_{i}.json'),'r') as _File:
            _Results = json.load(_File)
            _File.close()
        for sKey in _Results:
            if isinstance(_Results[sKey], dict):
                for sClassName in _Results[sKey].keys():
                    if not (f"{sKey}_{sClassName}" in list(dc_Results.keys())): dc_Results[f"{sKey}_{sClassName}"] = []
                    dc_Results[f"{sKey}_{sClassName}"].append(_Results[sKey][sClassName])
            else:
                if not (sKey in list(dc_Results.keys())): dc_Results[sKey] = []
                dc_Results[sKey].append(_Results[sKey])
    
    dc_OutputData = {}

    for sKey in dc_Results:
        lData = dc_Results[sKey]
        dc_OutputData[sKey] = {
            "mean": np.mean(lData),
            "std": np.std(lData)
        }

    with open(os.path.join(ROOT_DIR, 'validation_results', sResultsFolder, f'results_final.json'),'w') as _File:
        json.dump(
            dc_OutputData,
            _File, 
            indent=4)
        _File.close()
    
    print(os.path.join(ROOT_DIR, 'validation_results', sResultsFolder, f'results_final.json'))