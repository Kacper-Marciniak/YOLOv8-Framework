"""
Functions used in dataset creation
"""

import cv2 as cv
import numpy as np
import os, shutil, time
import yaml

from parameters.parameters import *

SEED = int(time.time())

def _splitData(s_InputDir: str, s_OutputDir: str, _Seed = None) -> None:
    """
    Split data into 'train', 'val' and 'test' subsets
    """

    if _Seed != None: np.random.seed(_Seed)

    l_ListLabels = [s_FileName.lower().split('.')[0] for s_FileName in os.listdir(s_InputDir)
                    if s_FileName.lower().split('.')[-1] == 'txt']

    l_ListImages = [s_FileName.lower() for s_FileName in os.listdir(s_InputDir)
                    if s_FileName.lower().split('.')[-1] in ALLOWED_INPUT_FILES 
                    and s_FileName.lower().split('.')[0] in l_ListLabels]

    del(l_ListLabels)

    l_ImagesTest = l_ListImages[:round(len(l_ListImages)*SPLIT_TEST)]
    l_ImagesVal = l_ListImages[round(len(l_ListImages)*SPLIT_TEST):round(len(l_ListImages)*SPLIT_TEST)+round(len(l_ListImages)*SPLIT_VAL)]
    l_ImagesTrain = l_ListImages[round(len(l_ListImages)*SPLIT_TEST)+round(len(l_ListImages)*SPLIT_VAL):]

    del(l_ListImages)

    for _Folder, _Images in zip(['train','val','test'], [l_ImagesTrain, l_ImagesVal, l_ImagesTest]):
        _PathOut = os.path.join(s_OutputDir, _Folder)
        os.makedirs(_PathOut, exist_ok=True)

        for i,s_ImageName in enumerate(_Images):
            print(f"[{_Folder.upper()} {i+1}/{len(_Images)}] Moving image and label: {s_ImageName.split('.')[0]}")
            shutil.copy(
                os.path.join(s_InputDir, s_ImageName), 
                os.path.join(_PathOut, s_ImageName)
            )
            shutil.copy(
                os.path.join(s_InputDir, s_ImageName.split('.')[0]+'.txt'), 
                os.path.join(_PathOut, s_ImageName.split('.')[0]+'.txt')
            )            

        print("All files moved!")

def _prepareConfigFile(s_InputDir: str, s_OutputDir: str):
    """
    Prepare dataset YAML configuration file
    """

    s_OutputDir = os.path.abspath(s_OutputDir).replace('\\','/').replace('//','/')

    l_ClassNames = _loadClassNames(s_InputDir)

    dc_DatasetConfig = {
            'path': s_OutputDir,
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(l_ClassNames), # Number of classes
            'names': l_ClassNames, # Class names
        }

    with open(os.path.join(s_OutputDir, 'data.yaml'), "w") as _File:
        yaml.dump(dc_DatasetConfig, _File, default_flow_style=False)

def _loadClassNames(s_InputDir: str):
    """
    Load class data
    """

    s_InputDir = os.path.abspath(s_InputDir).replace('\\','/').replace('//','/')

    with open(os.path.join(s_InputDir, 'class_names.txt'), "r") as _File:
        l_ClassNames = [_Line.strip(' \t\n') for _Line in _File.readlines()]
    return l_ClassNames

def prepareDataset(s_InputDir: str, s_OutputDir: str):
    """
    Prepare dataset. Create output directory, write YAML configuration, split data into train, val and test subsets.
    """
    if not os.path.exists(s_OutputDir): raise Exception("Output path doesn't exist")
    if not os.path.exists(os.path.join(s_InputDir,'data')): raise Exception("Input data path doesn't exist")


    _prepareConfigFile(s_InputDir, s_OutputDir)
    _splitData(os.path.join(s_InputDir,'data'), s_OutputDir)