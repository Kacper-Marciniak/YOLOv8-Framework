"""
Functions used in dataset creation
"""

from tqdm import tqdm
import numpy as np
import os, shutil, time, glob
import yaml
import random

from parameters.parameters import *

SEED = int(time.time())

def _splitData(s_InputDir: str, s_OutputDir: str, _Seed = None) -> None:
    """
    Split data into 'train', 'val' and 'test' subsets
    """

    if _Seed != None: 
        np.random.seed(_Seed)
        random.seed = _Seed

    l_ListLabels = [os.path.splitext(s_FilePath)[0] for s_FilePath in glob.glob(os.path.join(s_InputDir, '*.txt'))]
    l_ListImages = [s_FilePath for s_FilePath in glob.glob(os.path.join(s_InputDir, '*.*'))
                    if os.path.splitext(s_FilePath)[1].lower().strip('.') in ALLOWED_INPUT_FILES
                    and os.path.splitext(s_FilePath)[0] in l_ListLabels ]
    del(l_ListLabels)

    random.shuffle(l_ListImages)

    l_ImagesTest = l_ListImages[:round(len(l_ListImages)*SPLIT_TEST)]
    l_ImagesVal = l_ListImages[round(len(l_ListImages)*SPLIT_TEST):round(len(l_ListImages)*SPLIT_TEST)+round(len(l_ListImages)*SPLIT_VAL)]
    l_ImagesTrain = l_ListImages[round(len(l_ListImages)*SPLIT_TEST)+round(len(l_ListImages)*SPLIT_VAL):]

    del(l_ListImages)

    for _Folder, _Images in zip(['train','val','test'], [l_ImagesTrain, l_ImagesVal, l_ImagesTest]):
        _PathOut = os.path.join(s_OutputDir, _Folder)
        os.makedirs(_PathOut, exist_ok=True)

        for s_ImagePath in tqdm(_Images, desc=f"Moving files [{_Folder}]", unit="file", total=len(_Images)):
            s_ImageName = os.path.basename(s_ImagePath)
            shutil.copy(
                s_ImagePath, 
                os.path.join(_PathOut, s_ImageName)
            )
            shutil.copy(
                os.path.splitext(s_ImagePath)[0]+'.txt',
                os.path.join(_PathOut, os.path.splitext(s_ImageName)[0]+'.txt')
            )            

        print("All files moved!")

def _prepareConfigFile(s_InputDir: str, s_OutputDir: str, b_NoTestDir: bool = False) -> None:
    """
    Prepare dataset YAML configuration file
    """

    s_OutputDir = os.path.abspath(s_OutputDir).replace('\\','/').replace('//','/')

    l_ClassNames = _loadClassNames(s_InputDir)

    dc_DatasetConfig = {
            'path': s_OutputDir,
            'train': 'train',
            'val': 'val',
            'test': 'test' if not b_NoTestDir else 'val',
            'nc': len(l_ClassNames), # Number of classes
            'names': l_ClassNames, # Class names
        }

    with open(os.path.join(s_OutputDir, 'data.yaml'), "w") as _File:
        yaml.dump(dc_DatasetConfig, _File, default_flow_style=False)

def _loadClassNames(s_InputDir: str):
    """
    Load class data
    """

    s_InputDir = os.path.abspath(s_InputDir)
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
    _splitData(os.path.abspath(os.path.join(s_InputDir,'data')), s_OutputDir)

def prepareCrossEvalData(s_InputDir: str, s_OutputDir: str, iNSegments: int, _Seed = None):
    """
    Split data into N segments.
    """
    if not os.path.exists(s_OutputDir): raise Exception("Output path doesn't exist")

    s_OutputDir = os.path.abspath(s_OutputDir).replace('\\','/').replace('//','/')

    if _Seed != None: 
        np.random.seed(_Seed)
        random.seed = _Seed

    s_InputDir = os.path.abspath(os.path.join(s_InputDir, 'data'))
    l_ListLabels = [os.path.splitext(s_FilePath)[0] for s_FilePath in glob.glob(os.path.join(s_InputDir, '*.txt'))]
    l_ListImages = [s_FilePath for s_FilePath in glob.glob(os.path.join(s_InputDir, '*.*'))
                    if os.path.splitext(s_FilePath)[1].lower().strip('.') in ALLOWED_INPUT_FILES
                    and os.path.splitext(s_FilePath)[0] in l_ListLabels ]
    del(l_ListLabels)

    random.shuffle(l_ListImages)

    for i in tqdm(range(iNSegments), desc=f"Splitting data", unit="segment", total=iNSegments):
    
        lImages = l_ListImages[i*iNSegments:(i+1)*iNSegments]

        sPathSegment = os.path.join(s_OutputDir, f"segment_{i}")
        if os.path.exists(sPathSegment): shutil.rmtree(sPathSegment)
        os.makedirs(sPathSegment, exist_ok=True)

        for s_ImagePath in lImages:
            s_ImageName = os.path.basename(s_ImagePath)
            shutil.copy(
                s_ImagePath, 
                os.path.join(sPathSegment, s_ImageName)
            )
            shutil.copy(
                os.path.splitext(s_ImagePath)[0]+'.txt',
                os.path.join(sPathSegment, os.path.splitext(s_ImageName)[0]+'.txt')
            )  

    print("Segments prepared!")

    print(f"Preparing {iNSegments} datasets.")

    for i in tqdm(range(iNSegments), desc=f"Preparing datasets", unit="dataset", total=iNSegments):
    
        s_PathData = os.path.join(s_OutputDir, f"dataset_{i}")
        if os.path.exists(s_PathData): shutil.rmtree(s_PathData)
        os.makedirs(s_PathData)
        for sFolder in ('train','val'): os.makedirs(os.path.join(s_PathData,sFolder))

        shutil.copytree(
            os.path.join(s_OutputDir, f"segment_{i}"),
            os.path.join(s_PathData, sFolder),
            dirs_exist_ok=True
        )
        for j in range(iNSegments):
            if j!=i:
                shutil.copytree(
                    os.path.join(s_OutputDir, f"segment_{j}"),
                    os.path.join(s_PathData,'train'),
                    dirs_exist_ok=True
                )

        _prepareConfigFile(os.path.dirname(s_InputDir), s_PathData, b_NoTestDir=True)

    for i in tqdm(range(iNSegments), desc=f"Cleaning segments", unit="segment", total=iNSegments):
        shutil.rmtree(os.path.join(s_OutputDir, f"segment_{i}"))

    print("Datasets prepared!")
