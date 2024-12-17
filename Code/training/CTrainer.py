"""
Trainer class for ML model training
"""

import os, torch, gc, shutil
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

from parameters.TrainingParameters import *
from parameters.parameters import ALLOWED_INPUT_FILES
from parameters.TrainingParameters import AVAILABLE_MODELS

from path.root import ROOT_DIR

class CTrainer():
    def __init__(self, s_ModelConfig: str, s_DatasetDirectory: str):
        s_ModelConfig = s_ModelConfig.lower()
        s_ModelName = os.path.basename(s_ModelConfig).split('.')[0]

        # Get dataset directory
        self._GetDatasetDir(s_DatasetDirectory)
        # Clear cache and remove tmp files
        self._ClearCacheDataset()

        # Get project and run names
        self.s_TrainingOutputPath = os.path.join(ROOT_DIR, r'training_output', str(datetime.now().strftime("%Y%m%d_%H%M%S")).replace(' ','_')+f'_{s_ModelName}')
        
        # Create output directory
        os.makedirs(self.s_TrainingOutputPath, exist_ok=True)

        # Clear memory/cache
        gc.collect()
        torch.cuda.empty_cache()

        # Initialize YOLO architecture
        if not (s_ModelConfig.split('.')[-1].lower() in ['pt','yaml']):
            if s_ModelConfig in AVAILABLE_MODELS:
                s_ModelConfig += '.pt'
            else:
                s_ModelConfig += '.yaml'
        
        print(f"Model configuraton: {s_ModelConfig}")

        self.c_Model = YOLO(s_ModelConfig)

        print(f"Model {s_ModelName} initialized!")

    def _GetDatasetDir(self, s_DatasetDirectory: str):
        """
        Get dataset directory
        """
        if os.path.exists(s_DatasetDirectory):
            self.s_DatasetDirectory = s_DatasetDirectory
        else: 
            raise Exception("Dataset does not exist")
        
        if not os.path.exists(os.path.join(s_DatasetDirectory,'data.yaml')):
            raise Exception("Dataset configuration file does not exist")

    def _CopyDataConfig(self):
        """
        Copy dataset config file
        """
        shutil.copy(
            os.path.join(self.s_DatasetDirectory,'data.yaml'),
            os.path.join(self.s_TrainingOutputPath,'weights','data.yaml')
        )

    def _ClearCacheDataset(self):
        # Remove cache
        for s_Filename in os.listdir(self.s_DatasetDirectory):
            s_Filename =  os.path.join(self.s_DatasetDirectory, s_Filename)
            if not os.path.isdir(s_Filename) and '.cache' in s_Filename.lower():
                os.remove(s_Filename)

    def PrintInfo(self):
        print('CUDA version:',torch.version.cuda)
        print('Torch version:',torch.__version__)
        print('Device:',torch.cuda.get_device_name(0))

    def Train(self, i_Epochs: int, i_BatchSize: int):
        """
        Train YOLO object detection model
        """
        print("Starting training task...")

        kwargs = {
            "data":              os.path.join(self.s_DatasetDirectory, 'data.yaml') if not 'classify'==self.c_Model.task else self.s_DatasetDirectory,
            "batch":             i_BatchSize,
            "epochs":            max(1,i_Epochs),
            "workers":           0,
            "project":           os.path.dirname(self.s_TrainingOutputPath),
            "name":              os.path.basename(self.s_TrainingOutputPath),
            "exist_ok":          True,
            "cache":             'ram',
            "plots":             True,
            "patience":          50,
        }

        self.c_Model.train(
            **(kwargs | TRAINING_PARAMETERS)
        )

    def TestInference(self, f_ConfThresh: float = 0.5):
        """
        Perform inference on a test set
        """

        print("Starting inference task...")

        s_PathToSaveInference = os.path.join(self.s_TrainingOutputPath,'test_inference')
        os.makedirs(s_PathToSaveInference, exist_ok=True)

        for s_FileName in os.listdir(os.path.join(self.s_DatasetDirectory, 'test')):
            if s_FileName.lower().split('.')[-1] not in ALLOWED_INPUT_FILES: continue
            try:
                s_FileName = os.path.join(self.s_DatasetDirectory, 'test', s_FileName)
                self.c_Model.predict(source=s_FileName, conf=f_ConfThresh, imgsz=TRAINING_PARAMETERS['imgsz'], save=True)

                if os.path.exists(os.path.join(self.s_TrainingOutputPath, os.path.basename(s_FileName))):
                    shutil.move(os.path.join(self.s_TrainingOutputPath, os.path.basename(s_FileName)),os.path.join(s_PathToSaveInference,os.path.basename(s_FileName)))
            except:
                pass

    def PlotResults(self):
        """
        Read and plot training results
        """
        def _ReadResultsFile(s_Path, s_FileName):
            s_File = os.path.join(s_Path, s_FileName)
            l_Results = []
            l_ColNames = []
            
            with open(s_File, 'r') as _File:
                for i,line in enumerate(_File):
                    line = line.strip().replace(' ','').split(',')
                    if i == 0:
                        l_ColNames = line
                    else:
                        l_Results.append(line)
                _File.close()
                del(_File)
                
            l_Results = np.moveaxis(np.array(l_Results).astype(float),0,-1)
            dc_Data = dict()

            for i,s_Name in enumerate(l_ColNames):
                dc_Data[s_Name] = l_Results[i]

            del(l_Results)
            del(l_ColNames)

            return dc_Data

        def _PlotResults(dc_Data, s_PlotValue, s_Path):
            s_PathToSave = os.path.join(s_Path,"plots")
            os.makedirs(s_PathToSave, exist_ok=True)

            fig, ax = plt.subplots(1, figsize=(6,4))
            fig.suptitle(f"{s_PlotValue}")

            ax.grid(True)
            ax.plot(dc_Data["epoch"],dc_Data[s_PlotValue], marker='o', color='black')
            ax.set_xlabel("Epochs")
            ax.set_ylabel(f"{s_PlotValue}")

            if "AP" in s_PlotValue:
                argmax = dc_Data["epoch"][np.argmax(dc_Data[s_PlotValue])]
                valmax = np.max(dc_Data[s_PlotValue])
                ax.axvline(argmax, linestyle='--', color='red', linewidth=2)
                ax.axhline(valmax, linestyle='--', color='red', linewidth=2)
                ax.text(0.05*np.max(dc_Data["epoch"]), valmax*0.90, f"Max: {valmax:.3f} at {int(argmax)}",  bbox=dict(facecolor='white', edgecolor='blue', pad=5.0))
                

            fig.tight_layout()
            fig.savefig(os.path.join(s_PathToSave,s_PlotValue.replace(":","-").replace(".","_").replace("/","_")+".png"),dpi=300)

            plt.cla()
            del(fig)
                

        dc_Data = _ReadResultsFile(self.s_TrainingOutputPath, r"results.csv")
        for s_PlotCategory in list(dc_Data.keys()):
            if s_PlotCategory == "epoch": continue
            _PlotResults(dc_Data, s_PlotCategory, self.s_TrainingOutputPath)
