"""
Model class definition
"""

import torch
import yaml, os, time, sys
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from parameters.parameters import DEFAULT_MODEL_THRESH, ALLOWED_INPUT_FILES
from path.root import ROOT_DIR


class CModelML():
    """
    YOLOv8 object detection model
    * s_PathWeights: str - path to weights file or name of official YOLOv8 model
    * f_Thresh: float = DEFAULT_MODEL_THRESH - confidence score threshold
    * s_ForceDevice: str = '' - force device (f.e. 'cpu', 'cuda:0')
    * b_PostProcess: bool - enable post-processing
    """
    def __init__(
            self, 
            s_PathWeights: str, 
            f_Thresh:float = DEFAULT_MODEL_THRESH, 
            s_ForceDevice:str = '',
            b_PostProcess: bool = True
        ):
        # Set device
        self._Device = torch.device(0) if s_ForceDevice == '' else s_ForceDevice
        self.s_DeviceName = torch.cuda.get_device_name(self._Device) if self._Device != 'cpu' else 'CPU'
        
        # Set confidence threshold
        self.f_Thresh = np.clip(f_Thresh,0.001,0.999)

        # Initialize YOLO architecture
        print(f"Initializing YOLOv8 model\n\tWeights: {s_PathWeights}\n\tDevice: {self.s_DeviceName}")
        self.C_Model = YOLO(
            model = s_PathWeights,
        )
        self.C_Model.conf_thresh = self.f_Thresh
        self.C_Model.to(self._Device)

        self.s_Task = 'detect' if self.C_Model.task != 'segment' else 'segment'
        
        # Load class names definition
        sConfigFilePath = os.path.join(os.path.dirname(s_PathWeights), 'data.yaml')
        if not os.path.exists(sConfigFilePath): 
            print(f"Local configuration file {sConfigFilePath} doesn't exist. Loading default COCO configuration.")
            sConfigFilePath = os.path.join(ROOT_DIR, 'configuration', 'data_coco.yaml')
        with open(sConfigFilePath,'r') as _File:
            self.l_ClassNames = yaml.safe_load(_File)['names']
            if isinstance(self.l_ClassNames, dict):
                self.l_ClassNames = list(self.l_ClassNames.values())
            elif not isinstance(self.l_ClassNames, list):
                raise Exception(f"Invalid class definition in {sConfigFilePath} configuration file!")
            else:
                self.l_ClassNames = list(self.l_ClassNames)
            print(f"Class names:")
            for s_Class in self.l_ClassNames:
                print(f"\t* {s_Class}")

        # Post-processing
        self.b_PostProcess = b_PostProcess

        print(f'Model successfully initialized. Task: {self.s_Task}')
    
    def Detect(self, _Input: np.ndarray | str, b_PrintOutput: bool = True):
        """
        Perform detection on image
        """
        f_Time = time.time()
        
        if not b_PrintOutput:            
            sys.stdout = open(os.devnull, 'w')

        # Check input
        if isinstance(_Input, np.ndarray): pass
        elif isinstance(_Input, str):
            _Input = _Input.lower()
            if os.path.exists(_Input) and _Input.split('.')[-1] in ALLOWED_INPUT_FILES:
                _Input = cv.imread(_Input)
            else:
                raise Exception("Invalid input path")
        else: 
            raise Exception("Invalid model input")

        a_Bboxes, l_Polygons, a_Scores, a_Classes = np.array([]), [], np.array([]), np.array([])
        f_InferenceTime = np.nan

        try:
            # Perform detection
            _Results = self.C_Model(_Input, verbose=False)[0]

            print(f"\n\n[YOLOv8 - {self.s_DeviceName}] image shape: {_Input.shape[1]}x{_Input.shape[0]}. Task: {self.s_Task}")

            # Format results
            if _Results.boxes.cls.size(dim=0):                
                a_Bboxes = np.round(_Results.boxes.xyxy.cpu().numpy()).astype(int)
                if self.s_Task == 'segment':
                    l_Polygons = [np.array(np.round(_Polygon),dtype=np.int32) for _Polygon in _Results.masks.xy]
                a_Scores = _Results.boxes.conf.cpu().numpy().astype(float)
                a_Classes = _Results.boxes.cls.cpu().numpy().astype(int)
                f_InferenceTime = sum(list(_Results.speed.values()))

                a_Indices = a_Scores>=self.f_Thresh
                a_Bboxes, a_Scores, a_Classes = a_Bboxes[a_Indices], a_Scores[a_Indices], a_Classes[a_Indices]
                l_Polygons= [_Polygon for _Polygon, _Val in zip(l_Polygons, a_Indices) if _Val]

                print(f"Detected {len(a_Bboxes)} objects, {len(np.unique(a_Classes))} unique classes. Inference time: {f_Time:.3f} ms")
                for _Bbox,_Score,_Class in zip(a_Bboxes,a_Scores,a_Classes):
                    print(f"\t* Class \'{self.l_ClassNames[_Class]}\' detected with confidence {_Score:.2f}: {_Bbox.tolist()}")
            else:
                print(f"No objects detected.")
 
        except Exception as E:
            print(f"Exception {E} during inference.")

        f_Time = (time.time()-f_Time)*1000.0

        dc_Results = {
            "bbox": a_Bboxes,
            "polygon": l_Polygons,
            "score": a_Scores,
            "class": a_Classes,
            "img_shape": _Input.shape,
            "task": self.s_Task,
            "names": self.l_ClassNames,
            "time": f_Time,
            "inference_time": f_InferenceTime
        }
        
        if self.b_PostProcess:
            dc_Results = self.PostProcess(dc_Results)

        sys.stdout = sys.__stdout__
        
        return dc_Results

    def PostProcess(self, dc_Results: dict):        
        #TODO: implement post-processing 
        return dc_Results