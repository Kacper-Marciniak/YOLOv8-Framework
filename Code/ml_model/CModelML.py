"""
Model class definition
"""

import torch
import os, time, sys
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from parameters.parameters import DEFAULT_MODEL_THRESH, ALLOWED_INPUT_FILES
from utility.tiles import makeTiles, resultStiching
from ultralytics.models.sam import Predictor as SAM

class CModelML():
    """
    YOLOv8 object detection model
    * s_PathWeights: str - path to weights file or name of official YOLOv8 model
    * f_Thresh: float = DEFAULT_MODEL_THRESH - confidence score threshold
    * s_ForceDevice: str = '' - force device (f.e. 'cpu', 'cuda:0')
    * b_SAMPostProcess: bool - enable post-processing using Segment Anything Model (SAM)
    * i_TileSize: int - tile size
    """
    def __init__(
            self, 
            s_PathWeights: str, 
            f_Thresh:float = DEFAULT_MODEL_THRESH, 
            s_ForceDevice:str = '',
            b_SAMPostProcess: bool = False,
            i_TileSize: int|None = None
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
        self.l_ClassNames = self.C_Model.names
        if isinstance(self.l_ClassNames, dict):
            self.l_ClassNames = list(self.l_ClassNames.values())
        else:
            self.l_ClassNames = list(self.l_ClassNames)
        print(f"Class names:")
        for s_Class in self.l_ClassNames:
            print(f"\t* {s_Class}")

        # SAM Post-processing
        self.b_PostProcess = b_SAMPostProcess
        if self.b_PostProcess:
            # Change task to 'segment'
            self.s_Task = 'segment'
            # Create SAM predictor
            self.C_SAMModel = SAM(overrides=dict(conf=self.f_Thresh, task='segment', mode='predict', model='sam_b.pt', save=False, verbose=False))            
        else:
            self.C_SAMModel = None

        # Tiling
        self.i_TileSize = i_TileSize
        self.f_TileOVerlap = 0.2

        print(f'Model successfully initialized. Task: {self.s_Task}')

    def __Inference(self, a_InputImg: np.ndarray | str):
        """
        Inference
        """

        a_Bboxes, l_Polygons, a_Scores, a_Classes = np.array([]), [], np.array([]), np.array([])
        f_InferenceTime = np.nan

        try:
            # Perform detection
            _Results = self.C_Model(a_InputImg, verbose=False)[0]

            # Format results
            if _Results.boxes.cls.size(dim=0):                
                a_Bboxes = np.round(_Results.boxes.xyxy.cpu().numpy()).astype(int)
                if self.s_Task == 'segment' and not self.b_PostProcess:
                    l_Polygons = [np.array(np.round(_Polygon),dtype=np.int32) for _Polygon in _Results.masks.xy]
                else:
                    l_Polygons = [[]]*a_Bboxes.shape[0]
                a_Scores = _Results.boxes.conf.cpu().numpy().astype(float)
                a_Classes = _Results.boxes.cls.cpu().numpy().astype(int)
                f_InferenceTime = sum(list(_Results.speed.values()))

                a_Indices = a_Scores>=self.f_Thresh
                a_Bboxes, a_Scores, a_Classes = a_Bboxes[a_Indices], a_Scores[a_Indices], a_Classes[a_Indices]
                l_Polygons= [_Polygon for _Polygon, _Val in zip(l_Polygons, a_Indices) if _Val]

        except Exception as E:
            print(f"Exception {E} during inference.")

        return {
            "bbox": a_Bboxes,
            "polygon": l_Polygons,
            "score": a_Scores,
            "class": a_Classes,
            "inference_time": f_InferenceTime,
        }

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


        print(f"\n\n[YOLOv8 - {self.s_DeviceName}] image shape: {_Input.shape[1]}x{_Input.shape[0]}. Task: {self.s_Task}")
            
        # Tiling
        if not (self.i_TileSize is None) and max(_Input.shape) > self.i_TileSize:
            lCoords = makeTiles(_Input, self.i_TileSize, self.f_TileOVerlap)
        else:
            lCoords = [[[0,0],[_Input.shape[1],_Input.shape[0]]]]

        l_Results = []
        # Inference
        for [[x1,y1],[x2,y2]] in lCoords:
            l_Results.append(self.__Inference(_Input[y1:y2,x1:x2]))

        # Tile stiching
        dc_Results = resultStiching(l_Results, lCoords)

        # SAM post-processing
        if self.b_PostProcess:
            dc_Results = self.PostProcess(_Input, dc_Results)

        dc_Results["img_shape"] = _Input.shape
        dc_Results["task"] = self.s_Task
        dc_Results["names"] = self.l_ClassNames
        dc_Results["time"] = (time.time()-f_Time)*1000.0

        print(f"Detected {len(dc_Results['class'])} objects, {len(np.unique(dc_Results['class']))} unique classes. Inference time: {dc_Results['time']:.3f} ms")
        for _Bbox,_Score,_Class in zip(dc_Results['bbox'],dc_Results['score'],dc_Results['class']):
            print(f"\t* Class \'{self.l_ClassNames[_Class]}\' detected with confidence {_Score:.2f}: {_Bbox.tolist()}")

        sys.stdout = sys.__stdout__
        
        return dc_Results

    def PostProcess(self, a_Img: np.ndarray, dc_Results: dict):
        try:
            self.C_SAMModel.set_image(a_Img)  # set with np.ndarray

            for i,a_Bbox in enumerate(dc_Results['bbox']):
                _Results = self.C_SAMModel(bboxes=a_Bbox)[0]
                # Format results
                if len(_Results.masks.data):                
                    dc_Results['polygon'][i] = np.array(np.round(_Results.masks.xy[0]),dtype=np.int32)
                    x,y,w,h = cv.boundingRect(dc_Results['polygon'][i])
                    dc_Results['bbox'][i] = np.array([x,y,x+w,y+h],dtype=int)

            # Reset image
            self.C_SAMModel.reset_image()
        except Exception as E:
            print(f"Exception {E} during SAM post-processing.")
        
        return dc_Results