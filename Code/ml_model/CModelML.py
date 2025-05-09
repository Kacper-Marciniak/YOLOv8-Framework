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
from ml_model.CResults import ImageResults, Prediction, Bbox, Polygon

class CModelML():
    """
    YOLOv8 object detection model
    * s_PathWeights: str - path to weights file or name of official YOLOv8 model
    * f_Thresh: float - confidence score threshold
    * s_ForceDevice: str - force device (f.e. 'cpu', 'cuda:0')
    * b_SAMPostProcess: bool - enable post-processing using Segment Anything Model (SAM)
    * i_TileSize: int - tile size
    """
    def __init__(
            self, 
            s_PathWeights: str, 
            f_Thresh: float = DEFAULT_MODEL_THRESH, 
            s_ForceDevice: str = '',
            b_SAMPostProcess: bool = False,
            i_TileSize: int|None = None
        ):
        # Set device
        try:
            self._Device = torch.device(0) if s_ForceDevice == '' else s_ForceDevice
            self.s_DeviceName = torch.cuda.get_device_name(self._Device) if self._Device != 'cpu' else 'CPU'
        except: # No CUDA
            print("No CUDA device found. Using CPU.")
            self._Device = 'cpu'
            self.s_DeviceName = 'CPU' 
                
        # Set confidence threshold
        self.f_Thresh = np.clip(f_Thresh,0.001,0.999)

        # Initialize YOLO architecture
        print(f"Initializing YOLO model\n\tWeights: {s_PathWeights}\n\tDevice: {self.s_DeviceName}")
        self.C_Model = YOLO(
            model = s_PathWeights,
        )
        self.C_Model.conf_thresh = self.f_Thresh
        self.C_Model.to(self._Device)

        self.s_Task = self.C_Model.task
        
        # Load class names definition
        self.dc_ClassNames = self.C_Model.names
        print(f"Class names:")
        for i_ID, s_Class in self.dc_ClassNames.items():
            print(f"\t[{i_ID}] {s_Class}")

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

    def __Inference(self, a_InputImg: np.ndarray | str) -> list[Prediction]:
        """
        Inference
        """

        l_Predictions = []

        try:
            # Perform detection
            _Results = self.C_Model.predict(a_InputImg, verbose=False, conf=self.f_Thresh)[0]

            # Format results
            if _Results.boxes.cls.size(dim=0):                
                a_Bboxes = np.round(_Results.boxes.xyxy.cpu().numpy()).astype(int)
                if self.s_Task == 'segment' and not self.b_PostProcess:
                    l_Polygons = [np.array(np.round(_Polygon),dtype=np.int32) for _Polygon in _Results.masks.xy]
                else:
                    l_Polygons = [[]]*a_Bboxes.shape[0]
                a_Scores = _Results.boxes.conf.cpu().numpy().astype(float)
                a_Classes = _Results.boxes.cls.cpu().numpy().astype(int)

                l_Predictions = [
                    Prediction(self.dc_ClassNames[a_Classes[i]], a_Classes[i], a_Scores[i], a_Bboxes[i], l_Polygons[i] if len(l_Polygons) else np.array([])) 
                    for i in range(len(a_Classes))
                ]

        except Exception as E:
            print(f"Exception {E} during inference.")

        return l_Predictions

    def Detect(self, _Input: np.ndarray | str, b_PrintOutput: bool = True, s_ImageID: str = None) -> ImageResults:
        """
        Perform detection/segmentation on image
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


        print(f"\n\n[YOLO - {self.s_DeviceName}] image shape: {_Input.shape[1]}x{_Input.shape[0]}. Task: {self.s_Task}")
            
        # Tiling
        if not (self.i_TileSize is None) and max(_Input.shape) > self.i_TileSize:
            lCoords = makeTiles(_Input, self.i_TileSize, self.f_TileOVerlap)
        else:
            lCoords = [[[0,0],[_Input.shape[1],_Input.shape[0]]]]

        # Inference
        l_Results = [self.__Inference(_Input[y1:y2,x1:x2]) for [[x1,y1],[x2,y2]] in lCoords]

        # Tile stiching
        l_Results = resultStiching(l_Results, lCoords)

        c_ImageResults = ImageResults(
            s_ImageID,
            _Input.shape,
            l_Results,
            (time.time()-f_Time)*1000.0
        )

        # SAM post-processing
        if self.b_PostProcess:
            c_ImageResults = self.PostProcess(_Input, c_ImageResults)


        print(f"Detected {c_ImageResults.get_n_predictions()} objects. Inference time: {c_ImageResults.get_inference_time():.3f} ms")
        c_ImageResults.list_results()

        sys.stdout = sys.__stdout__
        
        return c_ImageResults

    def PostProcess(self, a_Img: np.ndarray, c_ImageResults: ImageResults) -> list[Prediction]:
        try:
            # Set image
            self.C_SAMModel.set_image(a_Img)  # set with np.ndarray
            
            for i,_Pred in c_ImageResults.get_predictions():
                _Results = self.C_SAMModel(bboxes=_Pred.get_bbox().round().get_xyxy())[0]
                # Format results
                if len(_Results.masks.data):
                    a_Polygon = np.array(np.round(_Results.masks.xy[0]),dtype=np.int32)
                    _Pred.Polygon = Polygon(a_Polygon)
                    x,y,w,h = cv.boundingRect(a_Polygon)
                    _Pred.BoundingBox = Bbox([x,y,x+w,y+h])
                    c_ImageResults.set_prediction_by_index(i, _Pred)

            # Reset image
            self.C_SAMModel.reset_image()
        
        except Exception as E:
            print(f"Exception {E} during SAM post-processing.")
        
        return c_ImageResults