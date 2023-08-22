"""
Model class definition
"""

import torch
import yaml, os, time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from parameters.parameters import DEFAULT_MODEL_THRESH
from path.root import ROOT_DIR


class CModelML():
    """
    YOLOv8 object detection model
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
        self.f_Thresh = max( min(round(f_Thresh,3), 0.95), 0.05)

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
    
    def Detect(self, a_Img: np.ndarray, b_PrintOutput: bool = True):
        """
        Perform detection on image
        """
        def print(sText):
            if b_PrintOutput: print(sText)
            else: pass

        a_Bboxes, l_Polygons, a_Scores, a_Classes = np.array([]), [], np.array([]), np.array([])
        try:
            # Perform detection and get bboxes
            f_Time = time.time()
            _Results = self.C_Model(a_Img, verbose=False)[0]
            f_Time = (time.time()-f_Time)*1000.0

            print(f"\n\n[YOLOv8 - {self.s_DeviceName}] image shape: {a_Img.shape[1]}x{a_Img.shape[0]}. Task: {self.s_Task}")

            # Format results
            if _Results.boxes.cls.size(dim=0):                
                a_Bboxes = np.round(_Results.boxes.xyxy.cpu().numpy()).astype(int)
                if self.s_Task == 'segment':
                    l_Polygons = [np.array(np.round(_Polygon),dtype=np.int32) for _Polygon in _Results.masks.xy]
                a_Scores = _Results.boxes.conf.cpu().numpy().astype(float)
                a_Classes = _Results.boxes.cls.cpu().numpy().astype(int)

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

        dc_Results = {
            "bbox": a_Bboxes,
            "polygon": l_Polygons,
            "score": a_Scores,
            "class": a_Classes,
            "img_shape": a_Img.shape,
            "task": self.s_Task,
            "names": self.l_ClassNames,
            "time": f_Time
        }
        
        if self.b_PostProcess:
            dc_Results = self.PostProcess(dc_Results)

        return dc_Results

    def PostProcess(self, dc_Results: dict):        
        #TODO: implement post-processing 
        return dc_Results