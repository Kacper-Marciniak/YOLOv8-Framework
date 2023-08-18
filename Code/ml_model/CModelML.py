"""
Model class definition
"""

import torch
import yaml, os, time
import numpy as np
from ultralytics import YOLO
from parameters.parameters import DEFAULT_MODEL_THRESH

class CModelML():
    """
    YOLOv8 object detection model
    """
    def __init__(
            self, 
            s_PathWeights: str, 
            f_Thresh=DEFAULT_MODEL_THRESH, 
            s_ForceDevice=''
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
        self.C_Model.conf_thres = self.f_Thresh
        self.C_Model.to(self._Device)
        
        # Load class names definition
        with open(os.path.join(os.path.dirname(s_PathWeights), 'data.yaml'),'r') as _File:
            self.l_ClassNames = yaml.safe_load(_File)['names']
            print(f"Class names:")
            for s_Class in self.l_ClassNames:
                print(f"\t* {s_Class}")

        print('Model successfully initialized')
    
    def Detect(self, a_Img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform detection on image
        """
        a_Bboxes, a_Scores, a_Classes = np.array([]), np.array([]), np.array([])
        try:
            # Perform detection and get bboxes
            f_Time = time.time()
            _Results = self.C_Model(a_Img, verbose=False)[0].boxes
            f_Time = (time.time()-f_Time)*1000.0

            print(f"\n\n[YOLOv8 - {self.s_DeviceName}] image shape: {a_Img.shape[1]}x{a_Img.shape[0]}.")

            # Format results
            if _Results.cls.size(dim=0):
                a_Bboxes = np.round(_Results.xyxy.cpu().numpy()).astype(int)
                a_Scores = _Results.conf.cpu().numpy().astype(float)
                a_Classes = _Results.cls.cpu().numpy().astype(int)

                print(f"Detected {len(a_Bboxes)} objects, {len(np.unique(a_Classes))} unique classes. Inference time: {f_Time:.3f} ms")
                for _Bbox,_Score,_Class in zip(a_Bboxes,a_Scores,a_Classes):
                    print(f"\t* Class \'{self.l_ClassNames[_Class]}\' detected with confidence {_Score:.2f}: {_Bbox.tolist()}")
            else:
                print(f"No objects detected.")
 
        except Exception as E:
            print(f"Exception {E} during inference.")

        return {
            "bbox": a_Bboxes,
            "score": a_Scores,
            "class": a_Classes,
            "img_shape": a_Img.shape
        }