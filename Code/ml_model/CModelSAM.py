"""
SAM class definition
"""

import os, time, sys
import numpy as np
import cv2 as cv
from parameters.parameters import DEFAULT_MODEL_THRESH, ALLOWED_INPUT_FILES
from ultralytics.models.sam import Predictor as SAM
from ml_model.CResults import ImageResults, Prediction

class CModelSAM():
    """
    SAM segmentation
    * f_Thresh: float - confidence score threshold
    """
    def __init__(
            self, 
            f_Thresh:float = DEFAULT_MODEL_THRESH,
        ):
        # Set confidence threshold
        self.f_Thresh = np.clip(f_Thresh,0.001,0.999)

        self.s_Task = 'segment'

        self.l_ClassNames = ['SAM-segment']

        self.C_SAMModel = SAM(overrides=dict(conf=self.f_Thresh, task='segment', mode='predict', model='mobile_sam.pt', save=False, verbose=False))

        print(f'SAM successfully initialized.')

    def setImage(self, _Input: np.ndarray | str):
        """
        Set image for segmentation
        """

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

        self.C_SAMModel.set_image(_Input)

    def resetImage(self):
        """
        Reset image
        """
        self.C_SAMModel.reset_image()
    
    def __Inference(self, l_Bbox: list) -> list[Prediction]:
        """
        Inference
        """
        
        l_Predictions = []

        try:
            _Results = self.C_SAMModel(bboxes=l_Bbox)[0]
            # Format results
            for i,_Polygon in enumerate(_Results.masks.xy):
                a_Polygon = np.array(np.round(_Polygon),dtype=np.int32)
                x,y,w,h = cv.boundingRect(a_Polygon)
                l_Bbox = [x,y,x+w,y+h]
                l_Predictions.append(Prediction(self.l_ClassNames[0], 0, -1.0, l_Bbox, a_Polygon))

        except Exception as E:
            print(f"Exception {E} during SAM inference.")

        return l_Predictions

    def Segment(self, _Input: np.ndarray | str, l_Bbox: list, b_PrintOutput: bool = True, s_ImageID: str = None) -> ImageResults:
        """
        Perform SAM segmentation on image, needs bounding box prompt
        """   
        f_Time = time.time()

        if not b_PrintOutput:            
            sys.stdout = open(os.devnull, 'w')
            
        lPredictions = self.__Inference(l_Bbox)

        c_Results = ImageResults(
            s_ImageID,
            _Input.shape,
            lPredictions,
            (time.time() - f_Time) * 1000.0
        )
        print(f"Inference time: {c_Results.get_inference_time():.3f} ms.")

        sys.stdout = sys.__stdout__
        
        return c_Results