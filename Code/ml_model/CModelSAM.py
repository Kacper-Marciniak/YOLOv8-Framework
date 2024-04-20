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
    * f_Thresh: float = DEFAULT_MODEL_THRESH - confidence score threshold
    """
    def __init__(
            self, 
            f_Thresh:float = DEFAULT_MODEL_THRESH,
        ):
        # Set confidence threshold
        self.f_Thresh = np.clip(f_Thresh,0.001,0.999)

        self.s_Task = 'segment'

        self.l_ClassNames = ['SAM-segment']

        self.C_SAMModel = SAM(overrides=dict(conf=self.f_Thresh, task='segment', mode='predict', model='sam_b.pt', save=False, verbose=False))

        print(f'SAM successfully initialized.')

    def __Inference(self, a_InputImg: np.ndarray | str, l_Points: list) -> list[Prediction]:
        """
        Inference
        """
        
        l_Predictions = []

        try:
            self.C_SAMModel.set_image(a_InputImg)

            _Results = self.C_SAMModel(points=l_Points)[0]
            # Format results
            for i,_Polygon in enumerate(_Results.masks.xy):
                a_Polygon = np.array(np.round(_Polygon),dtype=np.int32)
                x,y,w,h = cv.boundingRect(a_Polygon)
                l_Bbox = [x,y,x+w,y+h]
                l_Predictions.append(Prediction(self.l_ClassNames[0], 0, -1.0, l_Bbox, a_Polygon))

            # Reset image
            self.C_SAMModel.reset_image()
        except Exception as E:
            print(f"Exception {E} during SAM inference.")

        return l_Predictions

    def Detect(self, _Input: np.ndarray | str, l_Points: list, b_PrintOutput: bool = True, s_ImageID: str = None) -> ImageResults:
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


        print(f"\n\n[SAM] image shape: {_Input.shape[1]}x{_Input.shape[0]}")
            
        lPredictions = self.__Inference(_Input, l_Points)

        c_Results = ImageResults(
            s_ImageID,
            _Input.shape,
            lPredictions,
            (time.time() - f_Time) * 1000.0
        )
        print(f"Inference time: {c_Results.get_inference_time():.3f} ms.")

        sys.stdout = sys.__stdout__
        
        return c_Results