"""
Model validation using COCO functions
"""

import os, json
import numpy as np
from datetime import datetime

from ml_model.CModelML import CModelML as Model
from parameters.parameters import DEFAULT_MODEL_THRESH

from path.root import ROOT_DIR

def Validate(s_DatasetDirectory: str, f_Thresh: float = DEFAULT_MODEL_THRESH, b_SaveOutput: bool = True):
    """
    Run system validation. 
    """

    # Initialize model
    C_Model = Model(
        s_PathWeights = os.path.join(ROOT_DIR,'models','model.pt'),
        f_Thresh = 0.75
    )
    s_PathConfigDataset = os.path.join(s_DatasetDirectory, 'data.yaml')

    if os.path.exists(s_DatasetDirectory):
            s_DatasetDirectory = s_DatasetDirectory
    else:
        raise Exception("Dataset does not exist")        
    if not os.path.exists(s_PathConfigDataset):
        raise Exception("Dataset configuration file does not exist")
        
    print(f"Running validation on imageset: {s_DatasetDirectory}")

    sResultsFolder = str(datetime.now().strftime("%Y%m%d_%H%M%S")).replace(' ','_')

    _Metrics = C_Model.C_Model.val(
        data = s_PathConfigDataset,
        plots = b_SaveOutput,
        save = b_SaveOutput,
        conf = f_Thresh,
        split = 'test',
        project = os.path.join(ROOT_DIR,'validation_results'),
        name = sResultsFolder
    )

    a_ClassIds = _Metrics.ap_class_index
    _QualityMetrics = _Metrics.box if C_Model.s_Task != 'segment' else _Metrics.seg


    dc_Results = {
        "mean_ap": _QualityMetrics.map,         # mAP50-95
        "mean_ap50": _QualityMetrics.map50,     # mAP50
        "ap50": {
           _Metrics.names[s_ClassID]: _QualityMetrics.ap50[i] 
           for i,s_ClassID  in enumerate(a_ClassIds)
        },  # AP50
        "ap": {
           _Metrics.names[s_ClassID]: _QualityMetrics.ap[i] 
           for i,s_ClassID  in enumerate(a_ClassIds)
        },  # AP50-95
        "mean_precission": _QualityMetrics.mp,
        "mean_recall": _QualityMetrics.mr,
        "precission": {
           _Metrics.names[s_ClassID]: _QualityMetrics.p[i] 
           for i,s_ClassID  in enumerate(a_ClassIds)
        },  # precission
        "recall": {
           _Metrics.names[s_ClassID]: _QualityMetrics.r[i] 
           for i,s_ClassID  in enumerate(a_ClassIds)
        },  # recall
        "mean_f1": float(np.nanmean(_QualityMetrics.f1)),
        "f1": {
           _Metrics.names[s_ClassID]: _QualityMetrics.f1[i] 
           for i,s_ClassID  in enumerate(a_ClassIds)
        },  # recall
        "speed": _Metrics.speed['inference']+_Metrics.speed['preprocess']+_Metrics.speed['postprocess']
    }

    with open(os.path.join(ROOT_DIR, 'validation_results', sResultsFolder, 'results.json'),'w') as _File:
        json.dump(dc_Results, _File, indent=4)
        _File.close()

    return dc_Results