import os
import json

from ml_model.CResults import ImageResults

def saveResultsYolo(c_ImageResults: ImageResults, s_PathToSave: str):
    s_PathToSave = s_PathToSave.lower()
    if not os.path.exists(os.path.dirname(s_PathToSave)): 
        raise Exception("Output directory doesn't exist")
    if not s_PathToSave.endswith('.txt'): 
        print("Wrong output file extension - changing to \'.txt\'")
        s_PathToSave = s_PathToSave.split('.')[0]+ '.txt'
    
    l_Lines = c_ImageResults.get_yolo_detection()
    
    with open(s_PathToSave, 'w') as _File:
        for _Line in l_Lines:
            _File.write(_Line+'\n')

    print(f"Results saved to {s_PathToSave}")

def saveResultsCoco(c_ImageResults: ImageResults, s_PathToSave: str):
    s_PathToSave = s_PathToSave.lower()
    if not os.path.exists(os.path.dirname(s_PathToSave)): 
        raise Exception("Output directory doesn't exist")
    if not s_PathToSave.endswith('.json'): 
        print("Wrong output file extension - changing to \'.json\'")
        s_PathToSave = s_PathToSave.split('.')[0]+ '.json'
    
    l_OutputData = c_ImageResults.get_coco_detection()
    
    with open(s_PathToSave, 'w') as _File:
        json.dump(l_OutputData, _File, indent=4)
        _File.close()

    print(f"Results saved to {s_PathToSave}")