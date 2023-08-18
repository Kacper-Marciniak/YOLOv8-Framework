import numpy as np
import os
import json

from utility.BboxOperations import convertAbsToRel, convertXYXYToXYWH

def saveResultsYolo(dc_Results: dict, s_PathToSave: str):
    s_PathToSave = s_PathToSave.lower()
    if not os.path.exists(os.path.dirname(s_PathToSave)): 
        raise Exception("Output directory doesn't exist")
    if not s_PathToSave.endswith('.txt'): 
        print("Wrong output file extension - changing to \'.txt\'")
        s_PathToSave = s_PathToSave.split('.')[0]+ '.txt'
    
    l_Lines = []

    for i,(a_Bbox, i_ClassID) in enumerate(zip(dc_Results['bbox'],dc_Results['class'])):
        a_Bbox = convertXYXYToXYWH(convertAbsToRel(a_Bbox, dc_Results['img_shape']))

        s_Line = f"{int(i_ClassID)} {a_Bbox[0]:.5f} {a_Bbox[1]:.5f} {a_Bbox[2]:.5f} {a_Bbox[3]:.5f}"

        if i!=0:
            l_Lines.append(f"\n{s_Line}")
        else:
            l_Lines.append(s_Line)

    with open(s_PathToSave, 'w') as _File:
        _File.writelines(l_Lines)
        _File.close()

    print(f"Results saved to {s_PathToSave}")

def saveResultsCoco(dc_Results: dict, s_PathToSave: str, i_ImageID: int = 0):
    s_PathToSave = s_PathToSave.lower()
    if not os.path.exists(os.path.dirname(s_PathToSave)): 
        raise Exception("Output directory doesn't exist")
    if not s_PathToSave.endswith('.json'): 
        print("Wrong output file extension - changing to \'.json\'")
        s_PathToSave = s_PathToSave.split('.')[0]+ '.json'
    
    l_OutputData = []

    for i,(a_Bbox, i_ClassID, f_Score) in enumerate(zip(dc_Results['bbox'],dc_Results['class'],dc_Results['score'])):
        a_Bbox = convertXYXYToXYWH(a_Bbox)

        l_OutputData.append({
            "image_id": i_ImageID,
            "category_id": int(i_ClassID),
            "bbox": a_Bbox.tolist(),
            "score": float(round(f_Score,3)),
        })
    
    with open(s_PathToSave, 'w') as _File:
        json.dump(l_OutputData, _File, indent=4)
        _File.close()

    print(f"Results saved to {s_PathToSave}")