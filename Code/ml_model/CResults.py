import json
from copy import deepcopy
import numpy as np
import cv2 as cv

class Bbox():
    
    def __init__(self, lBoxXYXY: list):
        self.MinX = float(lBoxXYXY[0])
        self.MinY = float(lBoxXYXY[1])
        self.MaxX = float(lBoxXYXY[2])
        self.MaxY = float(lBoxXYXY[3])

        self.Xc = (self.MinX + self.MaxX) / 2
        self.Yc = (self.MinY + self.MaxY) / 2

    def copy(self):
        return deepcopy(self)

    def scale_by(self, fScaleX: float, fScaleY: float):
        """
        Scale BBOX by a factor
        """
        self.MinX = self.MinX*fScaleX
        self.MinY = self.MinY*fScaleY
        self.MaxX = self.MaxX*fScaleX
        self.MaxY = self.MaxY*fScaleY

        self.Xc = self.Xc*fScaleX
        self.Yc = self.Yc*fScaleY

        return self

    def offset_by(self, fOffsetX: float, fOffsetY: float):
        """
        Offset BBOX by a factor
        """
        self.MinX = self.MinX+fOffsetX
        self.MinY = self.MinY+fOffsetY
        self.MaxX = self.MaxX+fOffsetX
        self.MaxY = self.MaxY+fOffsetY

        self.Xc = self.Xc+fOffsetX
        self.Yc = self.Yc+fOffsetY

        return self

    def round(self):
        """
        Round BBOX coordinates
        """
        self.MinX = int(round(self.MinX))
        self.MinY = int(round(self.MinY))
        self.MaxX = int(round(self.MaxX))
        self.MaxY = int(round(self.MaxY))

        self.Xc = int(round(self.Xc))
        self.Yc = int(round(self.Yc))

        return self

    def get_xywh(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return [self.MinX, self.MinY, self.MaxX - self.MinX, self.MaxY - self.MinY]

    def get_xywh_yolo(self):
        """
        Returns: [xc, yc, width, height]
        """
        return [self.Xc, self.Yc, self.MaxX - self.MinX, self.MaxY - self.MinY]

    def get_xyxy(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return [self.MinX, self.MinY, self.MaxX, self.MaxY]
    
    def get_center(self):
        """
        Returns BBOX center
        """
        return [self.Xc, self.Yc]
    
class Polygon():
    
    def __init__(self, aPolygon: np.ndarray):
        if aPolygon is None or len(aPolygon)==0:
            aPolygon = np.array([])
        if len(aPolygon.shape) == 2:
            aPolygon = aPolygon.reshape(-1,1,2)
        self.aPolygon = aPolygon.astype(float)

    def exists(self):
        return len(self.aPolygon)>0

    def copy(self):
        return deepcopy(self)
    
    def approx(self, fEps: float):
        """
        Approximate polygon
        """
        if self.exists():
            fPeri = cv.arcLength(self.aPolygon, True)
            self.aPolygon = cv.approxPolyDP(self.aPolygon, fEps * fPeri, True)
        
        return self

    def scale_by(self, fScaleX: float, fScaleY: float):
        """
        Scale BBOX by a factor
        """
        if self.exists():
            self.aPolygon[:,0,0] *= fScaleX
            self.aPolygon[:,0,1] *= fScaleY

        return self

    def offset_by(self, fOffsetX: float, fOffsetY: float):
        """
        Offset BBOX by a factor
        """
        if self.exists():
            self.aPolygon[:,0,0] += fOffsetX
            self.aPolygon[:,0,1] += fOffsetY

        return self

    def round(self):
        """
        Round BBOX coordinates
        """
        if self.exists():
            self.aPolygon = np.round(self.aPolygon).astype(np.int32)

        return self

    def get_array(self): 
        """
        Returns polygon array
        """ 
        return self.aPolygon

    def get_yolo(self):
        """
        Returns polygon in YOLO format (flatten)
        """
        self.aPolygon.flatten()
    
class Prediction():

    def __init__(self, sClass: str, iClass: int, fScore: float, lBoxXYXY: list[float] | list[int], aPolygon: np.ndarray = np.array([])):
        self.BoundingBox = Bbox(lBoxXYXY)
        self.Polygon = Polygon(aPolygon)
        self.sClass = sClass
        self.iClass = iClass
        self.fScore = fScore
    
    def copy(self):
        return deepcopy(self)

    def approx_polygon(self, fEps: float = 0.0080):
        self.Polygon.approx(fEps)
        return self
    
    def set_polygon(self, aPolygon: np.ndarray):
        self.Polygon = Polygon(aPolygon)
        return self

    def get_bbox(self):
        return self.BoundingBox.copy()
    
    def get_polygon(self):
        return self.Polygon.copy()
    
    def merge(self, Prediction2: Polygon):
        
        #Merge scores

        self.fScore = (self.fScore + float(Prediction2.fScore))/2.0

        # Merge polygons
        
        aTmpMask = np.logical_or(self.Polygon.get_mask(), Prediction2.get_polygon().get_mask()).astype(np.uint8)*255
        aCntr = cv.findContours(aTmpMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0].astype(float)
        aCntr[:,0,0] /= aTmpMask.shape[1]
        aCntr[:,0,1] /= aTmpMask.shape[0]
        self.Polygon = Polygon(aCntr)

        # Get BBOX 
                      
        self.BoundingBox = Bbox([
            aCntr[:,:,0].min(),
            aCntr[:,:,1].min(),
            aCntr[:,:,0].max(),
            aCntr[:,:,1].max(),
        ])

        return self
    
    def cut(self, Prediction2: Polygon, iDilate: int = 0):
        AREA_THRESH_ACCEPT_NONE = 250
        AREA_THRESH_ACCEPT_ALL = 2500

        # Get polygons
        aTmpMask = Prediction2.get_polygon().get_mask()
        if iDilate>0: aTmpMask = cv.dilate(aTmpMask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (iDilate,iDilate)))
        aTmpMask = np.logical_and(self.Polygon.get_mask(), np.logical_not(aTmpMask)).astype(np.uint8)*255
        
        lContours, lBBOXes = [],[]
        for aCntr in cv.findContours(aTmpMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:            
            aCntr = aCntr.astype(float)

            # Filter contours
            fArea = cv.contourArea(aCntr)
            if fArea<AREA_THRESH_ACCEPT_NONE: 
                continue
            elif fArea<AREA_THRESH_ACCEPT_ALL:
                (x,y),(w,h),ang = cv.fitEllipse(aCntr)
                if min(w,h)/max(w,h) < 0.35: continue # stretched
            else:
                pass
            
            aCntr[:,0,0] /= aTmpMask.shape[1]
            aCntr[:,0,1] /= aTmpMask.shape[0]

            lContours.append(aCntr)
            lBBOXes.append([
                aCntr[:,:,0].min(),
                aCntr[:,:,1].min(),
                aCntr[:,:,0].max(),
                aCntr[:,:,1].max(),
            ])        
        
        return [self.new(self.sClass, self.fScore, _BBOX, _Poly) for  _Poly, _BBOX in zip(lContours, lBBOXes)]

    @classmethod
    def from_dict(self, dcData: dict):
        return self(
            sClass = dcData['class'],
            fScore = dcData['score'],
            lBoxXYXY = dcData['bbox'],
            aPolygon = np.array(dcData['polygon'], dtype=float).reshape(-1,1,2)
        ).approx_polygon()
    
    @classmethod
    def new(self, sClass: str, fScore: float, lBoxXYXY: list, aPolygon: np.ndarray = []):
        return self(
            sClass = sClass,
            fScore = fScore,
            lBoxXYXY = lBoxXYXY,
            aPolygon = aPolygon
        )
    
    def to_dict(self):
        return {
            "class": self.sClass,
            "score": self.fScore,
            "bbox": self.BoundingBox.get_xyxy(),
            "polygon": self.Polygon.get_array().tolist() if self.Polygon.exists() else []
        }

class ImageResults():

    def __init__(self, sImageID: str, tImageShape: tuple, lPredictions: list[Prediction], fInferenceTime: float = None):
        self.sImageID = sImageID
        self.tImageShape = tImageShape
        self.fInferenceTime = fInferenceTime
        self.load_predictions(lPredictions)

    def get_inference_time(self):
        return self.fInferenceTime

    def get_n_predictions(self):
        return len(self.lPredictions)
    
    def get_n_predictions_by_class(self, sClass: str):
        return len([pred for pred in self.get_predictions() if pred.sClass==sClass])
    
    def load_predictions(self, lPredictions: list[Prediction]):
        self.lPredictions = lPredictions

    def set_prediction_by_index(self, iIndex: int, cPred: Prediction):
        if iIndex < len(self.lPredictions):
            self.lPredictions[iIndex] = cPred

    def add_predictions(self, lPredictions: list[Prediction]):
        self.lPredictions += lPredictions

    def remove_predictions_by_index(self, lPredictions: list[int]):
        self.lPredictions = [pred for i,pred in enumerate(self.get_predictions()) if i not in lPredictions]

    def list_results(self):
        print(f"Image \'{self.sImageID}\' - {self.get_n_predictions()} detections.")
        for i, pred in enumerate(self.get_predictions()):
            print(f"\t{i+1}) {pred.sClass} with score {pred.fScore:.2f}")
    
    def to_dict(self):
        return {
            "id": self.sImageID,
            "width": self.tImageShape[1],
            "height": self.tImageShape[0],
            "predictions": [pred.to_dict() for pred in self.lPredictions]
        }
    
    def to_json(self, sPath: str):
        with open(sPath.split('.')[0]+'.json', 'w') as f:
            json.dump(self.to_dict(), f)
            f.close()

    def get_coco_detection(self, iImageID: int):
        return [{
            "image_id": int(iImageID),
            "category_id": int(pred.iClass),
            "score": float(pred.fScore),
            "bbox": pred.BoundingBox.copy().round().get_xywh(),
            "segmentation": pred.Polygon.get_array().reshape(-1,2).astype(float) if pred.Polygon.exists() else []
        } for pred in self.lPredictions]
    
    def get_yolo_detection(self):
        lResults = []
        for pred in self.lPredictions:
            if pred.Polygon.exists():
                sTmpLine = f"{int(pred.iClass)}"
                for p in pred.get_polygon().scale_by(1.0/self.tImageShape[1],1.0/self.tImageShape[0]).get_yolo():
                    sTmpLine += f"{float(p):.5f}"
                lResults.append(sTmpLine)
            else:    
                lBbox = pred.BoundingBox.scale_by(1.0/self.tImageShape[1],1.0/self.tImageShape[0]).get_xywh_yolo()
                lResults.append(f"{int(pred.iClass)} {lBbox[0]:.5f} {lBbox[1]:.5f} {lBbox[2]:.5f} {lBbox[3]:.5f}")
        return lResults
    
    def get_predictions(self) -> list[Prediction]:
        return deepcopy(self.lPredictions)
    
    def get_predictions_by_class(self, sClass: str):
        return [pred for pred in self.get_predictions() if pred.sClass==sClass]
    
    def get_data_visualisation(self, tTargetShape: tuple = None):
        if tTargetShape is None: tTargetShape = self.tImageShape
        return [
            {
                "bbox": pred.BoundingBox.copy().scale_by(tTargetShape[1], tTargetShape[0]).round().get_xyxy(),
                "polygon": pred.Polygon.copy().scale_by(tTargetShape[1], tTargetShape[0]).round().get_array(),
                "class": pred.sClass,
                "score": pred.fScore
            }
            for pred in self.get_predictions()
        ]

    def get_data_mask_generation(self, tTargetShape: tuple = None):
        if tTargetShape is None: tTargetShape = self.tImageShape
        return [
            pred.Polygon.copy().scale_by(tTargetShape[1], tTargetShape[0]).round().get_array()
            for pred in self.get_predictions()
        ]

    def combine_predictions(self, fRCAthresh: float = 0.5):
        """
        Combine predictions
        """
        lPredictionsCombined = []
        
        for sClass in ('added', 'removed'):
            lBuffer = [pred for pred in self.get_predictions() if pred.sClass==sClass]
            
            i,j = 0,1
            while len(lBuffer)>1:
                if j>=len(lBuffer):
                    i+=1
                    j=0
                    if i>=len(lBuffer) and not bIsChange: 
                        break
                
                i %= len(lBuffer)
                j %= len(lBuffer)
            
                if i != j:
                    fCommonAreaRelative = lBuffer[i].Polygon.get_common_area_realtive_to_min(lBuffer[j].Polygon)
                    if fCommonAreaRelative>fRCAthresh: 
                        lBuffer[i].merge(lBuffer[j])
                        lBuffer.pop(j)
                        bIsChange = True
                        if j<i: i-=1
                        continue
                bIsChange = False
                j+=1
            
            lPredictionsCombined += lBuffer

        self.load_predictions(lPredictionsCombined)

        return self