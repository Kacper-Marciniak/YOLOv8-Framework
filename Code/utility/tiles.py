import numpy as np


def makeTiles(a_Img: np.ndarray, i_TargetTileSize=600, f_Overlap=0.2):
        """
        Image tiling
        """
        # Get distance between tiles
        f_TileDistance = round(i_TargetTileSize * (1-f_Overlap))
        # Calculate number of tiles
        t_NTiles = (
            int(np.ceil(
                max(0, a_Img.shape[0]-i_TargetTileSize)/f_TileDistance
            )+1),
            int(np.ceil(
                max(0, a_Img.shape[1]-i_TargetTileSize)/f_TileDistance
            )+1)
        )
        # Get distance in X and Y
        i_dX = int((a_Img.shape[1]-i_TargetTileSize)/(t_NTiles[1]-1)) if t_NTiles[1]>1 else 0
        i_dY = int((a_Img.shape[0]-i_TargetTileSize)/(t_NTiles[0]-1)) if t_NTiles[0]>1 else 0

        l_Coordinates = []
        
        for x in range(t_NTiles[1]):
            for y in range(t_NTiles[0]):
                # Save coordinates of tile
                l_Coordinates.append([
                    [max(0,x*i_dX),max(0,y*i_dY)],
                    [min(a_Img.shape[1],x*i_dX+i_TargetTileSize),min(a_Img.shape[0],y*i_dY+i_TargetTileSize)]
                ])

        return l_Coordinates

def resultStiching(l_Results: list[dict], l_Coords: list):
    """
    Stiching results from tiled image
    """
    if len(l_Results) == 0:
        return l_Results[0]
        
    else:
        dc_Results = {
            "bbox": [],
            "polygon": [],
            "score": [],
            "class": [],
            "inference_time": [],
        }
        for dc_TileResults, [[x1,y1],[x2,y2]] in zip(l_Results, l_Coords):
            if len(dc_TileResults["class"]):

                dc_TileResults["bbox"][:,0] += x1
                dc_TileResults["bbox"][:,2] += x1
                dc_TileResults["bbox"][:,1] += y1
                dc_TileResults["bbox"][:,3] += y1
                for i,_ in enumerate(dc_TileResults["polygon"]):
                    if not len(dc_TileResults["polygon"][i]): continue
                    dc_TileResults["polygon"][i][:,0] += x1
                    dc_TileResults["polygon"][i][:,1] += y1

                dc_Results["bbox"] += (dc_TileResults["bbox"]).tolist()            
                dc_Results["polygon"] += dc_TileResults["polygon"]

                dc_Results["score"] += dc_TileResults["score"].tolist()
                dc_Results["class"] += dc_TileResults["class"].tolist()

            dc_Results["inference_time"].append(dc_TileResults["inference_time"])
        

        dc_Results["class"] = np.array(dc_Results["class"])
        dc_Results["score"] = np.array(dc_Results["score"])
        dc_Results["bbox"] = np.array(dc_Results["bbox"])
        dc_Results["inference_time"] = np.nanmean(dc_Results["inference_time"])
        return dc_Results
         