import numpy as np
from ml_model.CResults import Prediction

def makeTiles(a_Img: np.ndarray, i_TargetTileSize=600, f_Overlap=0.2):
        """
        Split image into tiles.
        * a_Img: Image to split
        * i_TargetTileSize: Size of each tile (square)
        * f_Overlap: Overlap between tiles relative to tile size (0.0-1.0)
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

def resultStiching(l_Results: list[list[Prediction]], l_Coords: list) -> list[Prediction]:
    """
    Stiching results from tiled image
    """
    if len(l_Results) == 1:
        return l_Results[0]
        
    else:
        l_StitchedResults = []
        for _TilePredictions, [[x1,y1],[x2,y2]] in zip(l_Results, l_Coords):
            if len(_TilePredictions):                
                for _Pred in _TilePredictions:
                    _Pred.BoundingBox.offset_by(x1,y1)
                    _Pred.Polygon.offset_by(x1,y1)
                    l_StitchedResults.append(_Pred)
        
        return l_StitchedResults
         