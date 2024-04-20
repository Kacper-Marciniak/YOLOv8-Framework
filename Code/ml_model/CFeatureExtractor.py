"""
INCEPTIONv3 based feature extractor
"""

import torch
import numpy as np
import os
import cv2 as cv
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn


import torch
import cv2 as cv
import torchvision.transforms as T

_ListTransformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
])

class CFeatureExtractor():
    """
    Class definition for feature extractor
    """
    def __init__(self, iImageSize: float | int):
        # Get device (cuda or cpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load classifier model
        self.Model = inception_v3()
        self.Model.fc = nn.Sequential() # Empty classifier head
        #self.Model.load_state_dict(torch.load(os.path.join(PATH_PRODUCTION_MODEL,'extractor_weights.pth')))
        self.Model.eval()
        self.Model.to(self.device)

        self.fImageSize = float(iImageSize)

        print(f"Initialized extractor model using {self.device}")

    def _Transform(self, aImg: np.ndarray):
        """
        Apply transformations to the image
        """
    
        fRes = self.fImageSize/float(max(aImg.shape))
        aImg = cv.resize(aImg, (int(round(aImg.shape[1]*fRes)),int(round(aImg.shape[0]*fRes))))
        
        aImgSquare = np.zeros((max(aImg.shape),max(aImg.shape),3), dtype=np.uint8)
        aImgSquare[
            aImgSquare.shape[0]//2-aImg.shape[0]//2:aImgSquare.shape[0]//2-aImg.shape[0]//2+aImg.shape[0],
            aImgSquare.shape[1]//2-aImg.shape[1]//2:aImgSquare.shape[1]//2-aImg.shape[1]//2+aImg.shape[1],
        ] = aImg
        aImg = aImgSquare
    
        return Variable(torch.unsqueeze(_ListTransformations(aImg), dim=0).to(self.device))       

    def Inference(self, Img: np.ndarray) -> np.ndarray: 
        """
        Get feature vector
        """
        aFeatures = self.Model(self._Transform(Img)).cpu().detach().numpy()

        return aFeatures.flatten()