import cv2 as cv
import numpy as np

class CCamera:
    def __init__(self, i_Name: int = 0):
        print(f"Initializing camera {i_Name}")
        self.i_DeviceName = i_Name
        print(f"Trying to access camera {i_Name}...")
        self.CCamera = cv.VideoCapture(self.i_DeviceName, cv.CAP_DSHOW)
        if not self.CCamera.isOpened(): self.CCamera.open()
        print(f"Camera {self.i_DeviceName} is ready!")

    def setResolution(self, i_Width: int, i_Height: int):
        print(f"Trying to change the resolution to {i_Width}x{i_Height}")
        self.CCamera.set(cv.CAP_PROP_FRAME_WIDTH, i_Width) 
        self.CCamera.set(cv.CAP_PROP_FRAME_HEIGHT, i_Height)
        self.t_Shape = self.getCameraResolution()
        print(f"Resolution set to: {self.t_Shape[1]}x{self.t_Shape[0]}")

    def grabFrame(self):
        b_Res, a_Output = self.CCamera.read()
        if b_Res:
            return np.array(a_Output, dtype=np.uint8)
        else: 
            return np.zeros(self.t_Shape, dtype=np.uint8)

    def displayInLoop(self, s_BreakKey: str ='q', i_TargetFps: int = 60):
        while True:
            cv.imshow("Output", self.grabFrame())
            if cv.waitKey(1000//i_TargetFps) == ord(s_BreakKey):
                break

    def close(self):
        self.CCamera.release()
        print(f"Camera {self.i_DeviceName} was closed!")

    def getCameraResolution(self):
        try:            
            self.t_Shape = self.grabFrame().shape
        except:
            self.t_Shape = (-1,-1,-1)
        self.t_Shape = self.getCameraResolution()
        return self.t_Shape