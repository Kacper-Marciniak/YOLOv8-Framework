import cv2 as cv
import time
import numpy as np
import os

import threading


def threaded(fn):
    def wrapper(*args, **kwargs):
        #thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class CCamera:
    """
    Camera class from webcam functionality
    """
    def __init__(self, i_Name: int = 0):
        print(f"Initializing camera {i_Name}")
        self.i_DeviceName = i_Name
        self.b_Run = False
        print(f"Trying to access camera {i_Name}...")
        if os.name == 'nt':
            self.CCamera = cv.VideoCapture(self.i_DeviceName, cv.CAP_DSHOW)
        else:
            self.CCamera = cv.VideoCapture(self.i_DeviceName)
        if not self.CCamera.isOpened(): raise Exception(f"Cannot open camera {self.i_DeviceName}")
        self.getCameraResolution()
        self.start()

    def setResolution(self, i_Width: int, i_Height: int):
        """
        Set new resolution
        """
        self.stop()
        print(f"Trying to change the resolution to {i_Width}x{i_Height} for camera {self.i_DeviceName}")
        self.CCamera.set(cv.CAP_PROP_FRAME_WIDTH, i_Width) 
        self.CCamera.set(cv.CAP_PROP_FRAME_HEIGHT, i_Height)
        self.getCameraResolution()
        print(f"Resolution set to: {self.t_Shape[1]}x{self.t_Shape[0]} for camera {self.i_DeviceName}")
        self.start()

    def start(self):
        """
        Start camera loop
        """
        self._CameraThread = self.__run()
        self.__waitCameraStart() # Wait for camera to start
        print(f"Camera {self.i_DeviceName} is on!")

    def stop(self):
        """
        Stop camera loop
        """
        if self.b_Run:
            self.b_Run = False
            self._CameraThread.join()
        print(f"Camera {self.i_DeviceName} is off!")
        

    @threaded
    def __run(self):
        self.b_Run = True
        self.b_IsFrame, self.a_Frame = False, None
        while self.b_Run and self.CCamera.isOpened():
            try:
                b_Ret, a_Output = self.CCamera.read()
                if b_Ret: self.b_IsFrame, self.a_Frame = b_Ret, a_Output
            except Exception as E:
                print(f"Exception {E} in camera loop.")

    def __del__(self):
        self.close()


    def __waitCameraStart(self):
        while not self.b_IsFrame:
            time.sleep(0.1)

    
    def close(self):
        """
        Close camera
        """
        self.stop()
        self.CCamera.release()
        print(f"Camera {self.i_DeviceName} was closed!")



    def grabFrame(self):
        """
        Grab new frame from webcam
        """
        b_Res, a_Output = self.CCamera.read()
        if b_Res:
            return np.array(a_Output, dtype=np.uint8)
        else: 
            return np.zeros(self.t_Shape, dtype=np.uint8)

    def displayInLoop(self, s_BreakKey: str ='q', i_TargetFps: int = 60):
        """
        Display webcam feed in a loop
        """
        while True:
            cv.imshow("Output", self.grabFrame())
            if cv.waitKey(1000//i_TargetFps) == ord(s_BreakKey):
                break

    def close(self):
        """
        Close camera
        """
        self.CCamera.release()
        print(f"Camera {self.i_DeviceName} was closed!")

    def getCameraResolution(self):
        """
        Get input resolution
        """
        try:            
            self.t_Shape = (int(self.CCamera.get(cv.CAP_PROP_FRAME_HEIGHT)),int(self.CCamera.get(cv.CAP_PROP_FRAME_WIDTH)),3)
        except:
            self.t_Shape = (-1,-1,-1)
        return self.t_Shape