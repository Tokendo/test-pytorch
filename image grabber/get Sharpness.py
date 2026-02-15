from pypylon import pylon
import cv2
from PIL import Image
import numpy as np
import random as rand
import os

def getBlurValue(image):
    canny = cv2.Canny(image, 50,250)
    return np.mean(canny)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

os.environ["XDG_SESSION_TYPE"] = "xcb"

camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        blur=getBlurValue(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(round(blur,2)),(100,100), font, 4,(255,0,0),2,cv2.LINE_AA)
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()