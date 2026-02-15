"""
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

"""
from pypylon import pylon
import cv2
from PIL import Image
import numpy as np
import random as rand
import os

def GetThressholdedImage(inImg):
    gray = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)

    cannyImg=cv2.Canny(cv2.blur(gray,(3,3)),10,255)

    _,thresh=cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

    kernelClose= np.ones((11,11),np.float32)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernelClose)

    kernelOpen= np.ones((5,5),np.float32)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernelClose)

    #kernelErode = np.ones((7,7),np.float32)
    #thresh=cv2.erode(thresh,kernelErode)

    #kernelDilate = np.ones((7,7),np.float32)
    #thresh=cv2.dilate(thresh,kernelDilate)

    return thresh,cannyImg

def GetBlobs(inImg):
    # Find contours
    contours = cv2.findContours(inImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # Draw contours

    outimg=np.copy(inImg)
    points=None
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area>40000:
            cv2.drawContours(outimg, [cntr], 0, (0,255,0), 2)
            x,y,w,h = cv2.boundingRect(cntr)
            points=(x,y,h,w)
    #cv2.namedWindow("Blob", cv2.WINDOW_NORMAL)
    #cv2.imshow("Blob", outimg)
    return points, outimg



os.environ["XDG_SESSION_TYPE"] = "xcb"
nodeFile = "NodeMap.pfs"
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
camera.ExposureTime.SetValue(800)
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
clippedImg=None
offset=25
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)# threshold to binary
        img = image.GetArray()# Load image
        threshImg,cannyImg=GetThressholdedImage(img)

        points,blobImg=GetBlobs(threshImg)
        imgShow=np.copy(img)
        if points!=None:
            pt1x=points[0]-offset
            pt1y=points[1]-offset
            pt2x=points[0]+points[2]+offset
            pt2y=points[1]+points[3]+offset
            cv2.rectangle(imgShow,(pt1x,pt1y),(pt2x,pt2y),(0,255,0),3)

        

        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        cv2.imshow("Threshold", threshImg)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera", imgShow)


        k = cv2.waitKey(100)
        if k == 27:
            break
        grabResult.Release()
image = Image.fromarray(img.astype(np.uint8))
image=image.crop((pt1x,pt1y,pt2x,pt2y))
path="/home/marla/repo/test pytorch/image grabber/out/"
path += str(rand.randint(0, 1000))
path+=".bmp"
print(path)
image.save(path)
cv2.destroyAllWindows()
# Releasing the resource
camera.StopGrabbing()
