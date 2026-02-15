from pypylon import pylon
import cv2
import numpy as np
import os
import time

import torch
import torchvision.transforms as transforms
from model import CNNModel

os.environ["XDG_SESSION_TYPE"] = "xcb"
device = ('cuda' if torch.cuda.is_available() else 'cpu')

def GetThressholdedImage(inImg):
    gray = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator
    sobelx = cv2.Sobel(cv2.blur(gray,(5,5)), cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(cv2.blur(gray,(5,5)), cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    # Convert to uint8
    edgeImage = cv2.convertScaleAbs(gradient_magnitude)
    edgeImage=cv2.convertScaleAbs(edgeImage,alpha=3,beta=0)
    edgeImage=cv2.blur(edgeImage,(5,5))
    hist = cv2.calcHist([edgeImage],[0],None,[256],[0,256])
    sum = 0
    h, w = img.shape[:2]
    tot = h * w
    for i in range (0,256):
        sum = sum + i*hist[i][0]
    mean = sum/tot
    _,thresh=cv2.threshold(edgeImage,mean,255,cv2.THRESH_BINARY)

    kernelClose= np.ones((11,11),np.float32)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernelClose)

    kernelOpen= np.ones((5,5),np.float32)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernelClose)

    return thresh,edgeImage

def getBlobs(inImg):
    # Find contours
    contours = cv2.findContours(inImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # Draw contours
    outimg=np.copy(inImg)
    for cntr in contours:
        area = cv2.contourArea(cntr)
        M= cv2.moments(cntr)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00']) 

        if area>40000 and cx>100 and cy >100:
            cv2.drawContours(outimg, [cntr], 0, (0,255,0), 2)
            x,y,w,h = cv2.boundingRect(cntr)
            points=(x,y,h,w)
            return points, outimg
    return None, outimg

def inferBlob(imgIn):
    imageRGB = cv2.cvtColor(imgIn, cv2.COLOR_BGR2RGB)
    imageRGB = transform(imageRGB)
    # add batch dimension
    imageRGB = torch.unsqueeze(imageRGB, 0)
    with torch.no_grad():
        outputs = model(imageRGB.to(device))
        sm = torch.nn.functional.softmax(outputs, dim=1)
        probability1 = sm.data.max(1, keepdim=True)[0].item()


    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]
    return pred_class,round(probability1*100,2)

# list containing all the class labels
labels = ['ligne','mat']

model = CNNModel().to(device)
checkpoint = torch.load('/home/marla/repo/test pytorch/outputs/model.pth', map_location=device,weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])  
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
        startTime = time.perf_counter()
        # Access the image data
        image = converter.Convert(grabResult)# threshold to binary
        img = image.GetArray()# Load image
        
        threshImg,cannyImg=GetThressholdedImage(img)

        points,blobImg=getBlobs(threshImg)

        imgShow=np.copy(img)

        if points!=None:
            print(points[0])
            print(points[1])

            if points[0]>offset and points[1] > offset :
                pt1x=points[0]-offset
                pt1y=points[1]-offset
                pt2x=points[0]+points[2]+offset
                pt2y=points[1]+points[3]+offset
                cv2.rectangle(img,(pt1x,pt1y),(pt2x,pt2y),(0,0,255),3)
                pred_class,probability=inferBlob(imgShow[pt1y:pt2y,pt1x:pt2x])
                cv2.putText(img,f"Pred: {pred_class}",(10, 20),0,0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img,f"Pred: {probability}",(10, 50),0,0.6, (0, 0, 255), 2, cv2.LINE_AA)
        stopTime=time.perf_counter()
        print(f"executionTime {stopTime - startTime:0.4f} seconds")
        cv2.imshow("Canny", cannyImg)   
        cv2.imshow("Threshold", threshImg)
        cv2.imshow('Result', img)
        k = cv2.waitKey(100)
        if k == 27:
            break
        grabResult.Release()

cv2.destroyAllWindows()
# Releasing the resource
camera.StopGrabbing()
