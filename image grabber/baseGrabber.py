from pypylon import pylon
import cv2
from PIL import Image
import numpy as np
import random as rand
import os

os.environ["XDG_SESSION_TYPE"] = "xcb"
# The name of the pylon file handle
nodeFile = "NodeMap.pfs"
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
#camera.ExposureTime.SetValue(1000)
# Save the content of the camera's node map into the file.
#pylon.FeaturePersistence.Save(nodeFile, camera.GetNodeMap())
#pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap())
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

if grabResult.GrabSucceeded():
    image = converter.Convert(grabResult)# threshold to binary
    img = image.GetArray()# Load image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    thresh= cv2.bitwise_not(thresh)
    thresh = cv2.erode(thresh, None, iterations=2)
    contour_img = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    index = 1
    isolated_count = 0
    cluster_count = 0
    ROI=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blobFound=0
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > 100:
            convex_hull = cv2.convexHull(cntr)
            convex_hull_area = cv2.contourArea(convex_hull)
            print(index, area, convex_hull_area)
            ratio = area / convex_hull_area
            #print(index, area, convex_hull_area, ratio)
            #x,y,w,h = cv2.boundingRect(cntr)
            #cv2.putText(label_img, str(index), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
            print(ratio)
            if ratio < 0.80:
                # cluster contours in red
                cv2.drawContours(contour_img, [cntr], 0, (0,0,255), 2)
                cluster_count = cluster_count + 1
            else:
                # isolated contours in green
                cv2.drawContours(contour_img, [cntr], 0, (0,255,0), 2)
                isolated_count = isolated_count + 1
                x, y, w, h = cv2.boundingRect(cntr)
                ROI = gray[y-10:y + h+20, x-10:x + w+20]
                blobFound+=1
            index = index + 1
    # Show the output
    if blobFound==0:
        ROI=gray
    cv2.imshow("Blobs Detected", img)
    #cv2.pollKey()
    cv2.waitKey(250)
    cv2.destroyAllWindows()
    image = Image.fromarray(ROI.astype(np.uint8))
    #image.show()
    ##path= str(rand.randint(0,1000))
    path="/home/marla/repo/test pytorch/image grabber/"
    path += str(rand.randint(0, 1000))
    path+=".bmp"
    image.save(path)

    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()

cv2.destroyAllWindows()