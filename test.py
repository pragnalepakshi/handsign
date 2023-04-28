import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
#to get matrix form of image captured

import math
import time
import tensorflow

cap = cv2.VideoCapture(0)
#here is indicated as webcam index
detector = HandDetector(maxHands=1)
#only hand is detected at a time
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
# fixing image size

#folder = "Data/C"
#counter = 0

labels = ['A', 'B', 'C']

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        #to capture one hand

        x, y, w, h = hand['bbox']
        # The bounding box is defined by its top-left corner coordinates (x, y), and its width and height (w, h)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            # When the height is greater than the width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # This will give us the prediction and the index
            print(prediction, index)



        else:
            #when width is greater than the width
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # This will give us the prediction and the index
            print(prediction, index)
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255, 0, 255),
                      cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

# Now the webcam is able to detect left and right hands

