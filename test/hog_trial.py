import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't read frame")
        break
    
    procImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    procImg = cv.GaussianBlur(procImg, (5, 5), 0)
    # procImg = cv.threshold(procImg, 80, 255, cv.THRESH_BINARY)

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    (humans, weights) = hog.detectMultiScale(procImg, padding = (8, 8), scale = 1.01)
    print("humans: ", len(humans))

    grouped, newWeights = cv.groupRectangles(humans, 1, 0.1)
    print(len(grouped))

    for (x,y,w,h) in grouped:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()