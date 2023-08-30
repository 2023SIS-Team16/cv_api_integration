import numpy as np
from PIL import Image
import mediapipe as mp
import cv2 as cv

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    _, image = cap.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            h, w, c = image.shape
            x_min=w
            y_min=h
            x_max = y_max = 0
            for id,lm in enumerate(handLandmarks.landmark):
                print(lm)
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                # cv.circle(image, (cx, cy), 5, (255, 0, 0), cv.FILLED)
            cv.rectangle(image, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)

    
    cv.imshow("Image", image)
    if cv.waitKey(1) == ord('q'):
        break