import numpy as np
from PIL import Image
import mediapipe as mp
import cv2 as cv

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    _, image = cap.read()
    gray = image.cvtColor(image, cv.COLOR_BGR2GRAY)

    results = hands.process(gray)

    if results.multi_hand_landmarks:
        for _, landmarks in enumerate(results.multi_hand_landmarks):
            h, w, c = image.shape
            cx, cy = int(landmarks.x * w), int(landmarks.y * h)

            cv.circle(image, (cx, cy), 5, (255, 0, 0), cv.FILLED)

    
    cv.imshow("Image", image)