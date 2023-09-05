import cv2 as cv
import mediapipe as mp
import mediapipe.tasks as mpt
import mediapipe.tasks.python as mptv

class HandSegmenter:
    def __init__(self, maxHands = 1, boundingBoxOffset = (20, 20)):
        self.numHands = maxHands
        self.offset = boundingBoxOffset

        base_options = mpt.BaseOptions()
        handOptions = mptv.HandLandmarkOptions(
            base_options=base_options, 
            num_hands=self.maxNumHands,
        )

        self.detector = mptv.HandLandmarkDetector.create_from_options(options=handOptions)
    
    def preprocess(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return gray

    def detect(self, image):
        results = self.detector.detect(image)
        return results
