from landmarks import LandmarkProcessor
import cv2 as cv
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

images = []
for i in range(1, 102):
    images.append(cv.imread(f"/Users/jon/development/university/sis/videos/fig_1/frame_{i:04}.png"))

processor = LandmarkProcessor(
    pose_landmarker="/Users/jon/development/university/sis/models/pose_landmarker_full.task",
    hand_landmarker="/Users/jon/development/university/sis/models/hand_landmarker.task",
    face_landmarker="/Users/jon/development/university/sis/models/face_landmarker.task"
)

HAND_INDICES = list(range(0, 21))
POSE_INDICES = [12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21]
FACE_INDICES = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 
                405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317,
                402, 318, 324, 308]
FACE_INDICES_BREAK = [ 1, 2, 98, 327, 33, 7, 163, 144, 145, 153,
    154, 155, 133, 246, 161, 160, 159, 
    158, 157, 173, 263, 249, 390, 373, 
    374, 380, 381, 382, 362, 
    466, 388, 387, 386, 385, 384, 398
]

prediction_data = []

for i in range(len(images)):
    prediction_data.append([])

    image = images[i]
    cv.cvtColor(image, cv.COLOR_BGR2RGB)

    pose_land, hand_land, handedness, face_land = processor.get_landmarks(images[i])
    include = False
    if hand_land != [] and len(hand_land) >= 1:
        include = True
        rhand = {
            "x": [hand_land[0][i].x for i in HAND_INDICES], 
            "y": [hand_land[0][i].y for i in HAND_INDICES], 
            "z": [hand_land[0][i].z for i in HAND_INDICES]
        }

        print(i)

        lhand = {
            "x": list(np.empty(len(HAND_INDICES))),
            "y": list(np.empty(len(HAND_INDICES))),
            "z": list(np.empty(len(HAND_INDICES))),
        }
    else: 
        rhand = {
            "x": list(np.empty(len(HAND_INDICES))),
            "y": list(np.empty(len(HAND_INDICES))),
            "z": list(np.empty(len(HAND_INDICES))),
        }

        lhand = {
            "x": list(np.empty(len(HAND_INDICES))),
            "y": list(np.empty(len(HAND_INDICES))),
            "z": list(np.empty(len(HAND_INDICES))),
        }

    pose = {
        "x": [pose_land[i].x for i in POSE_INDICES], 
        "y": [pose_land[i].y for i in POSE_INDICES], 
        "z": [pose_land[i].z for i in POSE_INDICES]
    }

    if face_land != []:
        face = {
            "x": [face_land[i].x for i in FACE_INDICES], 
            "y": [face_land[i].y for i in FACE_INDICES], 
            "z": [face_land[i].z for i in FACE_INDICES]
        }

        face_2 = {
            "x": [face_land[i].x for i in FACE_INDICES_BREAK], 
            "y": [face_land[i].y for i in FACE_INDICES_BREAK], 
            "z": [face_land[i].z for i in FACE_INDICES_BREAK]
        }
    else:
        face = {
            "x": list(np.empty(len(FACE_INDICES))),
            "y": list(np.empty(len(FACE_INDICES))),
            "z": list(np.empty(len(FACE_INDICES))),
        }

        face_2 = {
            "x": list(np.empty(len(FACE_INDICES_BREAK))),
            "y": list(np.empty(len(FACE_INDICES_BREAK))),
            "z": list(np.empty(len(FACE_INDICES_BREAK))),
        }

    if include:
        prediction_data[i].extend(face["x"])
        prediction_data[i].extend(lhand["x"])
        prediction_data[i].extend(rhand["x"])
        prediction_data[i].extend(face_2["x"])
        prediction_data[i].extend(pose["x"])

        prediction_data[i].extend(face["y"])
        prediction_data[i].extend(lhand["y"])
        prediction_data[i].extend(rhand["y"])
        prediction_data[i].extend(face_2["y"])
        prediction_data[i].extend(pose["y"])

        prediction_data[i].extend(face["z"])
        prediction_data[i].extend(lhand["z"])
        prediction_data[i].extend(rhand["z"])
        prediction_data[i].extend(face_2["z"])
        prediction_data[i].extend(pose["z"])

prediction_data = [x for x in prediction_data if x != []]

import tensorflow as tf
import json

prediction_data = tf.cast(prediction_data, tf.float32)

# interpreter = tf.lite.Interpreter(model_path=f"/Users/jon/development/university/sis/models/v0_0_1_model_export_tflite/asl_model.tflife")
interpreter = tf.lite.Interpreter(model_path=f"/Users/jon/development/university/sis/models/model.tflite")
# interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print(input_details)

chars_path = "/Users/jon/development/university/sis/character_to_prediction_index.json"

with open(chars_path, "r") as f:
   character_map = json.load(f)
   rev_character_map = {j:i for i,j in character_map.items()}

found_signatures = list(interpreter.get_signature_list().keys())
print(found_signatures)
REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

if REQUIRED_SIGNATURE not in found_signatures:
    raise Exception('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")

print(np.array(prediction_data[0:1]).shape)

output = prediction_fn(inputs=np.array(prediction_data[0]))

prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
print(prediction_str)
