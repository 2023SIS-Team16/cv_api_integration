import tensorflow as tf
import cv2 as cv
import numpy as np
import PIL.Image as Image

def resize_to_28x28(img):
    img_h, img_w = img.shape
    dim_size_max = max(img.shape)

    if dim_size_max == img_w:
        im_h = (26 * img_h) // img_w
        if im_h <= 0 or img_w <= 0:
            print("Invalid Image Dimention: ", im_h, img_w, img_h)
        tmp_img = cv.resize(img, (26,im_h),0,0,cv.INTER_NEAREST)
    else:
        im_w = (26 * img_w) // img_h
        if im_w <= 0 or img_h <= 0:
            print("Invalid Image Dimention: ", im_w, img_w, img_h)
        tmp_img = cv.resize(img, (im_w, 26),0,0,cv.INTER_NEAREST)

    out_img = np.zeros((28, 28), dtype=np.ubyte)

    nb_h, nb_w = out_img.shape
    na_h, na_w = tmp_img.shape
    y_min = (nb_w) // 2 - (na_w // 2)
    y_max = y_min + na_w
    x_min = (nb_h) // 2 - (na_h // 2)
    x_max = x_min + na_h

    out_img[x_min:x_max, y_min:y_max] = tmp_img

    return out_img

model = tf.keras.models.load_model('export_model/sign_language_model')
print(model.summary())

# cap = cv.VideoCapture(0)

# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# while True:
#     ret, capFrame = cap.read()

capFrame = cv.imread('test_image_1.jpg')
frame = cv.cvtColor(capFrame, cv.COLOR_BGR2GRAY)
frame = resize_to_28x28(frame)

data = np.copy(frame).reshape(1,28,28,1)
Image.fromarray(data.reshape(28,28), 'L').show()

prediction = model.predict(data)

print(prediction)
    # cv.imshow('Thing', capFrame)