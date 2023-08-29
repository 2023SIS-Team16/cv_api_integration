import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import PIL.Image as Image


def load_dataframe(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

train_path = '/home/jon/development/university/sis/cv_api/data/sign_mnist/sign_mnist_train.csv'
test_path = '/home/jon/development/university/sis/cv_api/data/sign_mnist/sign_mnist_test.csv'

train_dataframe = load_dataframe(train_path)
test_dataframe = load_dataframe(test_path)

# train_labels = train_dataframe['label'].to_numpy()
# train_images = train_dataframe.drop('label', axis=1).to_numpy()

def form_datasets(labels, images):
    if len(labels) != len(images):
        raise Exception('The number of labels and images are not the same.')

    output_labels = []
    output_images = []

    for x in range(len(labels)):
        label = labels[x]
        image = images[x]
        image = image.reshape((28, 28))

        output_labels.append(label)
        output_images.append(image)

    return np.array(output_labels).astype(float), np.array(output_images).astype(float)

train_labels, train_images = form_datasets(train_dataframe['label'].to_numpy(), train_dataframe.drop('label', axis=1).to_numpy())
test_labels, test_images = form_datasets(test_dataframe['label'].to_numpy(), test_dataframe.drop('label', axis=1).to_numpy())

train_images = np.expand_dims(train_images, axis=3) # Expanding Dimension for Convolutional Layer
test_images = np.expand_dims(test_images, axis=3) # Expanding Dimension for Convolutional Layer

def getImageGenerator(training_labels, training_images, test_labels, test_images):
    pass