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

def getImageGenerators(training_labels, training_images, test_labels, test_images):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = datagen.flow(x=training_images, y=training_labels, batch_size=32)
    test_generator = test_datagen.flow(x=test_images, y=test_labels, batch_size=32)

    return train_generator, test_generator

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    return model

train_labels, train_images = form_datasets(train_dataframe['label'].to_numpy(), train_dataframe.drop('label', axis=1).to_numpy())
test_labels, test_images = form_datasets(test_dataframe['label'].to_numpy(), test_dataframe.drop('label', axis=1).to_numpy())

train_images = np.expand_dims(train_images, axis=3) # Expanding Dimension for Convolutional Layer
test_images = np.expand_dims(test_images, axis=3) # Expanding Dimension for Convolutional Layer

train_generator, test_generator = getImageGenerators(train_labels, train_images, test_labels, test_images)

model = build_model()

history = model.fit(train_generator, epochs=20, validation_data=test_generator)