import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class TokenEmbeddingLayer(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=hid)
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
    
class LandmarkEmbeddingLayer(layers.Layer):
    def __init__(self, hid=64, maxlen=100):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv1D(hid, 11, strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(hid, 11, strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(hid, 11, strides=2, padding='same', activation='relu')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
