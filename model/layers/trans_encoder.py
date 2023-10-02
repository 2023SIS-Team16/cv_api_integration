import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class TransformerEncoder(layers.Layer):
    def __init__(self, keyDim, heads, feedForwardDimension, rate=0.1):
        super().__init__()

        self.atten = layers.MultiHeadAttention(num_heads=heads, key_dim=keyDim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feedForwardDimension, activation="relu"), 
                layers.Dense(keyDim),
            ]
        )

        self.layerNormal1 = layers.LayerNormalization() # originally epsilon=1e-6
        self.layerNormal2 = layers.LayerNormalization() # originally epsilon=1e-6

        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)
    
    def call(self, inputs, training):
        atten_out = self.atten(inputs, inputs)
        atten_out = self.drop1(atten_out, training=training)

        out = self.layerNormal1(inputs + atten_out)
        ffn_out = self.ffn(out)
        ffn_out = self.drop2(ffn_out, training=training)

        return self.layerNormal2(out + ffn_out)
