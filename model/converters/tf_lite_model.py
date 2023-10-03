import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

# This is taken and modified from https://www.kaggle.com/code/gusthema/asl-fingerspelling-recognition-w-tensorflow

class TFLiteModel(tf.Module):
    def __init__(self, model, preprocess_fn):
        super(TFLiteModel, self).__init__()
        self.target_start_token_idx = 60
        self.target_end_token_idx = 61
        # Load the feature generation and main models
        self.model = model
        self.preprocess_fn = preprocess_fn
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 156], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        # Preprocess Data
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, 156)), lambda: tf.identity(x))
        x = x[0]
        x = self.preprocess_fn(x)
        x = x[None]
        x = self.model.generate(x, self.target_start_token_idx)
        x = x[0]
        idx = tf.argmax(tf.cast(tf.equal(x, self.target_end_token_idx), tf.int32))
        idx = tf.where(tf.math.less(idx, 1), tf.constant(2, dtype=tf.int64), idx)
        x = x[1:idx]
        x = tf.one_hot(x, 59)
        return {'outputs': x}