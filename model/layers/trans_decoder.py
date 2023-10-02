import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class TransformerDecoder(layers.Layer):
    def __init__(self, keyDim, heads, feedForwardDim, rate=0.1):
        super().__init__()

        self.layerNormal1 = layers.Normalization() # originally epsilon=1e-6
        self.layerNormal2 = layers.Normalization() # originally epsilon=1e-6
        self.layerNormal3 = layers.Normalization() # originally epsilon=1e-6

        self.self_atten = layers.MultiHeadAttention(num_heads=heads, key_dim=keyDim)

        self.self_drop = layers.Dropout(0.5)
        self.enc_dec_drop = layers.Dropout(0.1)
        self.ffn_drop = layers.Dropout(0.1)

        self.enc_atten = layers.MultiHeadAttention(num_heads=heads, key_dim=keyDim)

        self.ffn = keras.Sequential(
            [
                layers.Dense(feedForwardDim, activation="relu"),
                layers.Dense(keyDim),
            ]
        )
    
    def casual_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest

        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        multi = tf.concat([batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0)
        return  tf.tile(mask, multi)
    
    def call(self, enc_outputs, target, training):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        mask = self.casual_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_atten = self.self_atten(target, target, attention_mask=mask)
        target_normal = self.layerNormal1(target + self.self_drop(target_atten, training=training))

        enc_outputs = self.enc_atten(target_normal, enc_outputs)
        enc_norm = self.layerNormal2(self.enc_dec_drop(enc_outputs, training=training) + target_normal)
        ffn_out = self.ffn(enc_norm)
        ffn_norm = self.layerNormal3(enc_norm + self.ffn_drop(ffn_out, training=training))

        return ffn_norm
