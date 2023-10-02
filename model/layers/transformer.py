from layers.embedding import TokenEmbeddingLayer, LandmarkEmbeddingLayer
from layers.trans_encoder import TransformerEncoder
from layers.trans_decoder import TransformerDecoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class Transformer(keras.Model):
    def __init__(
            self, 
            hid = 64,
            heads = 2,
            numFeedForward = 128,
            source_maxlen = 100,
            target_maxlen = 100,
            num_enc_layers = 4,
            num_dec_layers = 1,
            num_class = 60,
    ):
        super().__init__()

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.accuracy_metric = keras.metrics.Mean(name="edit_dist") # Probably need to change this to something betetr
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        self.target_maxlen = target_maxlen
        self.num_class = num_class

        self.enc_input = LandmarkEmbeddingLayer(hid=hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbeddingLayer(num_vocab=num_class, maxlen=target_maxlen, hid=hid)

        self.encoder = keras.Sequential(
            [self.enc_input]
            +
            [
                TransformerEncoder(hid, heads, numFeedForward) for _ in range(num_enc_layers)
            ]
        )

        for i in range(num_dec_layers):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(hid, heads, numFeedForward)
            )
        
        self.classifier = layers.Dense(num_class)
    
    def decode(self, enc_outputs, target, training):
        y = self.dec_input(target)
        for i in range(self.num_dec_layers):
            y = getattr(self, f"dec_layer_{i}")(enc_outputs, y, training)
        return y
    
    def call(self, inputs, training):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source, training)
        y = self.decode(x, target, training)
        return self.classifier(y)
    
    @property
    def metrics(self):
        return [self.loss_metric, self.accuracy_metric] # Should I be providing the accurascy metric
    
    def train_step(self, batch):
        source = batch[0]
        target = batch[1]

        input_shape = tf.shape(target)
        batch_size = input_shape[0]

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        with tf.GradientTape() as tape:
            preds = self([source, target_input])
            one_hot = tf.one_hot(target_output, depth=self.num_class)
            mask = tf.math.logical_not(tf.math.equal(target_output, 59)) # 59 = Pad token index
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        edit_distance = tf.edit_distance(tf.sparse.from_dense(target), tf.sparse.from_dense(tf.cast(tf.argmax(preds, axis=2), tf.int32)))
        edit_distance = tf.reduce_mean(edit_distance)
        self.accuracy_metric.update_state(edit_distance)
        self.loss_metric.update_state(loss)

        return {"loss": self.loss_metric.result(), "edit_dist": self.accuracy_metric.result()}
    
    def test_step(self, batch):
        source = batch[0]
        target = batch[1]

        input_shape = tf.shape(target)
        batch_size = input_shape[0]

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        preds = self([source, target_input])
        one_hot = tf.one_hot(target_output, depth=self.num_class)
        mask = tf.math.logical_not(tf.math.equal(target_output, 59))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)

        edit_distance = tf.edit_distance(tf.cast(tf.argmax(preds, axis=2), tf.int32), target_output, normalize=False)
        edit_distance = tf.reduce_mean(edit_distance)

        self.accuracy_metric.update_state(edit_distance)
        self.loss_metric.update_state(loss)

        return {"loss": self.loss_metric.result(), "edit_dist": self.accuracy_metric.result()}
    
    def generate(self, source, target_start_token_index):
        bs = tf.shape(source)[0]
        encoder = self.encoder(source, training=False)
        decod_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_index
        dec_logits = []

        for i in range(self.target_maxlen - 1):
            decod_out = self.decode(encoder, decod_input, training=False)
            logits = self.classifier(decod_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_logits.append(last_logit)
            decod_input = tf.concat([decod_input, last_logit], axis=-1)
        
        return decod_input