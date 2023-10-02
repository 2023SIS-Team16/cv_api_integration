import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.target_start_token_index = 60
        self.target_end_token_index = 61

        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(FEA)])])