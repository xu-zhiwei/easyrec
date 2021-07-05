import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import DenseFeatures, Dense

import pyrec.layers as my_layers


class FFM(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 k=4,
                 ):
        super().__init__(self, FFM)
        if one_hot_feature_columns and multi_hot_feature_columns:
            self.fm_input_layer = DenseFeatures(feature_columns=one_hot_feature_columns + multi_hot_feature_columns)
        elif one_hot_feature_columns:
            self.fm_input_layer = DenseFeatures(feature_columns=one_hot_feature_columns)
        elif multi_hot_feature_columns:
            self.fm_input_layer = DenseFeatures(feature_columns=multi_hot_feature_columns)
        else:
            raise ValueError('len(one_hot_feature_columns) + len(multi_hot_feature_columns) should greater than 0')


    def call(self, inputs, training=None, mask=None):
        pass

