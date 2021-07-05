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
        super(FFM, self).__init__()
        if one_hot_feature_columns and multi_hot_feature_columns:
            self.input_layer = DenseFeatures(feature_columns=one_hot_feature_columns + multi_hot_feature_columns)
        elif one_hot_feature_columns:
            self.input_layer = DenseFeatures(feature_columns=one_hot_feature_columns)
        elif multi_hot_feature_columns:
            self.input_layer = DenseFeatures(feature_columns=multi_hot_feature_columns)
        else:
            raise ValueError('len(one_hot_feature_columns) + len(multi_hot_feature_columns) should greater than 0')
        self.ffm = my_layers.FFM(one_hot_feature_columns, multi_hot_feature_columns, k)

    def call(self, inputs, training=None, mask=None):
        inputs = self.input_layer(inputs)
        logits = self.ffm(inputs)
        return sigmoid(logits)

