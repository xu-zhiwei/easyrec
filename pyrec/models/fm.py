import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import DenseFeatures, Dense

import pyrec.layers as my_layers


class FM(tf.keras.models.Model):
    """
    Factorization Machines

    """
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 one_hot_shape,
                 multi_hot_shape,
                 k=16,
                 use_dense_feature_columns=False,
                 ):
        super(FM, self).__init__()
        self.use_dense_feature_columns = use_dense_feature_columns
        if one_hot_feature_columns and multi_hot_feature_columns:
            self.fm_input_layer = DenseFeatures(feature_columns=one_hot_feature_columns + multi_hot_feature_columns)
        elif one_hot_feature_columns:
            self.fm_input_layer = DenseFeatures(feature_columns=one_hot_feature_columns)
        elif multi_hot_feature_columns:
            self.fm_input_layer = DenseFeatures(feature_columns=multi_hot_feature_columns)
        else:
            raise ValueError('len(one_hot_feature_columns) + len(multi_hot_feature_columns) should greater than 0')
        if self.use_dense_feature_columns:
            self.dense_input_layer = DenseFeatures(feature_columns=dense_feature_columns)
            self.fc = Dense(units=1)
        self.fm = my_layers.FM(input_dimension=one_hot_shape + multi_hot_shape, k=k)

    def call(self, inputs, training=None, mask=None):
        fm_inputs = self.fm_input_layer(inputs)
        logits = self.fm(fm_inputs)
        if self.use_dense_feature_columns:
            dense_inputs = self.dense_input_layer(inputs)
            logits += self.fc(dense_inputs)
        return sigmoid(logits)
