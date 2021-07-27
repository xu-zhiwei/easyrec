import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense


class LR(tf.keras.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 ):
        super(LR, self).__init__()
        self.input_layer = DenseFeatures(
            feature_columns=one_hot_feature_columns + multi_hot_feature_columns + dense_feature_columns)
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        return self.score(x)
