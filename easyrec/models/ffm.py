import tensorflow as tf
from tensorflow.keras.activations import sigmoid

from easyrec import blocks


class FFM(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 k=4,
                 ):
        super(FFM, self).__init__()
        if not one_hot_feature_columns:
            raise ValueError('len(one_hot_feature_columns) should greater than 0')
        self.ffm = blocks.FFM(one_hot_feature_columns, k)

    def call(self, inputs, training=None, mask=None):
        logits = self.ffm(inputs)
        return sigmoid(logits)
