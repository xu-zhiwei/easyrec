import tensorflow as tf
from tensorflow.keras.activations import sigmoid

from easyrec import blocks


class AFM(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 k=16):
        super(AFM, self).__init__()
        if not one_hot_feature_columns:
            raise ValueError('len(one_hot_feature_columns) should greater than 0')
        self.afm = blocks.AFM(one_hot_feature_columns, k=k)

    def call(self, inputs, training=None, mask=None):
        logits = self.afm(inputs)
        return sigmoid(logits)
