import tensorflow as tf
from tensorflow.keras.activations import sigmoid

from easyrec import blocks


class NFM(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 k=32
                 ):
        super(NFM, self).__init__()
        if not one_hot_feature_columns:
            raise ValueError('len(one_hot_feature_columns) should greater than 0')
        self.nfm = blocks.NFM(one_hot_feature_columns, k=k)

    def call(self, inputs, training=None, mask=None):
        logits = self.nfm(inputs)
        return sigmoid(logits)
