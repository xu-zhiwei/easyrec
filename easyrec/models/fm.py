import tensorflow as tf
from tensorflow.keras.activations import sigmoid

from easyrec import blocks


class FM(tf.keras.models.Model):
    """
    Factorization Machines.

    """

    def __init__(self,
                 one_hot_feature_columns,
                 k=16,
                 ):
        super(FM, self).__init__()
        if not one_hot_feature_columns:
            raise ValueError('len(one_hot_feature_columns) should greater than 0')
        self.fm = blocks.FM(one_hot_feature_columns, k=k)

    def call(self, inputs, training=None, mask=None):
        logits = self.fm(inputs)
        return sigmoid(logits)
