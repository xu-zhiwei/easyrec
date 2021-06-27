import tensorflow as tf

import pyrec.layers as my_layers


class FM(tf.keras.models.Model):
    """
    Factorization Machines

    """
    def __init__(self, learning_rate):
        super(FM, self).__init__()
        pass

    def call(self, inputs, training=None, mask=None):
        return my_layers.SimplifiedFM(inputs)
