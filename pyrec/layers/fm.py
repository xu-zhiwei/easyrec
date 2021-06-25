import tensorflow as tf


class SimplifiedFM(tf.keras.layers.Layer):
    """
    Simplified factorization machine layer using square of sum and sum of square.
    """
    def __init__(self):
        super(SimplifiedFM, self).__init__()

    def call(self, inputs, *args, **kwargs):
        """
        Args:
            inputs: [batch, ]
        """
        pass


class FM(tf.keras.layers.Layer):
    """
    Basic factorization machine layer using vector w and matrix v.
    """
    def __init__(self, w=None, v=None):
        super(FM, self).__init__()
        if not w:
            w = tf.Variable()

    def call(self, inputs, *args, **kwargs):
        pass

