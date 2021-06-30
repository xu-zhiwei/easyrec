import tensorflow as tf


class FM(tf.keras.layers.Layer):
    """
    Factorization machine layer using vector w and matrix v.
    """

    def __init__(self, input_dimension, k):
        super(FM, self).__init__()
        self.k = k
        self.w = tf.Variable(tf.random.normal(shape=(input_dimension, 1)), name='w')
        self.v = tf.Variable(tf.random.normal(shape=(input_dimension, self.k)), name='v')

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs: [batch_size, dimension]
        Returns:
            logits
        """
        logits = tf.squeeze(tf.matmul(inputs, self.w))
        square_of_sum = tf.square(tf.matmul(inputs, self.v))
        sum_of_square = tf.matmul(tf.square(inputs), tf.square(self.v))
        logits += 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1)
        return logits
