import tensorflow as tf
from tensorflow.keras.layers import Dense


class DenseBlock(tf.keras.models.Model):
    def __init__(self, units_list=None, activation=None):
        super(DenseBlock, self).__init__()
        if units_list is None:
            units_list = [256, 128, 64]
        self.fcs = [Dense(units=units, activation=activation) for units in units_list]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for fc in self.fcs:
            x = fc(x)
        return x


class ResidualBlock(tf.keras.models.Model):
    def __init__(self, units_list=None, activation=None):
        super(ResidualBlock, self).__init__()
        if units_list is None:
            units_list = [256, 256]
        self.dense_block = DenseBlock(units_list, activation)

    def call(self, inputs, training=None, mask=None):
        return inputs + self.dense_block(inputs)


class SelfAttention(tf.keras.models.Model):
    def __init__(self, input_dimension, qkv_dimension, use_normalization=True):
        super(SelfAttention, self).__init__()
        self.use_normalization = use_normalization
        self.q = tf.Variable(tf.random.normal((input_dimension, qkv_dimension)), name='q')
        self.k = tf.Variable(tf.random.normal((input_dimension, qkv_dimension)), name='k')
        self.v = tf.Variable(tf.random.normal((input_dimension, qkv_dimension)), name='v')

    def call(self, inputs, training=None, mask=None):
        q = tf.matmul(inputs, self.q)
        k = tf.matmul(inputs, self.k)
        v = tf.matmul(inputs, self.v)

        weights = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        if self.use_normalization:
            weights /= tf.sqrt(float(q.shape[2]))
        weights = tf.nn.softmax(weights)

        return tf.matmul(weights, v)


class MultiHeadSelfAttention(tf.keras.models.Model):
    def __init__(self, input_dimension, qkv_dimension, num_heads, output_dimension, use_normalization=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = [SelfAttention(input_dimension, qkv_dimension, use_normalization) for _ in range(num_heads)]
        self.w = tf.Variable(tf.random.normal((num_heads * qkv_dimension, output_dimension)))

    def call(self, inputs, training=None, mask=None):
        return tf.matmul(tf.concat([head(inputs) for head in self.heads], axis=2), self.w)
