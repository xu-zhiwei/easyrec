import tensorflow as tf
from tensorflow.keras.layers import Dense


class DenseBlock(tf.keras.models.Model):
    def __init__(self, hidden_units=None, activation=None):
        super(DenseBlock, self).__init__()
        if hidden_units is None:
            hidden_units = [256, 128, 64]
        self.fcs = [Dense(units=units, activation=activation) for units in hidden_units]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for fc in self.fcs:
            x = fc(x)
        return x


class ResidualBlock(tf.keras.models.Model):
    def __init__(self, hidden_units=None, activation=None):
        super(ResidualBlock, self).__init__()
        if hidden_units is None:
            hidden_units = [256, 256]
        self.dense_block = DenseBlock(hidden_units, activation)

    def call(self, inputs, training=None, mask=None):
        return inputs + self.dense_block(inputs)
