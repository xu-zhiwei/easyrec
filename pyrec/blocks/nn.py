import tensorflow as tf
from tensorflow.keras.layers import Dense


class ResidualBlock(tf.keras.models.Model):
    def __init__(self, hidden_units=None, activation=None):
        super(ResidualBlock, self).__init__()
        if hidden_units is None:
            hidden_units = [256, 256]
        self.fcs = [Dense(units, activation=activation) for units in hidden_units]

    def call(self, inputs, training=None, mask=None):
        net = inputs
        for fc in self.fcs:
            net = fc(net)
        net += inputs
        return net
