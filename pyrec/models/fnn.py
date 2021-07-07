import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense

from pyrec import blocks


class FNN(tf.keras.Model):
    def __init__(self, one_hot_feature_columns, hidden_units=None):
        super(FNN, self).__init__()
        if hidden_units is None:
            hidden_units = [256, 128]
        self.fm = blocks.FM(one_hot_feature_columns)
        self.fcs = [Dense(units) for units in hidden_units]

    def call(self, inputs, pretraining=True, training=None, mask=None):
        if pretraining:
            return sigmoid(self.fm(inputs))
        else:
            pass

