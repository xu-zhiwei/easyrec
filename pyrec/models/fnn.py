import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense

from pyrec.models import FM


class FNN(tf.keras.Model):
    def __init__(self, fm: FM, hidden_units=None):
        """
        Factorization Machine supported Neural Network (FNN).

        :param fm: pretrained Factorization Machines.
        :param hidden_units:
        """
        super(FNN, self).__init__()
        if hidden_units is None:
            hidden_units = [256, 128, 1]
        if hidden_units[-1] != -1:
            raise ValueError('last element of hidden_units should be 1')
        self.fm = fm
        self.fcs = [Dense(units) for units in hidden_units]

    def call(self, inputs, training=None, mask=None):
        ws = [self.fm.fm.w[i](inputs) for i in range(self.fm.fm.num_fields)]
        ws = tf.transpose(tf.convert_to_tensor(ws), [1, 0, 2])
        ws = tf.squeeze(ws)
        vs = [self.fm.fm.v[i](inputs) for i in range(self.fm.fm.num_fields)]
        vs = tf.transpose(tf.convert_to_tensor(vs), [1, 0, 2])
        vs = tf.concat(vs, axis=1)
        x = tf.concat((ws, vs), axis=1)
        for fc in self.fcs:
            x = fc(x)
        return sigmoid(x)
