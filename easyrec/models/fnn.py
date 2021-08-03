import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Flatten

from easyrec import blocks


class FNN(tf.keras.Model):
    """
    Factorization-machine supported Neural Network.
    Reference: Weinan Zhang. Deep Learning over Multi-field Categorical Data – A Case Study on User Response
        Prediction. ECIR. 2016.
    """

    def __init__(self,
                 one_hot_feature_columns,
                 k=16,
                 units_list=None,
                 activation='tanh'
                 ):
        """
        fm: Pretrained Factorization Machines.
        one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
        units_list: Dimensionality of fully connected stack outputs.
        activation: Activation to use.
        """
        super(FNN, self).__init__()
        if units_list is None:
            units_list = [256, 128]

        self.fm = blocks.FM(one_hot_feature_columns, k=k)
        self.num_fields = len(one_hot_feature_columns)

        self.dense_block = blocks.DenseBlock(units_list, activation)
        self.score = Dense(1, activation='sigmoid')
        self.flatten = Flatten()

    def call(self, inputs, pretraining=True, training=None, mask=None):
        if pretraining:
            logits = self.fm(inputs)
            return sigmoid(logits)
        else:
            self._freeze_fm()
            ws = tf.concat([self.fm.w[i](inputs) for i in range(self.num_fields)], axis=1)
            vs = tf.concat([self.fm.v[i](inputs) for i in range(self.num_fields)], axis=1)
            x = tf.concat((ws, vs), axis=1)
            x = self.dense_block(x)
            x = self.score(x)
            return x

    def _freeze_fm(self):
        self.fm.trainable = False
        for layer in self.fm.layers:
            layer.trainable = False
