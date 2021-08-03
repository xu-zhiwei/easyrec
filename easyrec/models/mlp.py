import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense

from easyrec.blocks import DenseBlock


class MLP(tf.keras.models.Model):
    """
    Multi-layer Perceptron.
    """

    def __init__(self,
                 feature_columns,
                 units_list=None,
                 activation='relu'
                 ):
        """

        Args:
            feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which basically
                contains all feature fields.
            units_list: Dimensionality of fully connected stack outputs.
            activation: Activation to use.
        """
        super(MLP, self).__init__()
        if units_list is None:
            units_list = [256, 128, 64]
        self.input_layer = DenseFeatures(feature_columns=feature_columns)
        self.dense_block = DenseBlock(units_list=units_list, activation=activation)
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        inputs = self.input_layer(inputs)
        x = self.dense_block(inputs)
        return self.score(x)
