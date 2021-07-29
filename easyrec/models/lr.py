import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense


class LR(tf.keras.Model):
    """
    Logisitic Regression.
    """

    def __init__(self,
                 feature_columns
                 ):
        """

        Args:
            feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which basically
                contains all feature fields.
        """
        super(LR, self).__init__()
        self.input_layer = DenseFeatures(feature_columns=feature_columns)
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        return self.score(x)
