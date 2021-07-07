import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import DenseFeatures, Dense


class LR(tf.keras.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 ):
        super(LR, self).__init__()
        self.input_layer = DenseFeatures(
            feature_columns=one_hot_feature_columns + multi_hot_feature_columns + dense_feature_columns)
        self.fc = Dense(units=1)

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        logits = self.fc(x)
        return sigmoid(logits)
