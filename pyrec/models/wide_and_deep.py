import tensorflow as tf


class WideAndDeep(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 deep_hidden_units=None,
                 deep_activations='relu'
                 ):
        super(WideAndDeep, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass


