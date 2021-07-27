import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense


class LR(tf.keras.Model):
    """
    Logisitic Regression.
    """

    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 ):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            multi_hot_feature_columns: List[CategoricalColumn] encodes multi hot feature fields, such as
                historical_item_ids.
            dense_feature_columns: List[NumericalColumn] encodes numerical feature fields, such as age.
        """
        super(LR, self).__init__()
        self.input_layer = DenseFeatures(
            feature_columns=one_hot_feature_columns + multi_hot_feature_columns + dense_feature_columns)
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        return self.score(x)
