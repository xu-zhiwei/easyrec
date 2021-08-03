import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense

from easyrec import blocks


class WideAndDeep(tf.keras.models.Model):
    """
    Wide & Deep proposed by Google.

    Note:
        the dense_feature_columns should contain original dense feature columns
        and hand-crafted cross transformation feature columns.
    """

    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 embedding_dimension=64,
                 deep_hidden_units=None,
                 deep_activation='relu'
                 ):
        super(WideAndDeep, self).__init__()
        if deep_hidden_units is None:
            deep_hidden_units = [1024, 512, 256]
        wide_feature_columns = [
            *[tf.feature_column.indicator_column(feature_column) for feature_column in one_hot_feature_columns],
            *[tf.feature_column.indicator_column(feature_column) for feature_column in multi_hot_feature_columns],
        ]
        deep_feature_columns = [
            *[tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension)
              for feature_column in one_hot_feature_columns],
            *[tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension)
              for feature_column in multi_hot_feature_columns],
        ]
        if dense_feature_columns:
            deep_feature_columns += dense_feature_columns

        self.wide_input_layer = DenseFeatures(wide_feature_columns)
        self.deep_input_layer = DenseFeatures(deep_feature_columns)
        self.wide_dense_block = blocks.DenseBlock(units_list=[1], activation=None)
        self.deep_dense_block = blocks.DenseBlock(units_list=deep_hidden_units, activation=deep_activation)
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        wide = self.wide_input_layer(inputs)
        wide = self.wide_dense_block(wide)
        deep = self.deep_input_layer(inputs)
        deep = self.deep_dense_block(deep)
        return self.score(wide + deep)
