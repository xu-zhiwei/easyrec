import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense

from easyrec import blocks


class WideAndDeep(tf.keras.models.Model):
    """
    Wide & Deep.
    Reference: Heng-Tze Cheng et al. Wide & Deep Learning for Recommender Systems. RecSys. 2016.
    """

    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns=None,
                 dense_feature_columns=None,
                 embedding_dimension=64,
                 deep_units_list=None,
                 deep_activation='relu'
                 ):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            multi_hot_feature_columns: List[CategoricalColumn] encodes multi hot feature fields, such as
                historical_item_ids.
            dense_feature_columns: List[NumericalColumn] encodes numerical feature fields, such as age.
            embedding_dimension: Dimension of embedded CategoricalColumn.
            deep_units_list: Dimensionality of fully connected stack outputs in deep dense block.
            deep_activation: Activation to use in deep dense block.
        """
        super(WideAndDeep, self).__init__()
        if deep_units_list is None:
            deep_units_list = [1024, 512, 256]
        wide_feature_columns = [tf.feature_column.indicator_column(feature_column)
                                for feature_column in one_hot_feature_columns + multi_hot_feature_columns]
        deep_feature_columns = [tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension)
                                for feature_column in one_hot_feature_columns + multi_hot_feature_columns]
        if dense_feature_columns:
            wide_feature_columns += dense_feature_columns
            deep_feature_columns += dense_feature_columns

        self.wide_input_layer = DenseFeatures(wide_feature_columns)
        self.deep_input_layer = DenseFeatures(deep_feature_columns)
        self.wide_dense_block = blocks.DenseBlock(units_list=[1], activation=None)
        self.deep_dense_block = blocks.DenseBlock(units_list=deep_units_list, activation=deep_activation)
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        wide = self.wide_input_layer(inputs)
        wide = self.wide_dense_block(wide)
        deep = self.deep_input_layer(inputs)
        deep = self.deep_dense_block(deep)
        return self.score(wide + deep)
