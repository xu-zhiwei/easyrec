import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense, Flatten

from easyrec import blocks


class PNN(tf.keras.models.Model):
    """
    Product-based Neural Network (PNN).
    Reference: Yanru Qu et al. Product-based Neural Networks for User Response Prediction. ICDM. 2016.
    """
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 embedding_dimension=32,
                 use_inner_product=False,
                 use_outer_product=False,
                 units_list=None,
                 activation='relu'
                 ):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            multi_hot_feature_columns: List[CategoricalColumn] encodes multi hot feature fields, such as
                historical_item_ids.
            embedding_dimension: embedding dimension of each field.
            use_inner_product: whether to use IPNN.
            use_outer_product: whether to use OPNN.
            units_list: Dimensionality of fully connected stack outputs.
            activation: Activation to use.
        """
        super(PNN, self).__init__()
        if units_list is None:
            units_list = [128, 64]
        self.embeddings = [
            DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
            for feature_column in one_hot_feature_columns + multi_hot_feature_columns
        ]
        self.use_inner_product = use_inner_product
        self.use_outer_product = use_outer_product
        self.flatten = Flatten()
        self.dense_block = blocks.DenseBlock(units_list, activation)
        self.score = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        inputs = [embedding(inputs) for embedding in self.embeddings]
        inputs = tf.stack(inputs, axis=1)

        z = self.flatten(inputs)

        if self.use_inner_product:
            inner_p = tf.matmul(inputs, tf.transpose(inputs, [0, 2, 1]))
            inner_p = self.flatten(inner_p)
            z = tf.concat((z, inner_p), axis=1)

        if self.use_outer_product:
            outer_p = tf.expand_dims(tf.reduce_sum(inputs, axis=1), -1)
            outer_p = tf.matmul(outer_p, tf.transpose(outer_p, [0, 2, 1]))
            outer_p = self.flatten(outer_p)
            z = tf.concat((z, outer_p), axis=1)

        z = self.dense_block(z)
        z = self.score(z)
        return z
