import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Flatten, Dense

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
        self.hot_embeddings = [
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in one_hot_feature_columns],
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in multi_hot_feature_columns],
        ]
        if dense_feature_columns:
            self.dense_embedding = DenseFeatures(dense_feature_columns)
        self.flatten = Flatten()
        self.dense_block = blocks.DenseBlock(hidden_units=deep_hidden_units, activation=deep_activation)
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = [embedding(inputs) for embedding in self.hot_embeddings]
        x = self.flatten(tf.transpose(tf.convert_to_tensor(x), [1, 0, 2]))
        x = self.dense_block(x)
        if hasattr(self, 'dense_embedding'):
            x = tf.concat((x, self.dense_embedding(inputs)), axis=1)
        return self.score(x)
