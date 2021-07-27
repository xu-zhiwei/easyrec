import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense, Flatten

from easyrec import blocks


class PNN(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 embedding_dimension=32,
                 use_inner_product=False,
                 use_outer_product=False,
                 hidden_units=None,
                 activation='relu'
                 ):
        super(PNN, self).__init__()
        if hidden_units is None:
            hidden_units = [128, 64]
        self.embeddings = [
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in one_hot_feature_columns],
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in multi_hot_feature_columns]
        ]
        self.use_inner_product = use_inner_product
        self.use_outer_product = use_outer_product
        self.flatten = Flatten()
        self.dense_block = blocks.DenseBlock(hidden_units, activation)
        self.score = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        inputs = [embedding(inputs) for embedding in self.embeddings]
        inputs = tf.transpose(tf.convert_to_tensor(inputs), [1, 0, 2])

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
