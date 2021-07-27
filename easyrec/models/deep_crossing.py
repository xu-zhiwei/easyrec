import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense, Flatten

from easyrec.blocks import ResidualBlock


class DeepCrossing(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 embedding_dimension=32,
                 num_residual_blocks=5,
                 residual_hidden_units=None,
                 residual_activation='relu'
                 ):
        super(DeepCrossing, self).__init__()
        if residual_hidden_units is None:
            residual_hidden_units = [256, 256]

        self.hot_embeddings = [
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in one_hot_feature_columns],
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in multi_hot_feature_columns],
        ]
        if dense_feature_columns:
            self.dense_embedding = DenseFeatures(dense_feature_columns)
        self.flatten = Flatten()
        self.embedding = Dense(units=residual_hidden_units[0], activation='relu')
        self.residual_blocks = [ResidualBlock(residual_hidden_units, residual_activation)
                                for _ in range(num_residual_blocks)]
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = [embedding(inputs) for embedding in self.hot_embeddings]
        x = self.flatten(tf.transpose(tf.convert_to_tensor(x), [1, 0, 2]))
        if hasattr(self, 'dense_embedding'):
            x = tf.concat((x, self.dense_embedding(inputs)), axis=1)
        x = self.embedding(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return self.score(x)
