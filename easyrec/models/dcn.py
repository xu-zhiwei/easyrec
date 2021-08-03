import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense

from easyrec import blocks


class DCN(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 embedding_dimension=32,
                 num_crosses=5,
                 deep_units_list=None,
                 deep_activation='relu'):
        super(DCN, self).__init__()
        if deep_units_list is None:
            deep_units_list = [256, 128, 64]
        self.input_layer = DenseFeatures(
            [tf.feature_column.embedding_column(feature_column, embedding_dimension)
             for feature_column in one_hot_feature_columns] +
            [tf.feature_column.embedding_column(feature_column, embedding_dimension)
             for feature_column in multi_hot_feature_columns] +
            dense_feature_columns
        )
        input_dimension = (len(one_hot_feature_columns) + len(multi_hot_feature_columns)) * embedding_dimension + len(
            dense_feature_columns)
        self.deep = blocks.DenseBlock(deep_units_list, deep_activation)
        self.cross_w = [tf.Variable(tf.random.normal((input_dimension, 1))) for _ in range(num_crosses)]
        self.cross_b = [tf.Variable(tf.random.normal((input_dimension,))) for _ in range(num_crosses)]
        self.score = Dense(1, 'sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)

        deep = self.deep(x)

        cross = x
        for w, b in zip(self.cross_w, self.cross_b):
            x1 = tf.expand_dims(x, -1)
            x2 = tf.expand_dims(cross, 1)
            cross = tf.squeeze(
                tf.matmul(
                    tf.matmul(x1, x2), w
                )
            ) + b + cross

        x = tf.concat((cross, deep), axis=1)
        return self.score(x)
