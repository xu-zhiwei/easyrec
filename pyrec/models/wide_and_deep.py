import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Flatten


class WideAndDeep(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 deep_hidden_units=None,
                 deep_activations='relu'
                 ):
        super(WideAndDeep, self).__init__()
        self.hot_embeddings = [
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in one_hot_feature_columns],
            *[DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=embedding_dimension))
              for feature_column in multi_hot_feature_columns],
        ]
        if dense_feature_columns:
            self.dense_embedding = DenseFeatures(dense_feature_columns)
        self.flatten = Flatten()


    def call(self, inputs, training=None, mask=None):
        pass
