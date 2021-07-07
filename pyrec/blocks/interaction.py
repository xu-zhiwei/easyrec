import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures

from pyrec.utils import get_feature_column_shape


class FM(tf.keras.Model):
    """
    Factorization machine layer using vector w and matrix v.
    """

    def __init__(self, one_hot_feature_columns, k=16):
        super(FM, self).__init__()
        self.num_one_hot_feature_columns = len(one_hot_feature_columns)
        self.w = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=1))
                  for feature_column in one_hot_feature_columns]
        self.v = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=k))
                  for feature_column in one_hot_feature_columns]

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs: [batch_size, dimension]
        Returns:
            logits
        """
        ws = [self.w[i](inputs) for i in range(self.num_one_hot_feature_columns)]
        ws = tf.transpose(tf.convert_to_tensor(ws), [1, 0, 2])  # [batch_size, num_fields, embedding_size=1]
        logits = tf.reduce_sum(tf.squeeze(ws), axis=1)

        vs = [self.v[i](inputs) for i in range(self.num_one_hot_feature_columns)]
        vs = tf.transpose(tf.convert_to_tensor(vs), [1, 0, 2])  # [batch_size, num_fields, embedding_size=k]
        square_of_sum = tf.square(tf.reduce_sum(vs, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(vs), axis=1)
        logits += 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1)
        return logits


class FFM(tf.keras.Model):
    """
    Field-aware factorization machine layer.
    """

    def __init__(self, one_hot_feature_columns, k=4):
        super(FFM, self).__init__()
        one_hot_shape = sum([get_feature_column_shape(feature_column) for feature_column in one_hot_feature_columns])
        self.field_dims = [get_feature_column_shape(feature_column) for feature_column in one_hot_feature_columns]
        self.num_fields = len(self.field_dims)

        self.k = k
        self.w = tf.Variable(tf.random.normal(shape=(one_hot_shape, 1)), name='w')
        self.v = tf.Variable(tf.random.normal(shape=(self.num_fields, one_hot_shape, self.k)), name='v')

    def call(self, inputs, *args, **kwargs):
        logits = tf.matmul(inputs, self.w)
        for i in range(self.num_fields - 1):
            for j in range(1, self.num_fields):
                logits += self.v[j, i, :] * self.v[i, j, :]  # still have bugs
        return logits
