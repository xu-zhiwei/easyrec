import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures


class FM(tf.keras.models.Model):
    """
    Factorization machine layer using vector w and matrix v.
    """

    def __init__(self, one_hot_feature_columns, k=16):
        super(FM, self).__init__()
        self.num_fields = len(one_hot_feature_columns)
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
        ws = [self.w[i](inputs) for i in range(self.num_fields)]
        ws = tf.transpose(tf.convert_to_tensor(ws), [1, 0, 2])  # [batch_size, num_fields, embedding_size=1]
        logits = tf.reduce_sum(tf.squeeze(ws), axis=1)

        vs = [self.v[i](inputs) for i in range(self.num_fields)]
        vs = tf.transpose(tf.convert_to_tensor(vs), [1, 0, 2])  # [batch_size, num_fields, embedding_size=k]
        square_of_sum = tf.square(tf.reduce_sum(vs, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(vs), axis=1)
        logits += 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1)
        return logits


class FFM(tf.keras.models.Model):
    """
    Field-aware factorization machine layer.
    """

    def __init__(self, one_hot_feature_columns, k=4):
        super(FFM, self).__init__()
        self.num_fields = len(one_hot_feature_columns)
        self.w = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=1))
                  for feature_column in one_hot_feature_columns]
        self.vv = [
            [
                DenseFeatures(tf.feature_column.embedding_column(one_hot_feature_columns[j], dimension=k))
                if i != j else None
                for j in range(self.num_fields)
            ]
            for i in range(self.num_fields)
        ]

    def call(self, inputs, *args, **kwargs):
        ws = [self.w[i](inputs) for i in range(self.num_fields)]
        ws = tf.transpose(tf.convert_to_tensor(ws), [1, 0, 2])  # [batch_size, num_fields, embedding_size=1]
        logits = tf.reduce_sum(tf.squeeze(ws), axis=1)

        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                logits += tf.reduce_sum(self.vv[i][j](inputs) * self.vv[j][i](inputs), axis=1)
        return logits
