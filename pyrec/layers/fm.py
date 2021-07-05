import tensorflow as tf
from pyrec.utils import get_feature_column_shape


class FM(tf.keras.layers.Layer):
    """
    Factorization machine layer using vector w and matrix v.
    """

    def __init__(self, one_hot_feature_columns, multi_hot_feature_columns, k=16):
        super(FM, self).__init__()
        one_hot_shape = sum([get_feature_column_shape(feature_column) for feature_column in one_hot_feature_columns])
        multi_hot_shape = sum(
            [get_feature_column_shape(feature_column) for feature_column in multi_hot_feature_columns])

        self.k = k
        self.w = tf.Variable(tf.random.normal(shape=(one_hot_shape + multi_hot_shape, 1)), name='w')
        self.v = tf.Variable(tf.random.normal(shape=(one_hot_shape + multi_hot_shape, self.k)), name='v')

    def call(self, inputs, *args, **kwargs):
        """

        Args:
            inputs: [batch_size, dimension]
        Returns:
            logits
        """
        logits = tf.squeeze(tf.matmul(inputs, self.w))
        square_of_sum = tf.square(tf.matmul(inputs, self.v))
        sum_of_square = tf.matmul(tf.square(inputs), tf.square(self.v))
        logits += 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1)
        return logits


class FFM(tf.keras.layers.Layer):
    """
    Field-aware factorization machine layer.
    """

    def __init__(self, one_hot_feature_columns, multi_hot_feature_columns, k=16):
        super(FFM, self).__init__()
        one_hot_shape = sum([get_feature_column_shape(feature_column) for feature_column in one_hot_feature_columns])
        multi_hot_shape = sum(
            [get_feature_column_shape(feature_column) for feature_column in multi_hot_feature_columns])
        self.field_dims = [*[get_feature_column_shape(feature_column) for feature_column in one_hot_feature_columns],
                           *[get_feature_column_shape(feature_column) for feature_column in multi_hot_feature_columns]]
        self.num_fields = len(self.field_dims)

        self.k = k
        self.w = tf.Variable(tf.random.normal(shape=(one_hot_shape + multi_hot_shape, 1)),
                             name='w')
        self.v = tf.Variable(tf.random.normal(shape=(self.num_fields, one_hot_shape + multi_hot_shape, self.k)),
                             name='v')

    def call(self, inputs, *args, **kwargs):
        logits = tf.matmul(inputs, self.w)

        for i in range(self.num_fields - 1):
            for j in range(1, self.num_fields):
                ix, jx = [], []
                #
                # self.v[j]

                # x = x + x.new_tensor(self.offsets).unsqueeze(0)
                # xs = [self.embeddings[i](x) for i in range(self.num_fields)]
                # ix = list()
                # for i in range(self.num_fields - 1):
                #     for j in range(i + 1, self.num_fields):
                #         ix.append(xs[j][:, i] * xs[i][:, j])
                # ix = torch.stack(ix, dim=1)
                # return ix


