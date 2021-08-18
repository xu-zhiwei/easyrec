import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense

from easyrec import blocks


class FM(tf.keras.models.Model):
    """
    Factorization machine layer using vector w and matrix v.
    """

    def __init__(self, one_hot_feature_columns, k=16):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
        """
        super(FM, self).__init__()
        self.num_fields = len(one_hot_feature_columns)
        self.b = tf.Variable(tf.random.normal(shape=(1,)), name='b')
        self.w = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=1),
                                name=f'w_{feature_column.name}')
                  for feature_column in one_hot_feature_columns]
        self.v = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=k),
                                name=f'v_{feature_column.name}')
                  for feature_column in one_hot_feature_columns]

    def call(self, inputs, *args, **kwargs):
        ws = tf.stack([self.w[i](inputs) for i in range(self.num_fields)], axis=1)
        logits = tf.reduce_sum(tf.squeeze(ws), axis=1)

        vs = tf.stack([self.v[i](inputs) for i in range(self.num_fields)], axis=1)
        square_of_sum = tf.square(tf.reduce_sum(vs, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(vs), axis=1)
        logits += 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1)

        logits += self.b
        return logits


class FFM(tf.keras.models.Model):
    """
    Field-aware factorization machine layer.
    """

    def __init__(self, one_hot_feature_columns, k=4):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
        """
        super(FFM, self).__init__()
        self.num_fields = len(one_hot_feature_columns)
        self.b = tf.Variable(tf.random.normal(shape=(1,)), name='b')
        self.w = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=1),
                                name=f'w_{feature_column.name}')
                  for feature_column in one_hot_feature_columns]
        self.vv = [
            [
                DenseFeatures(tf.feature_column.embedding_column(one_hot_feature_columns[j], dimension=k),
                              name=f'vv_{one_hot_feature_columns[i].name}_{one_hot_feature_columns[j].name}')
                if i != j else None
                for j in range(self.num_fields)
            ]
            for i in range(self.num_fields)
        ]

    def call(self, inputs, *args, **kwargs):
        ws = tf.stack([self.w[i](inputs) for i in range(self.num_fields)], axis=1)
        logits = tf.reduce_sum(tf.squeeze(ws), axis=1)

        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                logits += tf.reduce_sum(self.vv[i][j](inputs) * self.vv[j][i](inputs), axis=1)

        logits += self.b
        return logits


class AFM(tf.keras.models.Model):
    """
    Attentional factorization machine layer.
    """

    def __init__(self, one_hot_feature_columns, k=16):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
        """
        super(AFM, self).__init__()
        self.num_fields = len(one_hot_feature_columns)
        self.b = tf.Variable(tf.random.normal(shape=(1,)), name='b')
        self.w = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=1),
                                name=f'w_{feature_column.name}')
                  for feature_column in one_hot_feature_columns]
        self.v = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=k),
                                name=f'v_{feature_column.name}')
                  for feature_column in one_hot_feature_columns]

        self.att = Dense(units=k, activation='relu')
        self.h = tf.Variable(tf.random.normal(shape=(k, 1)))
        self.p = Dense(units=1, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        ws = tf.stack([self.w[i](inputs) for i in range(self.num_fields)], axis=1)
        logits = tf.reduce_sum(tf.squeeze(ws), axis=1)

        vs = [self.v[i](inputs) for i in range(self.num_fields)]
        vvs = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                vvs.append(vs[i] * vs[j])
        weights = [tf.squeeze(tf.matmul(self.att(vv), self.h)) for vv in vvs]
        weights = tf.transpose(tf.convert_to_tensor(weights))
        weights = tf.nn.softmax(weights)
        for i in range(len(vvs)):
            logits += tf.squeeze(self.p(tf.expand_dims(weights[:, i], -1) * vvs[i]))

        logits += self.b
        return logits


class NFM(tf.keras.models.Model):
    """
    Neural factorization machine layer.
    """
    def __init__(self, one_hot_feature_columns, k=32, units_list=None, activation='relu'):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
            units_list: Dimension of fully connected stack outputs.
            activation: Activation to use.
        """
        super(NFM, self).__init__()
        if units_list is None:
            units_list = [64, 32, 16]
        self.num_fields = len(one_hot_feature_columns)
        self.b = tf.Variable(tf.random.normal(shape=(1,)), name='b')
        self.w = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=1),
                                name=f'w_{feature_column.name}')
                  for feature_column in one_hot_feature_columns]
        self.v = [DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=k),
                                name=f'v_{feature_column.name}')
                  for feature_column in one_hot_feature_columns]
        self.dense_block = blocks.DenseBlock(units_list, activation)
        self.score = Dense(units=1, activation=activation)

    def call(self, inputs, training=None, mask=None):
        ws = tf.stack([self.w[i](inputs) for i in range(self.num_fields)], axis=1)
        logits = tf.reduce_sum(tf.squeeze(ws), axis=1)

        vs = tf.stack([self.v[i](inputs) for i in range(self.num_fields)], axis=1)
        square_of_sum = tf.square(tf.reduce_sum(vs, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(vs), axis=1)
        x = 0.5 * (square_of_sum - sum_of_square)
        x = self.dense_block(x)
        logits += tf.squeeze(self.score(x))

        logits += self.b
        return logits
