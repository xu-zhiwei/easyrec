import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense

from easyrec import blocks


class xDeepFM(tf.keras.models.Model):
    """
    Extreme Deep Factorization Machine (xDeepFM).
    Reference: Jianxun Lian et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender
        Systems. KDD. 2018.
    """

    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 k=32,
                 deep_units_list=None,
                 deep_activation='relu',
                 cross_units_list=None
                 ):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            multi_hot_feature_columns: List[CategoricalColumn] encodes multi hot feature fields, such as
                historical_item_ids.
            k: Dimension of the second-order weights.
            deep_units_list: Dimensionality of fully connected stack outputs in deep block.
            deep_activation: Activation to use in deep block.
            cross_units_list: Number of fields in the cross layer.
        """
        super(xDeepFM, self).__init__()
        if deep_units_list is None:
            deep_units_list = [256, 128]
        if cross_units_list is None:
            cross_units_list = [10, 6]

        self.wide = [
            DenseFeatures(tf.feature_column.indicator_column(feature_column))
            for feature_column in one_hot_feature_columns + multi_hot_feature_columns
        ]
        self.embeddings = [
            DenseFeatures(tf.feature_column.embedding_column(feature_column, dimension=k))
            for feature_column in one_hot_feature_columns + multi_hot_feature_columns
        ]
        self.num_fields = len(one_hot_feature_columns) + len(multi_hot_feature_columns)
        self.deep = blocks.DenseBlock(deep_units_list, deep_activation)
        cross_units_list = [self.num_fields, *cross_units_list]
        self.cross = [tf.Variable(tf.random.normal((self.num_fields, cross_units_list[i], cross_units_list[i + 1])))
                      for i in range(len(cross_units_list) - 1)]  # [num_fields, Hk, Hk+1]
        self.score = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        embeddings = [embedding(inputs) for embedding in self.embeddings]

        wide = tf.concat([wide(inputs) for wide in self.wide], axis=1)

        deep = self.deep(tf.concat(embeddings, axis=1))

        cross = []
        x0 = tf.stack(embeddings, axis=1)
        x = x0
        for w in self.cross:
            x = tf.stack(
                [
                    tf.matmul(
                        tf.expand_dims(x0[:, :, i], 2), tf.expand_dims(x[:, :, i], 1)
                    )  # [batch, num_fields, Hk]
                    for i in range(x0.shape[2])
                ], axis=1
            )  # [batch, k, num_fields, Hk]
            x = tf.expand_dims(x, -1) * w  # [batch, k, num_fields, Hk, Hk+1]
            x = tf.reduce_sum(x, axis=(2, 3))  # [batch, k, Hk+1]
            x = tf.transpose(x, [0, 2, 1])  # [batch, Hk+1, k]
            cross.append(tf.reduce_sum(x, axis=2))  # [batch, Hk+1]
        cross = tf.concat(cross, axis=1)

        return self.score(tf.concat((wide, deep, cross), axis=1))
