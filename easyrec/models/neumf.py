import tensorflow as tf

from tensorflow.keras.layers import DenseFeatures, Dense
from easyrec import blocks


class NeuMF(tf.keras.models.Model):
    """
    Neural Matrix Factorization.
    Xiangnan He et al. Neural Factorization Machines for Sparse Predictive Analytics. SIGIR. 2017.
    """
    def __init__(self,
                 user_feature_column,
                 item_feature_column,
                 user_embedding_dimension=64,
                 item_embedding_dimension=64,
                 units_list=None,
                 activation='relu'):
        """

        Args:
            user_feature_column: CategoricalColumn to represent user_id.
            item_feature_column: CategoricalColumn to represent item_id.
            user_embedding_dimension: Dimension of user embedding.
            item_embedding_dimension: Dimension of item embedding.
            units_list: Dimensionality of fully connected stack outputs.
            activation: Activation to use.
        """
        super(NeuMF, self).__init__()
        self.user_input_layer1 = DenseFeatures(
            tf.feature_column.embedding_column(user_feature_column, dimension=user_embedding_dimension)
        )
        self.user_input_layer2 = DenseFeatures(
            tf.feature_column.embedding_column(user_feature_column, dimension=user_embedding_dimension)
        )
        self.item_input_layer1 = DenseFeatures(
            tf.feature_column.embedding_column(item_feature_column, dimension=item_embedding_dimension)
        )
        self.item_input_layer2 = DenseFeatures(
            tf.feature_column.embedding_column(item_feature_column, dimension=item_embedding_dimension)
        )
        self.dense_block = blocks.DenseBlock(units_list, activation)
        self.score = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        gmf = self.user_input_layer1(inputs) * self.item_input_layer1(inputs)
        mlp = self.dense_block(
            tf.concat((self.user_input_layer2(inputs), self.item_input_layer2(inputs)), axis=1)
        )
        x = tf.concat((gmf, mlp), axis=1)
        return self.score(x)


