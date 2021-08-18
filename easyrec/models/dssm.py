import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import DenseFeatures

from easyrec import blocks


class DSSM(tf.keras.models.Model):
    """
    Deep Structured Semantic Model (DSSM).
    Po-Sen Huang et al. Learning Deep Structured Semantic Models for Web Search using Clickthrough Data. CIKM. 2013.
    """

    def __init__(self,
                 user_feature_columns,
                 item_feature_columns,
                 user_units_list=None,
                 user_activation='relu',
                 item_units_list=None,
                 item_activation='relu',
                 score_function='inner_product'
                 ):
        """

        Args:
            user_feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which
                basically contains user feature fields.
            item_feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which
                basically contains item feature fields.
            user_units_list: Dimension of fully connected stack outputs in user dense block.
            user_activation: Activation to use in user dense block.
            item_units_list: Dimension of fully connected stack outputs in item dense block.
            item_activation: Activation to use in item dense block.
            score_function: Final output function to combine the user embedding and item embedding.
        """
        super(DSSM, self).__init__()
        if user_units_list is None:
            user_units_list = [256, 128, 64]
        if item_units_list is None:
            item_units_list = [256, 128, 64]
        self.user_input_layer = DenseFeatures(user_feature_columns)
        self.item_input_layer = DenseFeatures(item_feature_columns)
        self.user_tower = blocks.DenseBlock(units_list=user_units_list, activation=user_activation)
        self.item_tower = blocks.DenseBlock(units_list=item_units_list, activation=item_activation)
        self.score_function = score_function
        if self.score_function not in ('inner_product', 'cosine_similarity'):
            raise ValueError(f'{self.score_function} is not supported')

    def call(self, inputs, training=None, mask=None):
        user_inputs = self.user_input_layer(inputs)
        item_inputs = self.item_input_layer(inputs)
        user_embeddings = self.user_tower(user_inputs)
        item_embeddings = self.item_tower(item_inputs)
        if self.score_function == 'inner_product':
            return sigmoid(tf.reduce_sum(user_embeddings * item_embeddings, axis=1))
        elif self.score_function == 'cosine_similarity':
            return sigmoid(tf.reduce_sum(user_embeddings * item_embeddings, axis=1) / (
                    tf.sqrt(tf.reduce_sum(user_embeddings * user_embeddings, axis=1)) *
                    tf.sqrt(tf.reduce_sum(item_embeddings * item_embeddings, axis=1))
            ))
