import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import DenseFeatures

from easyrec import blocks


class DSSM(tf.keras.models.Model):
    def __init__(self,
                 user_feature_columns,
                 item_feature_columns,
                 user_dimension=256,
                 item_dimension=256,
                 user_hidden_units=None,
                 user_activation='relu',
                 item_hidden_units=None,
                 item_activation='relu',
                 score_function='inner_product'
                 ):
        super(DSSM, self).__init__()
        if user_hidden_units is None:
            user_hidden_units = [256, 128, 64]
        if item_hidden_units is None:
            item_hidden_units = [256, 128, 64]
        self.user_input_layer = DenseFeatures(user_feature_columns)
        self.item_input_layer = DenseFeatures(item_feature_columns)
        self.user_tower = blocks.DenseBlock(hidden_units=user_hidden_units, activation=user_activation)
        self.item_tower = blocks.DenseBlock(hidden_units=item_hidden_units, activation=item_activation)
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
