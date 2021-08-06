import tensorflow as tf

from tensorflow.keras.layers import DenseFeatures, Flatten, Dense
from tensorflow.keras.activations import relu
from easyrec import blocks


class AutoInt(tf.keras.models.Model):
    """
    Automatic Feature Interaction (AutoInt).
    Reference: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks. CIKM. 2019.
    """
    def __init__(self,
                 one_hot_feature_columns,
                 multi_hot_feature_columns,
                 dense_feature_columns,
                 embedding_dimension=32,
                 num_heads=5,
                 attention_qkv_dimension=24,
                 attention_output_dimension=32
                 ):
        """

        Args:
            one_hot_feature_columns:
            multi_hot_feature_columns:
            dense_feature_columns:
            embedding_dimension:
            num_heads:
            attention_qkv_dimension:
            attention_output_dimension:
        """
        super(AutoInt, self).__init__()
        assert embedding_dimension == attention_output_dimension, \
            '`embedding_dimension` should be equal to `attention_output_dimension` to apply residual combination'

        self.hot_input_layer = [
            DenseFeatures(tf.feature_column.embedding_column(feature_column, embedding_dimension),
                          name=f'hot_input_layer_{feature_column.name}')
            for feature_column in one_hot_feature_columns + multi_hot_feature_columns
        ]
        if dense_feature_columns:
            self.dense_input_layer = DenseFeatures(dense_feature_columns, name='dense_input_layer')
            self.dense_embedding = tf.Variable(tf.random.normal(len(dense_feature_columns), embedding_dimension),
                                               name='dense_embedding')
        self.multi_head_self_attention = blocks.MultiHeadSelfAttention(
            embedding_dimension, attention_qkv_dimension, num_heads, attention_output_dimension, use_normalization=False
        )
        self.flatten = Flatten()
        self.score = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = tf.stack([input_layer(inputs) for input_layer in self.hot_input_layer], axis=1)
        if hasattr(self, 'dense_input_layer') and hasattr(self, 'dense_embedding'):
            x = tf.stack([
                x,
                tf.expand_dims(self.dense_input_layer(inputs), -1) * self.dense_embedding
            ], axis=1)
        x = relu(x + self.multi_head_self_attention(x))
        x = self.flatten(x)
        return self.score(x)

