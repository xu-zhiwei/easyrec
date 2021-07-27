import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Flatten

from easyrec import blocks


class DeepFM(tf.keras.models.Model):
    def __init__(self,
                 one_hot_feature_columns,
                 embedding_dimension=32,
                 deep_hidden_units=None,
                 deep_activation='relu'
                 ):
        super(DeepFM, self).__init__()
        self.flatten = Flatten()
        self.fm = blocks.FM(one_hot_feature_columns=one_hot_feature_columns, k=embedding_dimension)
        self.dense_block = blocks.DenseBlock(deep_hidden_units, deep_activation)
        self.score = Dense(1)

    def call(self, inputs, training=None, mask=None):
        wide = self.fm(inputs)
        wide = tf.expand_dims(wide, -1)

        embeddings = [self.fm.v[i](inputs) for i in range(self.fm.num_fields)]
        embeddings = tf.transpose(tf.convert_to_tensor(embeddings), [1, 0, 2])
        embedding = self.flatten(embeddings)
        deep = self.dense_block(embedding)
        deep = self.score(deep)

        return sigmoid(wide + deep)
