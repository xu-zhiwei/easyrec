import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Flatten

from easyrec import blocks


class DeepFM(tf.keras.models.Model):
    """
    DeepFM.
    Reference: Huifeng Guo et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. arXiv. 2017.
    """
    def __init__(self,
                 one_hot_feature_columns,
                 k=32,
                 deep_units_list=None,
                 deep_activation='relu'
                 ):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
            deep_units_list: Dimensionality of fully connected stack outputs in deep block.
            deep_activation: Activation to use in deep block.
        """
        super(DeepFM, self).__init__()
        if deep_units_list is None:
            deep_units_list = [256, 128]
        self.flatten = Flatten()
        self.fm = blocks.FM(one_hot_feature_columns=one_hot_feature_columns, k=k)
        self.dense_block = blocks.DenseBlock(deep_units_list, deep_activation)
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
