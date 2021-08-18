import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense, Flatten

from easyrec.blocks import ResidualBlock


class DeepCrossing(tf.keras.models.Model):
    """
    Deep Crossing.
    Reference: Ying Shan et al. Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial
    Features. KDD. 2016.
    """

    def __init__(self,
                 feature_columns,
                 num_residual_blocks=5,
                 residual_units_list=None,
                 residual_activation='relu'
                 ):
        """

        Args:
            feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which basically
                contains all feature fields.
            num_residual_blocks: Number of residual blocks.
            residual_units_list: Dimension of fully connected stack outputs in residual block.
            residual_activation: Activation to use in residual block.
        """
        super(DeepCrossing, self).__init__()
        if residual_units_list is None:
            residual_units_list = [256, 256]

        self.input_layer = DenseFeatures(feature_columns)
        self.flatten = Flatten()
        self.embedding = Dense(units=residual_units_list[0], activation='relu')  # align dimension of residual block
        self.residual_blocks = [ResidualBlock(residual_units_list, residual_activation)
                                for _ in range(num_residual_blocks)]
        self.score = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        if hasattr(self, 'dense_embedding'):
            x = tf.concat((x, self.dense_embedding(inputs)), axis=1)
        x = self.embedding(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return self.score(x)
