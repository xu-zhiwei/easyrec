import tensorflow as tf
from tensorflow.keras.activations import sigmoid

from easyrec import blocks


class AFM(tf.keras.models.Model):
    """
    Attentional Factorization Machines (AFM).
    Reference: Jun Xiao et al. Attentional Factorization Machines:Learning the Weight of Feature Interactions
    via Attention Networks. arXiv. 2017.
    """

    def __init__(self,
                 one_hot_feature_columns,
                 k=16):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
        """
        super(AFM, self).__init__()
        if not one_hot_feature_columns:
            raise ValueError('len(one_hot_feature_columns) should greater than 0')
        self.afm = blocks.AFM(one_hot_feature_columns, k=k)

    def call(self, inputs, training=None, mask=None):
        logits = self.afm(inputs)
        return sigmoid(logits)
