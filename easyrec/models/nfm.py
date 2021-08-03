import tensorflow as tf
from tensorflow.keras.activations import sigmoid

from easyrec import blocks


class NFM(tf.keras.models.Model):
    """
    Neural Factorization Machine.
    Xiangnan He et al. Neural Factorization Machines for Sparse Predictive Analytics. SIGIR. 2017.
    """
    def __init__(self,
                 one_hot_feature_columns,
                 k=32
                 ):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
        """
        super(NFM, self).__init__()
        if not one_hot_feature_columns:
            raise ValueError('len(one_hot_feature_columns) should greater than 0')
        self.nfm = blocks.NFM(one_hot_feature_columns, k=k)

    def call(self, inputs, training=None, mask=None):
        logits = self.nfm(inputs)
        return sigmoid(logits)
