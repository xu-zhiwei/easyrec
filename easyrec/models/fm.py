import tensorflow as tf
from tensorflow.keras.activations import sigmoid

from easyrec import blocks


class FM(tf.keras.models.Model):
    """
    Factorization Machine (FM).
    Reference: Steffen Rendle. Factorization Machines. ICDM. 2010.
    """

    def __init__(self,
                 one_hot_feature_columns,
                 k=16,
                 ):
        """

        Args:
            one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
            k: Dimension of the second-order weights.
        """
        super(FM, self).__init__()
        if not one_hot_feature_columns:
            raise ValueError('len(one_hot_feature_columns) should greater than 0')
        self.fm = blocks.FM(one_hot_feature_columns, k=k)

    def call(self, inputs, training=None, mask=None):
        logits = self.fm(inputs)
        return sigmoid(logits)
