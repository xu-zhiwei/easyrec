import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Dense

from easyrec import blocks


class MMOE(tf.keras.models.Model):
    """
    Multi-gate Mixture-of-Experts.
    Reference: Jiaqi Ma et al. Modeling Task Relationships in Multi-task Learning with Multi-gate
    Mixture-of-Experts. KDD. 2018.
    """

    def __init__(self,
                 feature_columns,
                 num_experts=3,
                 expert_units_list=None,
                 expert_activation='relu',
                 num_towers=2,
                 tower_units_list=None,
                 tower_activation='relu'
                 ):
        """

        Args:
            feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which basically
                contains all feature fields.
            num_experts: Number of experts.
            expert_units_list: Dimension of fully connected stack outputs in expert dense block.
            expert_activation: Activation to use in expert dense block.
            num_towers: Number of towers (tasks).
            tower_units_list: Dimension of fully connected stack outputs in tower dense block.
            tower_activation: Activation to use in tower dense block.
        """
        super(MMOE, self).__init__()
        if expert_units_list is None:
            expert_units_list = [256, 128, 64]
        if tower_units_list is None:
            tower_units_list = [256, 128, 64]
        self.num_experts = num_experts
        self.num_towers = num_towers
        self.input_layer = DenseFeatures(feature_columns)
        self.experts = [blocks.DenseBlock(units_list=expert_units_list, activation=expert_activation)
                        for _ in range(self.num_experts)]
        self.towers = [blocks.DenseBlock(units_list=tower_units_list, activation=tower_activation)
                       for _ in range(self.num_towers)]
        self.scores = [Dense(units=1, activation='sigmoid') for _ in range(self.num_towers)]
        self.gates = [Dense(units=self.num_experts, activation='softmax', use_bias=False)
                      for _ in range(self.num_towers)]

    def call(self, inputs, use_tower=0, training=None, mask=None):
        inputs = self.input_layer(inputs)

        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs = tf.transpose(tf.convert_to_tensor(expert_outputs), [1, 0, 2])

        gate_outputs = self.gates[use_tower](inputs)
        gate_outputs = tf.expand_dims(gate_outputs, -1)

        outputs = gate_outputs * expert_outputs
        outputs = tf.reduce_sum(outputs, axis=1)

        logits = self.towers[use_tower](outputs)
        return self.scores[use_tower](logits)
