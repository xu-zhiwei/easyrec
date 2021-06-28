from typing import Tuple, List

import tensorflow as tf


def train_validation_test_split(dataset: tf.data.Dataset,
                                dataset_size: int,
                                train_size: float,
                                validation_size: float
                                ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    if train_size + validation_size >= 1:
        raise ValueError('train_size + validation_size should be less than 1')
    train_dataset = dataset.take(round(train_size * dataset_size))
    test_dataset = dataset.skip(round(train_size * dataset_size))
    validation_dataset = test_dataset.take(round(validation_size * dataset_size))
    test_dataset = test_dataset.skip(round(validation_size * dataset_size))
    return train_dataset, validation_dataset, test_dataset


def get_input_dimension(one_hot_feature_columns,
                        multi_hot_feature_columns,
                        dense_feature_columns,
                        inputs):
    one_hot_input_layer = tf.keras.layers.DenseFeatures(feature_columns=one_hot_feature_columns)
    multi_hot_input_layer = tf.keras.layers.DenseFeatures(feature_columns=multi_hot_feature_columns)
    dense_feature_input_layer = tf.keras.layers.DenseFeatures(feature_columns=dense_feature_columns)

