from typing import Tuple

import tensorflow as tf


def train_validation_test_split(dataset: tf.data.Dataset,
                                dataset_size: int,
                                train_ratio: float,
                                validation_ratio: float
                                ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    if train_ratio + validation_ratio >= 1:
        raise ValueError('train_size + validation_size should be less than 1')
    train_size, validation_size = round(train_ratio * dataset_size), round(validation_ratio * dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    validation_dataset = test_dataset.take(validation_size)
    test_dataset = test_dataset.skip(validation_size)
    return train_dataset, validation_dataset, test_dataset


def get_input_shape(one_hot_feature_columns,
                    multi_hot_feature_columns,
                    dense_feature_columns,
                    inputs):
    one_hot_dimension, multi_hot_dimension, dense_dimension = 0, 0, 0
    if one_hot_feature_columns:
        one_hot_input_layer = tf.keras.layers.DenseFeatures(feature_columns=one_hot_feature_columns)
        one_hot_dimension = one_hot_input_layer(inputs).shape[1]
    if multi_hot_feature_columns:
        multi_hot_input_layer = tf.keras.layers.DenseFeatures(feature_columns=multi_hot_feature_columns)
        multi_hot_dimension = multi_hot_input_layer(inputs).shape[1]
    if dense_feature_columns:
        dense_feature_input_layer = tf.keras.layers.DenseFeatures(feature_columns=dense_feature_columns)
        dense_dimension = dense_feature_input_layer(inputs).shape[1]
    return one_hot_dimension, multi_hot_dimension, dense_dimension
