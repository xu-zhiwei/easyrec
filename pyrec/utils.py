from typing import Tuple

import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import IndicatorColumn


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


def get_feature_column_shape(feature_column):
    if isinstance(feature_column, IndicatorColumn):
        return feature_column.categorical_column.number_buckets

