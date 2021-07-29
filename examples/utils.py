from typing import Tuple

import pandas as pd
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


def transform_ragged_lists_to_sparse_tensor(ragged_lists: list):
    indices, values = [], []
    max_length = 0
    for i, ragged_list in enumerate(ragged_lists):
        for j, value in enumerate(ragged_list):
            indices.append((i, j))
            values.append(value)
        max_length = max(max_length, len(ragged_list))

    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=(len(ragged_lists), max_length)
    )


def get_vocabulary_list_from_ragged_list_series(series: pd.Series):
    values = []
    for ragged_list in series.values:
        values.extend(ragged_list)
    return set(values)
