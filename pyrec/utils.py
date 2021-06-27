from typing import Tuple

import tensorflow as tf


def train_validation_test_split(dataset: tf.data.Dataset,
                                dataset_size: int,
                                train_size: float,
                                validation_size: float
                                ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_dataset = dataset.take(round(train_size * dataset_size))
    test_dataset = dataset.skip(round(train_size * dataset_size))
    validation_dataset = test_dataset.take(round(validation_size * dataset_size))
    test_dataset = test_dataset.skip(round(validation_size * dataset_size))
    return train_dataset, validation_dataset, test_dataset
