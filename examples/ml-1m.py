from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD

from pyrec.models import FM
from pyrec.utils import train_validation_test_split


def main():
    # load the data
    dataset_path = Path(args.dataset_path)
    df = pd.read_csv(dataset_path / 'ratings.dat', sep='::', engine='python', header=None,
                     names=['user_id', 'item_id', 'ctr', 'timestamp'])
    df.loc[df['ctr'] <= 3, 'ctr'] = 0
    df.loc[df['ctr'] > 3, 'ctr'] = 1
    df.pop('timestamp')

    # construct the feature columns
    categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
    indicator_column = tf.feature_column.indicator_column
    one_hot_feature_columns = [
        indicator_column(
            categorical_column_with_identity(key='user_id', num_buckets=df['user_id'].max(), default_value=0)),
        indicator_column(
            categorical_column_with_identity(key='item_id', num_buckets=df['item_id'].max(), default_value=0)),
    ]
    multi_hot_feature_columns = []
    dense_feature_columns = []

    # hyper-parameter
    train_size, validation_size, test_size = [0.6, 0.2, 0.2]
    batch_size = 64
    learning_rate = 1e-6
    epochs = 50

    # construct the dataset
    labels = df.pop('ctr')
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    dataset = dataset.shuffle(buffer_size=200, seed=42).batch(batch_size)
    train_dataset, validation_dataset, test_dataset = train_validation_test_split(
        dataset, len(df), train_size, validation_size
    )

    # train the model
    model = FM(
        one_hot_feature_columns,
        multi_hot_feature_columns,
        dense_feature_columns,
        use_dense_feature_columns=False,
    )
    print(model.trainable_variables)
    loss = BinaryCrossentropy()
    optimizer = SGD(learning_rate=learning_rate)
    for epoch in range(epochs):
        for features, labels in train_dataset:
            with tf.GradientTape() as tape:
                logits = model(features)
                loss_value = loss(labels, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print(loss_value.values)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dataset_path')
    args = parser.parse_args()
    main()
