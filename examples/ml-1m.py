from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean, AUC

from pyrec.models import FM
from pyrec.utils import train_validation_test_split, get_input_shape


def main():
    # load the data
    output_ckpt_path = Path(args.output_ckpt_path)
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
            categorical_column_with_identity(key='user_id', num_buckets=df['user_id'].max() + 1, default_value=0)),
        indicator_column(
            categorical_column_with_identity(key='item_id', num_buckets=df['item_id'].max() + 1, default_value=0)),
    ]
    multi_hot_feature_columns = []
    dense_feature_columns = []

    # hyper-parameter
    train_ratio, validation_ratio, test_ratio = [0.6, 0.2, 0.2]
    batch_size = 128
    learning_rate = 1
    k = 16
    epochs = 75

    # construct the dataset
    labels = df.pop('ctr')
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    dataset = dataset.shuffle(buffer_size=200, seed=42)
    train_dataset, validation_dataset, test_dataset = train_validation_test_split(dataset,
                                                                                  len(df),
                                                                                  train_ratio,
                                                                                  validation_ratio
                                                                                  )
    train_dataset = train_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # train the model
    one_hot_shape, multi_hot_shape, dense_shape = get_input_shape(
        one_hot_feature_columns,
        multi_hot_feature_columns,
        dense_feature_columns,
        next(iter(train_dataset))[0]
    )
    if args.input_ckpt_path:
        model = tf.keras.models.load_model(args.input_ckpt_path)
    else:
        model = FM(
            one_hot_feature_columns,
            multi_hot_feature_columns,
            dense_feature_columns,
            one_hot_shape,
            multi_hot_shape,
            k=k,
            use_dense_feature_columns=False,
        )
    loss = BinaryCrossentropy()
    optimizer = SGD(learning_rate=learning_rate)
    mean_loss = Mean()
    auc = AUC()
    best_auc = 0
    for epoch in range(epochs):
        mean_loss.reset_state()
        for batch, (features, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(features)
                loss_value = loss(labels, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            mean_loss.update_state(loss_value)
            # if (batch + 1) % 1000 == 0:
            #     print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {loss_value}')
        print(f'epoch: {epoch + 1}, mean_loss: {mean_loss.result().numpy()}')
        auc.reset_state()
        for features, labels in validation_dataset:
            logits = model(features)
            auc.update_state(labels, logits)
        print(f'epoch: {epoch + 1}, auc: {auc.result().numpy()}')

        (output_ckpt_path / str(epoch + 1)).mkdir(parents=True, exist_ok=True)
        model.save(str(output_ckpt_path / str(epoch + 1) / 'model'))
        if auc.result().numpy() > best_auc:
            best_auc = auc.result().numpy()
            (output_ckpt_path / 'best').mkdir(parents=True, exist_ok=True)
            model.save(str(output_ckpt_path / 'best' / 'model'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dataset_path')
    parser.add_argument('--output_ckpt_path')
    parser.add_argument('--input_ckpt_path')
    args = parser.parse_args()
    main()
