from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean, AUC
from tensorflow.keras.optimizers import SGD

from easyrec import FFM
from examples.utils import transform_ragged_lists_to_sparse_tensor, train_validation_test_split


def main():
    # load the data
    dataset_path = Path(args.dataset_path)

    rating_df = pd.read_csv(dataset_path / 'ratings.dat', sep='::', engine='python', header=None,
                            names=['user_id', 'item_id', 'ctr', 'timestamp'])
    rating_df.loc[rating_df['ctr'] <= 3, 'ctr'] = 0
    rating_df.loc[rating_df['ctr'] > 3, 'ctr'] = 1
    rating_df.pop('timestamp')

    user_df = pd.read_csv(dataset_path / 'users.dat', sep='::', engine='python', header=None,
                          names=['user_id', 'sex_id', 'age_id', 'occupation_id', 'zip_code_id'])
    user_df['age_id'] = user_df['age_id'].astype(str)
    user_df['occupation_id'] = user_df['occupation_id'].astype(str)
    user_df['zip_code_id'] = user_df['zip_code_id'].astype(str)
    item_df = pd.read_csv(dataset_path / 'movies.dat', sep='::', engine='python', header=None,
                          names=['item_id', 'title', 'genre_ids'])
    item_df.pop('title')  # title is not used in the example
    item_df['genre_ids'] = item_df['genre_ids'].apply(lambda x: x.split('|'))

    df = pd.merge(rating_df, user_df, how='left', on='user_id')
    df = pd.merge(df, item_df, how='left', on='item_id')

    # construct the feature columns
    categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
    categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
    one_hot_feature_columns = [
        categorical_column_with_identity(key='user_id', num_buckets=df['user_id'].max() + 1, default_value=0),
        categorical_column_with_vocabulary_list(
            key='sex_id', vocabulary_list=set(df['sex_id'].values), num_oov_buckets=1),
        categorical_column_with_vocabulary_list(
            key='age_id', vocabulary_list=set(df['age_id'].values), num_oov_buckets=1),
        categorical_column_with_vocabulary_list(
            key='occupation_id', vocabulary_list=set(df['occupation_id'].values), num_oov_buckets=1),
        categorical_column_with_vocabulary_list(
            key='zip_code_id', vocabulary_list=set(df['zip_code_id'].values), num_oov_buckets=1),
        categorical_column_with_identity(key='item_id', num_buckets=df['item_id'].max() + 1, default_value=0),
    ]

    # hyper-parameter
    train_ratio, validation_ratio, test_ratio = [0.6, 0.2, 0.2]
    batch_size = 128
    learning_rate = 1e-1
    epochs = 15

    # construct the dataset
    labels = df.pop('ctr')
    features = dict(df)
    features['genre_ids'] = transform_ragged_lists_to_sparse_tensor(features['genre_ids'])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=200, seed=42)
    train_dataset, validation_dataset, test_dataset = train_validation_test_split(dataset,
                                                                                  len(df),
                                                                                  train_ratio,
                                                                                  validation_ratio
                                                                                  )
    train_dataset = train_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # initialize the environment for train
    output_ckpt_path = Path(args.output_ckpt_path)
    if args.input_ckpt_path:
        input_ckpt_path = Path(args.input_ckpt_path)
        model = tf.keras.models.load_model(args.input_ckpt_path)
        start_epoch = int(input_ckpt_path.name)
    else:
        model = FFM(
            one_hot_feature_columns,
            k=8
        )
        start_epoch = 0

    loss_obj = BinaryCrossentropy()
    optimizer = SGD(learning_rate=learning_rate)

    train_loss = Mean(name='train_loss')
    train_auc = AUC(name='train_auc')
    validation_loss = Mean(name='validation_loss')
    validation_auc = AUC(name='validation_auc')
    best_auc = 0

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_obj(y, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        train_auc(y, predictions)

    @tf.function
    def validation_step(x, y):
        predictions = model(x)
        loss = loss_obj(y, predictions)

        validation_loss(loss)
        validation_auc(y, predictions)

    # train
    for epoch in range(start_epoch, epochs):
        train_loss.reset_states()
        train_auc.reset_states()
        validation_loss.reset_states()
        validation_auc.reset_states()

        for features, labels in train_dataset:
            train_step(features, labels)
        for features, labels in validation_dataset:
            validation_step(features, labels)

        print('epoch: {}, train_loss: {}, train_auc: {}'.format(epoch + 1, train_loss.result().numpy(),
                                                                train_auc.result().numpy()))
        print('epoch: {}, validation_loss: {}, validation_auc: {}'.format(epoch + 1, validation_loss.result().numpy(),
                                                                          validation_auc.result().numpy()))

        model.save(output_ckpt_path / str(epoch + 1))
        if best_auc < validation_auc.result().numpy():
            best_auc = validation_auc.result().numpy()
            model.save(output_ckpt_path / 'best')

    # test
    @tf.function
    def test_step(x, y):
        predictions = model(x)
        loss = loss_obj(y, predictions)

        test_loss(loss)
        test_auc(y, predictions)

    model = tf.keras.models.load_model(output_ckpt_path / 'best')
    test_loss = Mean(name='test_loss')
    test_auc = AUC(name='test_auc')
    for features, labels in test_dataset:
        test_step(features, labels)
    print('test_loss: {}, test_auc: {}'.format(test_loss.result().numpy(),
                                               test_auc.result().numpy()))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dataset_path')
    parser.add_argument('--output_ckpt_path')
    parser.add_argument('--input_ckpt_path')
    args = parser.parse_args()
    main()
