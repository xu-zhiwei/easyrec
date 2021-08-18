# Background of example

*easyrec* provides a number of existing models proposed in recommender system fields. It is extremely easy to use and
what you only need is to prepare the input of models.

To quickly acquire the usage of *easyrec*, we have finished some examples and make them open-sourced
in [Github](https://github.com/xu-zhiwei/easyrec/tree/main/examples).

Here we take [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) (short as ml-1m) for dataset and
Factorization Machine (FM) for model as an example.

# Prepare dataset

After you have downloaded the dataset, you may clean the data as follows:

```python
from pathlib import Path
import pandas as pd

dataset_path = Path('/path/for/dataset/')

# load ratings.dat
rating_df = pd.read_csv(dataset_path / 'ratings.dat', sep='::', engine='python', header=None,
                        names=['user_id', 'item_id', 'ctr', 'timestamp'])
rating_df.loc[rating_df['ctr'] <= 3, 'ctr'] = 0
rating_df.loc[rating_df['ctr'] > 3, 'ctr'] = 1
rating_df.pop('timestamp')

# load users.dat
user_df = pd.read_csv(dataset_path / 'users.dat', sep='::', engine='python', header=None,
                      names=['user_id', 'sex_id', 'age_id', 'occupation_id', 'zip_code_id'])
user_df['age_id'] = user_df['age_id'].astype(str)
user_df['occupation_id'] = user_df['occupation_id'].astype(str)
user_df['zip_code_id'] = user_df['zip_code_id'].astype(str)

# load movies.dat
item_df = pd.read_csv(dataset_path / 'movies.dat', sep='::', engine='python', header=None,
                      names=['item_id', 'title', 'genre_ids'])
item_df.pop('title')  # title is not used in the example
item_df['genre_ids'] = item_df['genre_ids'].apply(lambda x: x.split('|'))

# join 3 tables
df = pd.merge(rating_df, user_df, how='left', on='user_id')
df = pd.merge(df, item_df, how='left', on='item_id')
```

Then, based on feature columns in Tensorflow 2, you can formally define the format of input for models and obtain the
dataset generator.

Note: detailed introduction of feature columns is illustrated in [Tutorial](https://easyrec-python.readthedocs.io/en/latest/tutorials.html#feature-columns).

```python
import tensorflow as tf

# define the feature columns
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


# construct dataset generator
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


train_ratio, validation_ratio, test_ratio = [0.6, 0.2, 0.2]
batch_size = 128

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
```

# Low-level APIs
Next, train the model according to Low-level APIs (or the High-level APIs mentioned below).

```python
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean, AUC
from tensorflow.keras.optimizers import SGD

learning_rate = 1e-1
epochs = 50

output_ckpt_path = Path(output_ckpt_path)
model = FM(
    one_hot_feature_columns,
    k=32
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
```

Finally, the model parameter with the best evaluation result can be loaded and carry out inference.

```python
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
```

# High-level APIs
Coming sooooooooon!
