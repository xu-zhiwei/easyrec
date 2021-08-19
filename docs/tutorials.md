# Feature columns

The most important part of *easyrec* is feature columns, which basically determines whether you can train models within
your customized dataset.

In *easyrec*, the type of feature columns can be concluded into 2 groups, i.e., by data type or by usage.

## Data type-aware feature columns

### One hot

One hot feature columns indicate a list of **categorical feature columns**, and a data sample can belong to one and only one
of the categories.

```python
"""
Example Args in Functions:
    one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
"""
import tensorflow as tf

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
```

### Multi hot

Multi hot feature columns indicate a list of **categorical feature columns**, and a data sample can belong to one or more
than one of the categories.

```python
"""
Example Args in Function:
    multi_hot_feature_columns: List[CategoricalColumn] encodes multi hot feature fields, such as
        historical_item_ids.
"""
import tensorflow as tf

categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
multi_hot_feature_columns = [
    categorical_column_with_vocabulary_list(
        key='genre_ids', vocabulary_list=get_vocabulary_list_from_ragged_list_series(item_df['genre_ids']),
        num_oov_buckets=1
    )
]
```

### Dense

Dense feature columns indicate a list of **numerical feature columns**.

```python
"""
Example Args in Function:
    dense_feature_columns: List[NumericalColumn] encodes numerical feature fields, such as age.
"""
import tensorflow as tf

dense_feature_columns = [
    tf.feature_column.numeric_column(key='age')
]
```

## Usage-aware feature columns

These feature columns indicate a list of feature columns that can be **directly** feed into model.

```python
"""
Example Args:
    user_feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which
        basically contains user feature fields.
    item_feature_columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which
        basically contains item feature fields.
    feature columns: List[FeatureColumn] to directly feed into tf.keras.layers.DenseFeatures, which basically
        contains all feature fields.
"""
import tensorflow as tf

categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
indicator_column = tf.feature_column.indicator_column
user_feature_columns = [
    categorical_column_with_identity(key='user_id', num_buckets=df['user_id'].max() + 1, default_value=0),
    categorical_column_with_vocabulary_list(
        key='sex_id', vocabulary_list=set(df['sex_id'].values), num_oov_buckets=1),
    categorical_column_with_vocabulary_list(
        key='age_id', vocabulary_list=set(df['age_id'].values), num_oov_buckets=1),
    categorical_column_with_vocabulary_list(
        key='occupation_id', vocabulary_list=set(df['occupation_id'].values), num_oov_buckets=1),
    categorical_column_with_vocabulary_list(
        key='zip_code_id', vocabulary_list=set(df['zip_code_id'].values), num_oov_buckets=1),
]
```
