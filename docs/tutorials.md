# Feature columns

The most important part of *easyrec* is feature columns, which basically determines whether you can train models within
your customized dataset.

In *easyrec*, the type of feature columns can be concluded into 2 groups, i.e., by data type or by usage.

# Data type-aware feature columns

## One hot

```python
"""
Example Args:
    one_hot_feature_columns: List[CategoricalColumn] encodes one hot feature fields, such as sex_id.
"""
```

## Multi hot

```python
"""
Example Args:
    multi_hot_feature_columns: List[CategoricalColumn] encodes multi hot feature fields, such as
        historical_item_ids.
"""
```

## Dense

```python
"""
Example Args:
    dense_feature_columns: List[NumericalColumn] encodes numerical feature fields, such as age.
"""
```

# Usage-aware feature columns

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
```
