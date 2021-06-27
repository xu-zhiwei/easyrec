import tensorflow as tf

name = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        'name', ['bob', 'george', 'wanda'], num_oov_buckets=1)
)
columns = [name]
# df = {
#     'name': [
#         ['bob', 'george', 'george'],
#         ['wanda', 'wanda', 'wanda']
#     ]
# }
# print(tf.SparseTensor(indices=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], values=['bob', 'wanda', 'vv', 'bob', 'wanda'],
#                       dense_shape=(3, 100)))
# df = {
#     'name': tf.SparseTensor(indices=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
#                             values=['bob', 'wanda', 'vv', 'bob', 'wanda', 'wanda'],
#                             dense_shape=(3, 100))
# }
# d = DenseFeatures(feature_columns=columns)
# print(d(df))

# features = tf.io.parse_example({},
#                                features=tf.feature_column.make_parse_example_spec(columns))
# dense_tensor = input_layer(features, columns)
#
# dense_tensor == [[1, 0, 0]]  # If "name" bytes_list is ["bob"]
# dense_tensor == [[1, 0, 1]]  # If "name" bytes_list is ["bob", "wanda"]
# dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]

# 可能的方式：
# 1. DenseFeatures为正解，只接受Tensor和SparseTensor
# 2. 使用tf.sparse.from_tensor, id都从1开始, 因此，default_value填0

# import tensorflow as tf
#
# video_id = tf.feature_column.categorical_column_with_identity(
#     key='video_id', num_buckets=1000000, default_value=0)
# columns = [tf.feature_column.indicator_column(video_id)]
# features = {'video_id': tf.sparse.from_dense([[2, 85, 0, 0, 0],
#                                               [33, 78, 2, 73, 1]])}
# input_layer = tf.keras.layers.DenseFeatures(columns)
# dense_tensor = input_layer(features)
# print(dense_tensor)

# x = [
#     [1, 2, 3],
#     [1]
# ]
# x[1] = tf.pad(x[1], paddings=[[0, 2]])
# print(x)
