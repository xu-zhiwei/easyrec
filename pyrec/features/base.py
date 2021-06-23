import tensorflow as tf
import numpy as np


class Feature:
    def __init__(self):
        pass

    def make_dataset(self):
        pass


if __name__ == '__main__':
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     {'one_hot_1': np.array([1, 2, 3]),
    #      'multi_hot_1': np.array([np.array([1, 3]), np.array([4, 5, 6]), np.array([3])]),
    #      'dense': np.array([[1, 2, 3], [2, 3, 3], [2, 3, 4]])
    #      }
    # )
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     {
    #         "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    #         "b": np.random.uniform(size=(5, 2))
    #     }
    # )
    # print(dataset)
    print(np.array([np.array([1, 3]), np.array([4, 5, 6]), np.array([3])], dtype=object))
