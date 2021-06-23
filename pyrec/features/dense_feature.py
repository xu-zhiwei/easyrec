from .base import Feature


class DenseFeature(Feature):
    """
    Object to save dense features, e.g., age.
    Shape: [batch, dimension]
    """
    def __init__(self, dimension):
        super(DenseFeature, self).__init__()
        self.dimension = dimension


