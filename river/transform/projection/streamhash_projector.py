from anomaly.base_transformer import BaseTransformer
import numpy as np
#from pysad.core.base_transformer import BaseTransformer


class StreamhashProjector(BaseTransformer):
    """Streamhash projection method  from Manzoor et. al.that is similar (or equivalent) to SparseRandomProjection. :cite:`xstream` The implementation is taken from the `cmuxstream-core repository <https://github.com/cmuxstream/cmuxstream-core>`_.
        Args:
            num_components (int): The number of dimensions that the target will be projected into.
            density (float): Density parameter of the streamhash projection.
    """

    def __init__(self, num_components, density=1 / 3.0):
        super().__init__(num_components)
        self.keys = np.arange(0, num_components, 1)
        self.constant = np.sqrt(1. / density) / np.sqrt(num_components)
        self.density = density
        self.n_components = num_components

    def learn_one(self, X):
        """Fits particular (next) timestep's features to train the projector.
        Args:
            X (np.float array of shape (n_components,)): Input feature vector.
        Returns:
            object: self.
        """
        return self

    def transform_one(self, X):
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.
        Args:
            X (np.float array of shape (num_features,)): Input feature vector.
        Returns:
            projected_X (np.float array of shape (num_components,)): Projected feature vector.
        """
        X = X.reshape(1, -1)

        ndim = X.shape[1]

        feature_names = [str(i) for i in range(ndim)]

        R = np.array([[self._hash_string(k, f)
                       for f in feature_names]
                      for k in self.keys])

        Y = np.dot(X, R.T).squeeze()

        return Y

    def _hash_string(self, k, s):
        import mmh3
        hash_value = int(mmh3.hash(s, signed=False, seed=k)) / (2.0 ** 32 - 1)
        s = self.density
        if hash_value <= s / 2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0
