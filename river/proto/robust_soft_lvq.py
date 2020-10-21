import numpy as np

from sklearn.metrics import euclidean_distances

from river.base import Classifier
from river.utils.skmultiflow_utils import check_random_state
from river.utils import dict2numpy


class RSLVQClassifier(Classifier):
    """Robust Soft Learning Vector Quantization for Streaming and Non-Streaming Data.

    By choosing another gradient descent method the Robust Soft Learning Vector Quantization
    (RSLVQ) method can be used as an adaptive version.

    Parameters
    ----------
    n_prototypes_per_class
        Number of prototypes per class.
    initial_prototypes
        Prototypes to start with (shape = [n_prototypes, n_features + 1]).
        If not given then the initialization is near the class mean.
        Class label must be placed as last entry of each prototype.
    sigma
        Variance of the distribution.
    seed
        Only used if `prototypes_per_class > 1`.
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
    gamma
        Decay rate used for momentum-based algorithm
    gradient_descent
        The gradient optimizer. To use momentum-based gradient descent,
        choose 'adadelta' instead of 'vanilla'.

    Attributes
    ----------
    prototypes
        Prototypes array with shape = (n_prototypes, n_features).
    prototypes_classes
        Prototypes classes array with shape = (n_prototypes).
    class_labels
        Class labels array with shape = (n_classes).

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import proto

    >>> dataset = datasets.Phishing()

    >>> model = proto.RSLVQClassifier(seed=42)

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 74.16%

    Notes
    -----
    * The RSLVQ [^2] can be used with vanilla SGD as gradient descent method or
      with a momentum-based gradient descent technique called Adadelta as proposed in [^1].
    * Example for one prototype per class on a binary classification problem:
      ```
      initial_prototypes = [[2.59922826, 2.57368134, 4.92501, 0],
                            [6.05801971, 6.01383352, 5.02135783, 1]]
     ```

    References
    ----------
    [^1]: Heusinger, M., Raab, C., Schleif, F.M.: Passive concept drift
          handling via momentum based robust soft learning vector quantization.
          In: Vellido, A., Gibert, K., Angulo, C., Martı́n Guerrero, J.D. (eds.)
          Advances in Self-Organizing Maps, Learning Vector Quantization,
          Clustering and Data Visualization. pp. 200–209. Springer International
          Publishing, Cham (2020)
    [^2]: Sambu Seo and Klaus Obermayer. 2003. Soft learning vector
          quantization. Neural Comput. 15, 7 (July 2003), 1589-1604

    """

    def __init__(self,
                 n_prototypes_per_class: int = 1,
                 initial_prototypes: np.ndarray = None,
                 sigma: float = 1.0,
                 seed: int or np.random.RandomState = None,
                 gradient_descent: str = 'vanilla',
                 gamma: float = 0.9):
        self.sigma = sigma
        self.seed = seed
        self.epsilon = 1e-8
        if n_prototypes_per_class < 0:
            raise ValueError(f'prototypes_per_class must be a positive int, '
                             f'got {n_prototypes_per_class}')
        self.n_prototypes_per_class = n_prototypes_per_class
        self._initial_fit = True
        self._class_labels = []
        self.learning_rate = 1 / sigma
        self.gamma = gamma
        self.gradient_descent = gradient_descent

        if sigma <= 0:
            raise ValueError('Sigma must be greater than 0')
        if gamma >= 1 or gamma < 0:
            raise ValueError('Decay rate gamma has to be between 0 and\
                             less than 1')
        allowed_gradient_optimizers = ['adadelta', 'vanilla']

        if gradient_descent not in allowed_gradient_optimizers:
            raise ValueError('{} is not a valid gradient optimizer, please\
                             use one of {}'
                             .format(gradient_descent,
                                     allowed_gradient_optimizers))

        if self.gradient_descent == 'adadelta':
            self._update_prototype = self._update_prototype_adadelta
        else:
            self._update_prototype = self._update_prototype_vanilla

        self.initial_prototypes = initial_prototypes
        self._prototype = None
        self._prototype_classes = None
        if self.initial_prototypes:
            self._initialize_prototypes()

        self._rng = check_random_state(self.seed)

    def _update_prototype_vanilla(self, j, x, y, prototypes):
        """Vanilla SGD"""
        d = x - prototypes[j]

        if self._prototype_classes[j] == y:
            # Attract prototype to data point
            self._prototype[j] += self.learning_rate * \
                                  (self._p(j, x, prototypes=self._prototype, y=y) -
                                   self._p(j, x, prototypes=self._prototype)) * d
        else:
            # Distance prototype from data point
            self._prototype[j] -= self.learning_rate * self._p(
                j, x, prototypes=self._prototype) * d

    def _update_prototype_adadelta(self, j, x, y, prototypes):
        """Implementation of Adadelta"""
        d = x - prototypes[j]

        if self._prototype_classes[j] == y:
            gradient = (self._p(j, x, prototypes=self._prototype, y=y) -
                        self._p(j, x, prototypes=self._prototype)) * d
        else:
            gradient = - self._p(j, x, prototypes=self._prototype) * d

        # Accumulate gradient
        self._squared_mean_gradient[j] = (self.gamma * self._squared_mean_gradient[j]
                                          + (1 - self.gamma) * gradient * gradient)

        # Compute update/step
        step = (np.sqrt((self._squared_mean_step[j] + self.epsilon)
                        / (self._squared_mean_gradient[j] + self.epsilon))
                * gradient)

        # Accumulate updates
        self._squared_mean_step[j] = (self.gamma * self._squared_mean_step[j]
                                      + (1 - self.gamma) * step * step)

        # Attract/Distract prototype to/from data point
        self._prototype[j] += step

    def _append_prototypes(self, x, y):
        n_features = x.size

        # initialize prototypes
        if self.initial_prototypes is None:
            prototype = np.zeros([self.n_prototypes_per_class, n_features],
                                 dtype=np.double)
            prototype_classes = np.empty([self.n_prototypes_per_class], dtype=int)

            if self.n_prototypes_per_class == 1:
                # If only one prototype we init it to x
                prototype[0: self.n_prototypes_per_class] = x
            else:
                # else we add some random noise
                prototype[0: self.n_prototypes_per_class] = x + (
                    self._rng.rand(self.n_prototypes_per_class, n_features) * 2 - 1)

            prototype_classes[0: self.n_prototypes_per_class] = y

            # Initialize or append to existing prototypes
            if self._prototype is None:
                self._prototype = prototype
                self._prototype_classes = prototype_classes
                if self.gradient_descent == 'adadelta':
                    self._squared_mean_gradient = np.zeros_like(self._prototype)
                    self._squared_mean_step = np.zeros_like(self._prototype)
            else:
                self._prototype = np.vstack((self._prototype, prototype))
                self._prototype_classes = np.hstack((self._prototype_classes, prototype_classes))
                if self.gradient_descent == 'adadelta':
                    self._squared_mean_gradient = np.vstack((self._squared_mean_gradient,
                                                             np.zeros_like(self._prototype)))
                    self._squared_mean_step = np.vstack((self._squared_mean_step,
                                                         np.zeros_like(self._prototype)))

    def _initialize_prototypes(self):
        self._prototype = self.initial_prototypes[:, :-1]
        self._prototype_classes = self.initial_prototypes[:, -1]

        if self.gradient_descent == 'adadelta':
            self._squared_mean_gradient = np.zeros_like(self._prototype)
            self._squared_mean_step = np.zeros_like(self._prototype)

    def learn_one(self, x, y):
        x_array = dict2numpy(x)
        try:
            y_coded = self._class_labels.index(y)
        except ValueError:
            self._class_labels.append(y)
            y_coded = self._class_labels.index(y)
            self._append_prototypes(x_array, y_coded)

        self._optimize(x_array, y_coded)
        return self

    def _optimize(self, x, y):
        n_prototypes = self._prototype_classes.size

        prototypes = self._prototype.reshape(n_prototypes, x.size)

        best_euclid_corr = np.inf
        best_euclid_incorr = np.inf
        correct_idx = -1
        incorrect_idx = -1
        # find nearest correct and nearest wrong prototype
        for j in range(prototypes.shape[0]):
            if self._prototype_classes[j] == y:
                eucl_dis = euclidean_distances(x.reshape(1, x.size),
                                               prototypes[j].reshape(1, prototypes[j].size))
                if eucl_dis < best_euclid_corr:
                    best_euclid_corr = eucl_dis
                    correct_idx = j
            else:
                eucl_dis = euclidean_distances(x.reshape(1, x.size),
                                               prototypes[j].reshape(1, prototypes[j].size))
                if eucl_dis < best_euclid_incorr:
                    best_euclid_incorr = eucl_dis
                    incorrect_idx = j
        # Update nearest wrong prototype and nearest correct prototype
        # if correct prototype isn't the nearest
        if best_euclid_incorr < best_euclid_corr:
            self._update_prototype(j=correct_idx, y=y, x=x, prototypes=prototypes)
            self._update_prototype(j=incorrect_idx, y=y, x=x, prototypes=prototypes)

    def predict_one(self, x: dict):
        """
        Predict class membership index for each input sample.

        This function does classification on an array of
        test vectors X.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples)
            Returns predicted values.
        """
        try:
            y_pred = int(self._prototype_classes[np.array([self._costf(dict2numpy(x), p)
                                                           for p in self._prototype]).argmax()])
            return self._class_labels[y_pred]
        except TypeError:
            # Model is empty, return class 0 as default
            return 0

    def _costf(self, x, w):
        d = (x - w)[np.newaxis].T
        d = d.T.dot(d)
        return - d / (2 * self.sigma)

    def _p(self, j, e, prototypes, y=None):
        if y is None:
            fs = [self._costf(e, w) for w in prototypes]
        else:
            fs = [self._costf(e, prototypes[i]) for i in
                  range(prototypes.shape[0]) if
                  self._prototype_classes[i] == y]

        fs_max = np.amax(fs)
        s = sum([np.math.exp(f - fs_max) for f in fs])
        o = np.math.exp(
            self._costf(e, prototypes[j]) - fs_max) / s
        return o

    def predict_proba_one(self, x: dict):
        """ Not implemented for this  method.
        """
        raise NotImplementedError('This method does not exist on Robust Soft '
                                  'Learning Vector Quantization')

    @property
    def prototypes(self):
        """The prototypes"""
        return self._prototype

    @property
    def prototypes_classes(self):
        """The prototypes classes"""
        return self._prototype_classes

    @property
    def class_labels(self):
        """The class labels"""
        return self._class_labels

    @property
    def _multiclass(self):
        return True
