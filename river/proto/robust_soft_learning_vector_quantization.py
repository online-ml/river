import math

import numpy as np

from sklearn.utils import validation
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels

from river.base import Classifier
from river.utils.skmultiflow_utils import check_random_state
from river.utils import dict2numpy


class RobustSoftLearningVectorQuantization(Classifier):
    """Robust Soft Learning Vector Quantization for Streaming and Non-Streaming Data.

    By choosing another gradient descent method the Robust Soft Learning Vector Quantization
    (RSLVQ) method can be used as an adaptive version.

    Parameters
    ----------
    prototypes_per_class
        Number of prototypes per class.
    initial_prototypes: array-like, shape =  , optional
        Prototypes to start with (shape = [n_prototypes, n_features + 1]).
        If not given then the initialization is near the class mean.
        Class label must be placed as last entry of each prototype.
    sigma
        Variance of the distribution.
    seed
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
        Prototypes classes array with shape = (n_prototypes)
    class_labels
        Class labels array with shape = (n_classes)

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

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import proto

    >>> dataset = datasets.Phishing()

    >>> model = proto.RobustSoftLearningVectorQuantization(classes={0, 1}, seed=42)

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 74.08%

    """

    def __init__(self,
                 classes: set,
                 prototypes_per_class: int = 1,
                 initial_prototypes: np.ndarray = None,
                 sigma: float = 1.0,
                 seed: int or np.random.RandomState = None,
                 gradient_descent: str = 'vanilla',
                 gamma: float = 0.9):
        self.sigma = sigma
        self.seed = seed
        self.epsilon = 1e-8
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        if not isinstance(classes, set) or len(classes) == 0:
            raise AttributeError(f"Invalid classes. Possible classes must be specified.")
        self.classes = np.array([i for i in sorted(classes)])
        # Validate that labels have correct format
        if self.classes.size != max(self.classes) + 1:
            raise ValueError(f'Labels must be zero-based int, got {classes}')
        self.learning_rate = 1 / sigma
        self.gamma = gamma
        self.gradient_descent = gradient_descent
        self._class_map = dict()

        if sigma <= 0:
            raise ValueError('Sigma must be greater than 0')
        if prototypes_per_class <= 0:
            raise ValueError('Prototypes per class must be more than 0')
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

    def _update_prototype_vanilla(self, j, xi, c_xi, prototypes):
        """Vanilla SGD"""
        d = xi - prototypes[j]

        if self.c_w_[j] == c_xi:
            # Attract prototype to data point
            self.w_[j] += self.learning_rate * \
                (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                 self._p(j, xi, prototypes=self.w_)) * d
        else:
            # Distance prototype from data point
            self.w_[j] -= self.learning_rate * self._p(
                j, xi, prototypes=self.w_) * d

    def _update_prototype_adadelta(self, j, c_xi, xi, prototypes):
        """Implementation of Adadelta"""
        d = xi - prototypes[j]

        if self.c_w_[j] == c_xi:
            gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                        self._p(j, xi, prototypes=self.w_)) * d
        else:
            gradient = - self._p(j, xi, prototypes=self.w_) * d

        # Accumulate gradient
        self.squared_mean_gradient[j] = (self.gamma * self.squared_mean_gradient[j]
                                         + (1 - self.gamma) * gradient * gradient)

        # Compute update/step
        step = (np.sqrt((self.squared_mean_step[j] + self.epsilon)
                        / (self.squared_mean_gradient[j] + self.epsilon))
                * gradient)

        # Accumulate updates
        self.squared_mean_step[j] = (self.gamma * self.squared_mean_step[j]
                                     + (1 - self.gamma) * step * step)

        # Attract/Distract prototype to/from data point
        self.w_[j] += step

    def _validate_train_parms(self, train_set, train_lab):
        rng = check_random_state(self.seed)

        if self.initial_fit:
            self.protos_initialized = np.zeros(self.classes.size)

        nb_classes = self.classes.size
        nb_features = train_set.shape[1]

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            # ppc is int so we can give same number ppc to for all classes
            if self.prototypes_per_class < 0:
                raise ValueError(f'prototypes_per_class must be a positive int, '
                                 f'got {self.prototypes_per_class}')
            # nb_ppc = number of protos per class
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        elif isinstance(self.prototypes_per_class, list):
            # its an array containing individual number of protos per class
            # - not fully supported yet
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:    # noqa
                raise ValueError(f'Values in prototypes_per_class must be positive, got {nb_ppc}')
            if nb_ppc.size != nb_classes:    # noqa
                raise ValueError(f'Length of prototypes_per_class ({nb_ppc.size}) '    # noqa
                                 f'does not match the number of classes ({nb_classes})')
        else:
            raise ValueError('Invalid data type for prototypes_per_class, '
                             'must be int or list of int')

        # initialize prototypes
        if self.initial_prototypes is None:
            if self.initial_fit:
                self.w_ = np.empty([np.sum(nb_ppc), nb_features],
                                   dtype=np.double)
                self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes.dtype)
            pos = 0
            for actClassIdx in range(len(self.classes)):
                actClass = self.classes[actClassIdx]
                nb_prot = nb_ppc[actClassIdx]  # nb_ppc: prototypes per class
                if (self.protos_initialized[actClassIdx] == 0 and
                        actClass in unique_labels(train_lab)):
                    mean = np.mean(
                        train_set[train_lab == actClass, :], 0)

                    if self.prototypes_per_class == 1:
                        # If only one prototype we init it to mean
                        self.w_[pos:pos + nb_prot] = mean
                    else:
                        # else we add some random noise to distribute them
                        self.w_[pos:pos + nb_prot] = mean + (
                            rng.rand(nb_prot, nb_features) * 2 - 1)

                    if math.isnan(self.w_[pos, 0]):
                        raise ValueError(f'Prototype on position {pos} for '
                                         f'class {actClass} is NaN.')
                    else:
                        self.protos_initialized[actClassIdx] = 1

                    self.c_w_[pos:pos + nb_prot] = actClass
                pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.w_ = x[:, :-1]
            self.c_w_ = x[:, -1]
            if self.w_.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.w_.shape[0], self.w_.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.c_w_) != set(self.classes):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes, self.c_w_))
        if self.initial_fit:
            if self.gradient_descent == 'adadelta':
                self.squared_mean_gradient = np.zeros_like(self.w_)
                self.squared_mean_step = np.zeros_like(self.w_)
            self.initial_fit = False

        return train_set, train_lab

    def learn_one(self, x, y):
        """Fit the LVQ model to the given training data and parameters using
        gradient ascent.

        Parameters
        ----------
        x
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y
            An array-like with the class labels of all samples in X

        """
        x_array = np.reshape(dict2numpy(x), (1, -1))
        if not isinstance(y, int):
            if y in self._class_map:
                y = self._class_map[y]
            else:
                key = len(self._class_map) + 1
                self._class_map[key] = y
                y = self._class_map[key]

        y_array = np.asarray([y])
        if y in self.classes or self.initial_fit is True:
            x_array, y_array = self._validate_train_parms(x_array, y_array)
        else:
            raise ValueError(f'Unknown class {y} - please declare all possible labels using'
                             f'the classes attribute.')

        self._optimize(x_array, y_array)
        return self

    def _optimize(self, x, y):
        nb_prototypes = self.c_w_.size

        _, n_dim = x.shape
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        x = x[0]
        c_x = int(y[0])
        best_euclid_corr = np.inf
        best_euclid_incorr = np.inf
        correct_idx = -1
        incorrect_idx = -1
        # find nearest correct and nearest wrong prototype
        for j in range(prototypes.shape[0]):
            if self.c_w_[j] == c_x:
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
            self._update_prototype(j=correct_idx, c_xi=c_x, xi=x, prototypes=prototypes)
            self._update_prototype(j=incorrect_idx, c_xi=c_x, xi=x, prototypes=prototypes)

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
            y_pred = self.c_w_[np.array([self._costf(dict2numpy(x), p) for p in self.w_]).argmax()]
            if self._class_map:
                y_pred = self._class_map[y_pred]
            return y_pred
        except AttributeError:
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
                  self.c_w_[i] == y]

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
        return self.w_

    @property
    def prototypes_classes(self):
        """The prototypes classes"""
        return self.c_w_

    @property
    def class_labels(self):
        """The class labels"""
        return self.classes
