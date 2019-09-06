import math
import numpy as np
from sklearn.utils import validation
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from skmultiflow.core.base import ClassifierMixin, BaseSKMObject


class RobustSoftLearningVectorQuantization(ClassifierMixin, BaseSKMObject):
    """Robust Soft Learning Vector Quantization for Streaming and Non-Streaming Data.

    By choosing another gradient descent method the Robust Soft Learning Vector Quantization
    (RSLVQ) method can be used as an adaptive version.

    Parameters
    ----------
    prototypes_per_class: int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class, not implemented yet.
    initial_prototypes: array-like, shape =  [n_prototypes, n_features + 1], optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.

        | Example for one prototype per class on a binary classification problem:
        | `initial_prototypes = [[2.59922826, 2.57368134, 4.92501, 0], [6.05801971, 6.01383352, 5.02135783, 1]]`
    sigma : float, optional (default=1.0)
        Variance of the distribution.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    gamma : float, Decay rate, optional (default=0.9)
        Decay rate used for momentum-based algorithm
    gradient_descent: string, specify gradient optimizer, optional
    (default='vanilla')
        To use momentum-based gradient descent, choose 'adadelta' instead
        of 'vanilla'

    Attributes
    ----------
    prototypes : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features
    prototypes_classes : array-like, shape = [n_prototypes]
        Prototypes classes
    class_labels : array-like, shape = [n_classes]
        Array containing labels.

    Notes
    -----
    The RSLVQ [2]_ can be used with vanilla SGD as gradient descent method or
    with a momentum-based gradient descent technique called Adadelta as proposed in [1]_.

    References
    ----------
    .. [1] Heusinger, M., Raab, C., Schleif, F.M.: Passive concept drift
       handling via momentum based robust soft learning vector quantization.\
       In: Vellido, A., Gibert, K., Angulo, C., Martı́n Guerrero, J.D. (eds.)
       Advances in Self-Organizing Maps, Learning Vector Quantization,
       Clustering and Data Visualization. pp. 200–209. Springer International
       Publishing, Cham (2020)
    .. [2] Sambu Seo and Klaus Obermayer. 2003. Soft learning vector
       quantization. Neural Comput. 15, 7 (July 2003), 1589-1604
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, random_state=None, gradient_descent='vanilla',
                 gamma=0.9):
        self.sigma = sigma
        self.random_state = random_state
        self.epsilon = 1e-8
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.classes_ = []
        self.learning_rate = 1 / sigma
        self.gamma = gamma
        self.gradient_descent = gradient_descent

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
        self.squared_mean_gradient[j] = self.gamma * \
            self.squared_mean_gradient[j] + (1 - self.gamma) \
            * gradient ** 2

        # Compute update/step
        step = ((self.squared_mean_step[j] + self.epsilon) /
                (self.squared_mean_gradient[j] + self.epsilon)) ** 0.5 * \
            gradient

        # Accumulate updates
        self.squared_mean_step[j] = self.gamma * \
            self.squared_mean_step[j] + (1 - self.gamma) * step ** 2

        # Attract/Distract prototype to/from data point
        self.w_[j] += step

    def _validate_train_parms(self, train_set, train_lab, classes=None):
        random_state = validation.check_random_state(self.random_state)
        train_set, train_lab = validation.check_X_y(train_set,
                                                    train_lab.ravel())

        if self.initial_fit:
            if classes:
                self.classes_ = np.asarray(classes)
                self.protos_initialized = np.zeros(self.classes_.size)
            else:
                self.classes_ = unique_labels(train_lab)
                self.protos_initialized = np.zeros(self.classes_.size)

            # Validate that labels have correct format
            for i in range(len(self.classes_)):
                if i not in self.classes_:
                    raise ValueError('Labels have to be ascending int,\
                                     starting at 0, got {}'
                                     .format(self.classes_))

        nb_classes = len(self.classes_)
        nb_features = train_set.shape[1]

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            # ppc is int so we can give same number ppc to for all classes
            if self.prototypes_per_class < 0:
                raise ValueError('prototypes_per_class must be a positive int')
            # nb_ppc = number of protos per class
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        elif isinstance(self.prototypes_per_class, list):
            # its an array containing individual number of protos per class
            # - not fully supported yet
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    'values in prototypes_per_class must be positive')
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    'length of prototypes_per_class'
                    ' does not fit the number of classes'
                    'classes=%d'
                    'length=%d' % (nb_classes, nb_ppc.size))
        else:
            raise ValueError('Invalid data type for prototypes_per_class, '
                             'must be int or list of int')

        # initialize prototypes
        if self.initial_prototypes is None:
            if self.initial_fit:
                self.w_ = np.empty([np.sum(nb_ppc), nb_features],
                                   dtype=np.double)
                self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClassIdx in range(len(self.classes_)):
                actClass = self.classes_[actClassIdx]
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
                            random_state.rand(nb_prot, nb_features) * 2 - 1)

                    if math.isnan(self.w_[pos, 0]):
                        raise ValueError('Prototype on position {} for class\
                                         {} is NaN.'
                                         .format(pos, actClass))
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
            if set(self.c_w_) != set(self.classes_):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes_, self.c_w_))
        if self.initial_fit:
            if self.gradient_descent == 'adadelta':
                self.squared_mean_gradient = np.zeros_like(self.w_)
                self.squared_mean_step = np.zeros_like(self.w_)
            self.initial_fit = False

        return train_set, train_lab

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Fit the LVQ model to the given training data and parameters using
        gradient ascent.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the class labels of all samples in X
        classes : numpy.ndarray, optional (default=None)
            Contains all possible/known class labels. Usage varies depending
            on the learning method.
        sample_weight : Not used.

        Returns
        --------
        self
        """
        if set(unique_labels(y)).issubset(set(self.classes_)) or \
                self.initial_fit is True:
            X, y = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all \
                             classes in first call of fit/partial_fit'
                             .format(y))

        self._optimize(X, y)
        return self

    def _optimize(self, X, y):
        nb_prototypes = self.c_w_.size

        n_data, n_dim = X.shape
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        for i in range(n_data):
            xi = X[i]
            c_xi = int(y[i])
            best_euclid_corr = np.inf
            best_euclid_incorr = np.inf

            # find nearest correct and nearest wrong prototype
            for j in range(prototypes.shape[0]):
                if self.c_w_[j] == c_xi:
                    eucl_dis = euclidean_distances(xi.reshape(1, xi.size),
                                                   prototypes[j]
                                                   .reshape(1, prototypes[j]
                                                   .size))
                    if eucl_dis < best_euclid_corr:
                        best_euclid_corr = eucl_dis
                        corr_index = j
                else:
                    eucl_dis = euclidean_distances(xi.reshape(1, xi.size),
                                                   prototypes[j]
                                                   .reshape(1, prototypes[j]
                                                   .size))
                    if eucl_dis < best_euclid_incorr:
                        best_euclid_incorr = eucl_dis
                        incorr_index = j

            # Update nearest wrong prototype and nearest correct prototype
            # if correct prototype isn't the nearest
            if best_euclid_incorr < best_euclid_corr:
                self._update_prototype(j=corr_index, c_xi=c_xi, xi=xi,
                                       prototypes=prototypes)
                self._update_prototype(j=incorr_index, c_xi=c_xi, xi=xi,
                                       prototypes=prototypes)

    def predict(self, X):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples)
            Returns predicted values.
        """
        return np.array([self.c_w_[np.array([self._costf(xi, p)
                                             for p in self.w_]).argmax()]
                         for xi in X])

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

    def predict_proba(self, X):
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
        return self.classes_
