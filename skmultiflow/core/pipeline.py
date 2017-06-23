__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.utils.utils import dict_to_list
from skmultiflow.core.base_object import BaseObject
from sklearn.utils import tosequence
import logging

class Pipeline(BaseObject):
    """ Pipeline
        
        Creates a Pipeline structure, that holds a set of Transforms and one Classifier.
        
        Sequentially execute each Pipeline module, applying transforms, fits, predicts etc.
    """
    def __init__(self, steps):
        """ Class initializer.
            __init__(self, steps)
            --------------------------------
            
            Creates a pipeline, which will execute the functions received in the dict.
            :param step: dictionary of functions to execute with one Classifier (has to be ordered).
        """

        #default values
        super().__init__()
        self.steps = tosequence(steps)
        self.active = False

        self._configure()
        pass

    def _configure(self):
        """ Initial Pipeline configuration.
        
            Validate the Pipeline's steps. Maybe some other functions later.
        
        :return: No return.
        """
        self._validate_steps()

    def predict(self, X):
        """ Sequentially applies all transforms and then predict with last step.
        
        :param X: A matrix of format (n_samples, n_features).
        :return: Returns the predicted class label.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt)

    def fit(self, X, y):
        """ Sequentially fit and transform data in all but last step, then fit the model in last step.
        
        :param X: A matrix of the format (n_samples, n_features).
        :param y: An array_like object of length n_samples, containing the true class labels.
        :return: self.
        """
        #self._validate_steps()
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y)
            else:
                Xt = transform.fit(Xt, y).transform(Xt)

        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """ Sequentially partial fit and transform data in all but last step, then partial fit data in last step.
        
        :param X: A matrix of the format (n_samples, n_features).
        :param y: An array_like object of length n_samples, containing the true class labels.
        :param classes: A list containing all the stream class labels (can be omitted after first partial_fit call.
        :return: self.
        """
        #
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, 'fit_transform'):
                Xt = transform.partial_fit_transform(Xt, y, classes=classes)
            else:
                Xt = transform.partial_fit(Xt, y, classes=classes).transform(Xt)

        if self._final_estimator is not None:
            self._final_estimator.partial_fit(Xt, y, classes)
        return self

    def partial_fit_predict(self, X, y):
        """ Partial fits and transforms data in all but last step, then partial fits and predicts in the last step
        
        :param X: A matrix of the format (n_samples, n_features).
        :param y: An array_like object of length n_samples, containing the true class labels.
        :return: The classifier's class label prediction.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, "partial_fit_transform"):
                Xt = transform.partial_fit_transform(Xt, y)
            else:
                Xt = transform.partial_fit(Xt, y).transform(Xt)

        if hasattr(self._final_estimator, "partial_fit_predict"):
            return self._final_estimator.partial_fit_predict(Xt, y)
        else:
            return self._final_estimator.partial_fit(Xt, y).predict(Xt)

    def partial_fit_transform(self, X, y = None):
        """ Partial fits and transforms data in all but last step, then partial_fit_transform in last step
        
        :param X: A matrix of the format (n_samples, n_features).
        :param y: An array_like object of length n_samples, containing the true class labels, can be omitted.
        :return: The transformed data.
        """
        pass

    def execute_cycle(self, *args):
        ''' Execute one Pipeline cycle - will probably be removed.
        
        :param args: matrix of the arguments to all step functions.
        :return: 
        '''
        pass

    def run(self, stream):
        """ Run the pipeline - will probably be removed.
        
        :param stream: 
        :return: 
        """
        pass

    def get_class_type(self):
        return 'estimator'

    def _validate_steps(self):
        """ Validates all the step, guaranteeing that there's at a Classifier as last step.
        
            Will raise errors if the steps don;t fit the required format.
            
        :return: No return.
        """

        #name, est = self.steps[len(self.steps) - 2]
        #has_evaluator = True if est.get_class_type() in ('evaluator') else False
        #name, est = self.steps[len(self.steps) - 1]
        #if hasattr(est, 'get_class_type'):
        #    has_classifier = True if est.get_class_type() in ('estimator') else False
        #else:
        #    has_classifier = False

        names, estimators = zip(*self.steps)
        #evaluator = transforms = classifier = None
        transforms = classifier = None
        #if has_classifier:
        classifier = estimators[-1]
        #if has_evaluator:
            #evaluator = estimators[-2]
            #transforms = estimators[:-2]
        #else:
            #transforms = estimators[:-1]
        transforms = estimators[:-1]

        self.active = True
        #self.has_evaluator = True

        for t in transforms:
            if t is None:
                continue
            else:
                if (not (hasattr(t, "fit") or hasattr(t, "fit_transform"))
                    or not hasattr(t, "transform")):
                    self.active = False
                    raise TypeError("All intermediate steps, including an evaluator, "
                                    "should implement fit and transform.")

        #if evaluator is not None and (not (hasattr(evaluator, "fit")
        #                                  or hasattr(evaluator, "fit_transform"))
        #                              or not hasattr(evaluator, "transform")):
        #    self.active = False
        #    self.has_evaluator = False
        #    raise TypeError("All intermediate steps, including an evaluator, "
        #                    "should implement fit and transform.")

        if classifier is not None and not hasattr(classifier, "partial_fit"):
            self.active = False
            raise TypeError("Last step of pipeline should implement partial_fit.")


    def named_steps(self):
        """ Creates a steps dict.
        
        :return: return a steps dictionary, so that each step can be accessed by name.
        """
        return dict(self.steps)

    def get_info(self):
        info = "Pipeline: "
        names, estimators = zip(*self.steps)
        classifier = estimators[-1]
        transforms = estimators[:-1]
        i = 0
        for t in transforms:
            if t.get_info() is not None:
                info += t.get_info()
                info += " #### "
            else:
                info += 'Transform: no info available'
            i += 1

        if classifier is not None:
            if hasattr(classifier, 'get_info'):
                info += classifier.get_info()
            else:
                info += 'Classifier: no info available'
        return info

    @property
    def _final_estimator(self):
        """ Easy to access classifier
        
        :return: Returns the Pipeline's classifier 
        """
        return self.steps[-1][-1]