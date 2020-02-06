from array import array

from skmultiflow.core import clone
from skmultiflow.core.base import _pprint
from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.core import is_classifier
from skmultiflow.core import is_regressor


def test_clone():
    stream = SEAGenerator(random_state=1)

    learner = NaiveBayes()

    cnt = 0
    max_samples = 5000
    y_pred = array('i')
    X_batch = []
    y_batch = []
    y_proba = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        X_batch.append(X[0])
        y_batch.append(y[0])
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y, classes=stream.target_values)
        cnt += 1

    cloned = clone(learner)

    assert learner._observed_class_distribution != {} and cloned._observed_class_distribution == {}


def test_pprint():
    learner = HoeffdingTreeClassifier()

    expected_string = "binary_split=False, grace_period=200, leaf_prediction='nba',\n" \
                      " max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0,\n" \
                      " no_preprune=False, nominal_attributes=None, remove_poor_atts=False,\n" \
                      " split_confidence=1e-07, split_criterion='info_gain',\n" \
                      " stop_mem_management=False, tie_threshold=0.05"
    assert _pprint(learner.get_params()) == expected_string


def test_set_params():
    learner = HoeffdingTreeClassifier()
    original_info = learner.get_info()

    params = learner.get_params()
    params.update(leaf_prediction='nb', split_criterion='gini', remove_poor_atts=True)

    learner.set_params(**params)

    updated_info = learner.get_info()

    assert original_info != updated_info

    expected_info = "HoeffdingTreeClassifier(binary_split=False, grace_period=200, leaf_prediction='nb', " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, remove_poor_atts=True, split_confidence=1e-07, split_criterion='gini', " \
                    "stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_get_tags():
    classifier = HoeffdingTreeClassifier()
    regressor = HoeffdingTreeRegressor()
    multi_output_regressor = iSOUPTreeRegressor()

    classifier_tags = classifier._get_tags()

    expected_tags = {'X_types': ['2darray'],
                     '_skip_test': False,
                     'allow_nan': False,
                     'multilabel': False,
                     'multioutput': False,
                     'multioutput_only': False,
                     'no_validation': False,
                     'non_deterministic': False,
                     'poor_score': False,
                     'requires_positive_data': False,
                     'stateless': False}
    assert classifier_tags == expected_tags

    regressor_tags = regressor._get_tags()
    expected_tags = {'X_types': ['2darray'],
                     '_skip_test': False,
                     'allow_nan': False,
                     'multilabel': False,
                     'multioutput': False,
                     'multioutput_only': False,
                     'no_validation': False,
                     'non_deterministic': False,
                     'poor_score': False,
                     'requires_positive_data': False,
                     'stateless': False}
    assert regressor_tags == expected_tags

    multi_output_regressor_tags = multi_output_regressor._get_tags()
    expected_tags = {'X_types': ['2darray'],
                     '_skip_test': False,
                     'allow_nan': False,
                     'multilabel': False,
                     'multioutput': True,
                     'multioutput_only': True,
                     'no_validation': False,
                     'non_deterministic': False,
                     'poor_score': False,
                     'requires_positive_data': False,
                     'stateless': False}
    assert multi_output_regressor_tags == expected_tags


def test_is_classifier():
    learner = NaiveBayes()
    assert is_classifier(learner) is True


def test_is_regressor():
    learner = HoeffdingTreeRegressor()
    assert is_regressor(learner) is True
