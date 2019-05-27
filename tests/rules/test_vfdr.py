from skmultiflow.rules import VFDR
from skmultiflow.data import AGRAWALGenerator
import numpy as np
from array import array
import sys


def test_vfdr():

    learner = VFDR(ordered_rules=True,
                   rule_prediction='first_hit',
                   nominal_attributes=[3, 4, 5],
                   expand_criterion='info_gain',
                   remove_poor_atts=True,
                   min_weight=100,
                   nb_prediction=False)
    stream = AGRAWALGenerator(random_state=11)
    stream.prepare_for_use()

    cnt = 0
    max_samples = 5000
    predictions = array('i')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                       0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                                       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    assert np.alltrue(predictions == expected_predictions)

    expected_info = "VFDR(drift_detector=None, expand_confidence=1e-07, expand_criterion='info_gain',\n" \
                    "     grace_period=200, max_rules=1000, min_weight=100, nb_prediction=False,\n" \
                    "     nb_threshold=0, nominal_attributes=[3, 4, 5], ordered_rules=True,\n" \
                    "     remove_poor_atts=True, rule_prediction='first_hit', tie_threshold=0.05)"
    assert learner.get_info() == expected_info

    expected_model_description = 'Rule 0 :Att (2) <= 39.550| class :0  {0: 1365.7101742993455}\n' + \
                                 'Rule 1 :Att (2) <= 58.180| class :1  {1: 1269.7307449971418}\n' + \
                                 'Rule 2 :Att (2) <= 60.910| class :0  {0: 66.24158839706533, 1: 54.0}\n' + \
                                 'Default Rule :| class :0  {0: 1316.7584116029348}'

    expected_model_description_ = 'Rule 0 :Att (2) <= 39.550| class :0  {0: 1365.7101742993455}\n' + \
                                 'Rule 1 :Att (2) <= 58.180| class :1  {1: 1269.7307449971418}\n' + \
                                 'Rule 2 :Att (2) <= 60.910| class :0  {0: 66.241588397065328, 1: 54.0}\n' + \
                                 'Default Rule :| class :0  {0: 1316.7584116029348}'

    assert (learner.get_model_description() == expected_model_description) or \
           (learner.get_model_description() == expected_model_description_)

    expected_model_measurements = {'Number of rules: ': 3, 'model_size in bytes': 62295}
    expected_model_measurements_ = {'Number of rules: ': 3, 'model_size in bytes': 73167}

    # if sys.version_info.minor != 6:
    #     assert (learner.get_model_measurements() == expected_model_measurements) or\
    #            (learner.get_model_measurements() == expected_model_measurements_)


def test_vfdr_foil():

    learner = VFDR(ordered_rules=False,
                   rule_prediction='weighted_sum',
                   nominal_attributes=[3,4,5],
                   expand_criterion='foil_gain',
                   remove_poor_atts=True,
                   min_weight=100,
                   nb_prediction=True)
    stream = AGRAWALGenerator(random_state=11)
    stream.prepare_for_use()

    cnt = 0
    max_samples = 5000
    predictions = array('i')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 1, 0, 0, 0, 0, 0, 0])

    assert np.alltrue(predictions == expected_predictions)

    expected_model_description = 'Rule 0 :Att (2) <= 25.450 | class: 1| class :0  {0: 464.44730579120136}\n' + \
                                 'Rule 1 :Att (4) = 3.000 | class: 0| class :0  {0: 95.0, 1: 45.0}\n' + \
                                 'Rule 2 :Att (2) <= 30.910 | class: 1| class :0  {0: 330.68821225514125}\n' + \
                                 'Default Rule :| class :0  {0: 573.0, 1: 336.0}'

    assert (learner.get_model_description() == expected_model_description)


def test_vfdr_hellinger():

    learner = VFDR(ordered_rules=False,
                   rule_prediction='weighted_sum',
                   nominal_attributes=[3,4,5],
                   expand_criterion='hellinger',
                   remove_poor_atts=True,
                   min_weight=100,
                   nb_prediction=True)
    stream = AGRAWALGenerator(random_state=11)
    stream.prepare_for_use()

    cnt = 0
    max_samples = 5000
    predictions = array('i')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                                       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
                                       0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])

    assert np.alltrue(predictions == expected_predictions)

    expected_model_description = 'Rule 0 :Att (2) > 58.180 and Att (5) = 4.000| class :0  {0: 202.0, 1: 3.0}\n' + \
                                 'Rule 1 :Att (2) <= 41.820| class :0  {0: 1387.1186637804824, 1: 151.83928023717402}\n' + \
                                 'Default Rule :| class :1  {0: 512.8813362195176, 1: 1356.160719762826}'

    expected_model_description_ = 'Rule 0 :Att (2) > 58.180 and Att (5) = 4.000| class :0  {0: 202.0, 1: 3.0}\n' + \
                                 'Rule 1 :Att (2) <= 41.820| class :0  {0: 1387.1186637804824, 1: 151.83928023717402}\n' + \
                                 'Default Rule :| class :1  {0: 512.8813362195176, 1: 1356.1607197628259}'

    if sys.version_info.minor != 6:
        assert (learner.get_model_description() == expected_model_description) or \
               (learner.get_model_description() == expected_model_description_)
