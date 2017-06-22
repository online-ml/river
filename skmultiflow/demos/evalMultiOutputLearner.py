__author__ = 'Guilherme Matsumoto'

from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from skmultiflow.data.FileStream import FileStream
from skmultiflow.options.FileOption import FileOption
from skmultiflow.classification.MultiOutputLearner import MultiOutputLearner
from skmultiflow.core.metrics import *
import logging
import numpy as np


def demo():
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Test with FileStream
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/Music.csv", "CSV", False)

    stream = FileStream(opt, 2, False, 6)
    stream.prepare_for_use()

    classifier = MultiOutputLearner()
    #classifier = MultiOutputLearner(h=linear_model.SGDClassifier(n_iter=100))

    #pipe = Pipeline([('perceptron', classifier)])
    #eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=500000)
    #eval.eval(stream=stream, classifier=pipe)

    '''
    for i in range(4):
        X, y = stream.next_instance(4)
        #print(X)
        print(y)
    '''


    X, y = stream.next_instance(250)
    classifier.fit(X, y)
    count = 0
    true_labels = []
    predicts = []
    while stream.has_more_instances():
        partial_predicts = []
        X, y = stream.next_instance()
        p = classifier.predict(X)
        predicts.extend(p)
        true_labels.extend(y)
    perf = hamming_score(true_labels, predicts)
    print(perf)
    logging.info('The classifier\'s static performance: %0.3f' % perf)

