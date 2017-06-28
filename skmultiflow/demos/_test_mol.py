__author__ = 'Guilherme Matsumoto'

import logging
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.data.file_stream import FileStream
from skmultiflow.options.file_option import FileOption
from skmultiflow.classification.multi_output_learner import MultiOutputLearner
from skmultiflow.core.metrics import *
from skmultiflow.core.pipeline import Pipeline
import numpy as np
from timeit import default_timer as timer


def demo():
    # Setup logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Setup the file stream
    opt = FileOption("FILE", "OPT_NAME", "../datasets/music.csv", "CSV", False)
    stream = FileStream(opt, 0, 6)
    stream.prepare_for_use()

    # Setup the classifier, by default it uses Logistic Regression
    classifier = MultiOutputLearner()
    #classifier = MultiOutputLearner(h=SGDClassifier(n_iter=100))
    #classifier = MultiOutputLearner(h=Perceptron())

    # Setup the pipeline
    pipe = Pipeline([('classifier', classifier)])

    pretrain_size = 200
    logging.info('Pre training on %s samples', str(pretrain_size))
    X, y = stream.next_instance(pretrain_size)
    #classifier.fit(X, y)
    pipe.fit(X, y)
    count = 0
    true_labels = []
    predicts = []
    init_time = timer()
    logging.info('Evaluating...')
    while stream.has_more_instances():
        X, y = stream.next_instance()
        #p = classifier.predict(X)
        p = pipe.predict(X)
        predicts.extend(p)
        true_labels.extend(y)
        count += 1
    perf = hamming_score(true_labels, predicts)
    logging.info('Evaluation time: %s s', str(timer() - init_time))
    logging.info('Total samples analyzed: %s', str(count))
    logging.info('The classifier\'s static Hamming score    : %0.3f' % perf)

if __name__ == '__main__':
    demo()