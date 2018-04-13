import logging
from timeit import default_timer as timer

from skmultiflow.classification.multi_output_learner import MultiOutputLearner
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.metrics.metrics import *
from sklearn.linear_model.perceptron import Perceptron


def demo():
    """ _test_mol

    This demo tests the MOL learner on a file stream, which reads from 
    the music.csv file.

    The test computes the performance of the MOL learner as well as 
    the time to create the structure and classify all the samples in 
    the file.

    """
    # Setup logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Setup the file stream
    stream = FileStream("../datasets/music.csv", 0, 6)
    stream.prepare_for_use()

    # Setup the classifier, by default it uses Logistic Regression
    # classifier = MultiOutputLearner()
    # classifier = MultiOutputLearner(h=SGDClassifier(n_iter=100))
    classifier = MultiOutputLearner(h=Perceptron())

    # Setup the pipeline
    pipe = Pipeline([('classifier', classifier)])

    pretrain_size = 150
    logging.info('Pre training on %s samples', str(pretrain_size))
    X, y = stream.next_sample(pretrain_size)
    # classifier.fit(X, y)
    pipe.partial_fit(X, y, classes=stream.get_targets())
    count = 0
    true_labels = []
    predicts = []
    init_time = timer()
    logging.info('Evaluating...')
    while stream.has_more_samples():
        X, y = stream.next_sample()
        # p = classifier.predict(X)
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
