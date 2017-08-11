__author__ = 'Guilherme Matsumoto'

from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.classification.perceptron import PerceptronMask
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_holdout import EvaluateHoldout


def demo(output_file=None, instances=40000):
    # Setup the File Stream
    #opt = FileOption("FILE", "OPT_NAME", "../datasets/covtype.csv", "CSV", False)
    #stream = FileStream(opt, -1, 1)
    stream = WaveformGenerator()
    stream.prepare_for_use()
    # Setup the classifier
    classifier = SGDClassifier()
    #classifier = PassiveAggressiveClassifier()
    #classifier = SGDRegressor()
    #classifier = PerceptronMask()

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    eval = EvaluateHoldout(pretrain_size=10000, test_size=2000, dynamic_test_set=True, max_instances=instances, batch_size=1, n_wait=15000, max_time=1000,
                               output_file=output_file, task_type='classification', show_plot=True, plot_options=['kappa', 'kappa_t', 'performance'])

    # Evaluate
    eval.eval(stream=stream, classifier=pipe)

if __name__ == '__main__':
    demo('log1.csv', 400000)