__author__ = 'Guilherme Matsumoto'

from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.classification.Perceptron import PerceptronMask
from skmultiflow.core.Pipeline import Pipeline
from skmultiflow.data.FileStream import FileStream, FileOption
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.evaluation.EvaluatePrequential import EvaluatePrequential


def demo():
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    #stream = WaveformGenerator()
    stream.prepare_for_use()
    #classifier = SGDClassifier()
    classifier = PerceptronMask()
    pipe = Pipeline([('Classifier', classifier)])
    eval = EvaluatePrequential(show_performance=True, pretrain_size=1000, show_kappa=True, max_instances=40000, batch_size=1,
                               show_scatter_points=False, n_wait=200, max_time=1000, output_file='two.csv', track_global_kappa=False)
    eval.eval(stream=stream, classifier=pipe)