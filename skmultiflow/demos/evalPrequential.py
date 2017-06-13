__author__ = 'Guilherme Matsumoto'

from skmultiflow.evaluation.EvaluatePrequential import EvaluatePrequential
from skmultiflow.data.FileStream import FileStream, FileOption
from sklearn.linear_model.perceptron import Perceptron


def demo():
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    stream.prepare_for_use()
    classifier = Perceptron()
    eval = EvaluatePrequential(show_plot=True, pretrain_size=1000)
    eval.eval(stream=stream, classifier=classifier)
    pass