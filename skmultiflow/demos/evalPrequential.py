__author__ = 'Guilherme Matsumoto'

from skmultiflow.evaluation.EvaluatePrequential import EvaluatePrequential
from skmultiflow.data.FileStream import FileStream, FileOption
from skmultiflow.core.pipeline.Pipeline import Pipeline
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.decomposition.incremental_pca import IncrementalPCA


def demo():
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    stream.prepare_for_use()
    classifier = SGDClassifier()
    pipe = Pipeline([('Classifier', classifier)])
    eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, show_kappa=True, max_instances=20000, batch_size=1,
                               show_scatter_points=False, n_wait=100)
    eval.eval(stream=stream, classifier=pipe)
    pass