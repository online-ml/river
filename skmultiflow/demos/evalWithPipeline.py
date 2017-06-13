__author__ = 'Guilherme Matsumoto'

from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.data.FileStream import FileStream
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator
from skmultiflow.evaluation.EvaluatePrequential import EvaluatePrequential
from skmultiflow.core.pipeline.Pipeline import Pipeline
from skmultiflow.options.FileOption import FileOption
from skmultiflow.classification.Perceptron import PerceptronMask
from skmultiflow.classification.NaiveBayes import NaiveBayes
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier


def demo():
    # Test with FileStream
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    stream.prepare_for_use()

    # Test with RandomTreeGenerator
    #opt_list = [['-c', '2'], ['-o', '0'], ['-u', '5'], ['-v', '4']]
    #stream = RandomTreeGenerator(opt_list)
    #stream.prepare_for_use()

    #classifier = PerceptronMask()
    #classifier = NaiveBayes()
    classifier = PassiveAggressiveClassifier()

    pipe = Pipeline([('perceptron', classifier)])
    eval = EvaluatePrequential(show_plot=True, pretrain_size=1000)
    eval.eval(stream=stream, classifier=pipe)