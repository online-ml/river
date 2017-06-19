__author__ = 'Guilherme Matsumoto'

from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier

from skmultiflow.core.Pipeline import Pipeline
from skmultiflow.data.FileStream import FileStream
from skmultiflow.evaluation.EvaluatePrequential import EvaluatePrequential
from skmultiflow.options.FileOption import FileOption


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
    eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=500000)
    eval.eval(stream=stream, classifier=pipe)