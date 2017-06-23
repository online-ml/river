__author__ = 'Guilherme Matsumoto'

from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier

from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption


def demo():
    # Setup the stream
    opt = FileOption("FILE", "OPT_NAME", "../datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    stream.prepare_for_use()

    # Test with RandomTreeGenerator
    #opt_list = [['-c', '2'], ['-o', '0'], ['-u', '5'], ['-v', '4']]
    #stream = RandomTreeGenerator(opt_list)
    #stream.prepare_for_use()

    # Setup the classifier
    #classifier = PerceptronMask()
    #classifier = NaiveBayes()
    classifier = PassiveAggressiveClassifier()

    # Setup the pipeline
    pipe = Pipeline([('perceptron', classifier)])

    # Setup the evaluator
    eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=500000)

    # Evaluate
    eval.eval(stream=stream, classifier=pipe)

if __name__ == '__main__':
    demo()