import numpy as np

from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.lazy.sam_knn import SAMKNN
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream


def demo(): 

    # The classifier we will use (other options: SAMKNN, LeverageBagging, SGD)
    h = HoeffdingTree()

    # Setup Stream
    stream = FileStream("../datasets/sea_stream.csv", -1, 1)
    stream.prepare_for_use()

    pretrain = 100
    evaluator = EvaluatePrequential(pretrain_size=pretrain, output_file='output.csv', max_samples=10000, batch_size=1,
                                    n_wait=1000, show_plot=True, metrics=['performance'])
    evaluator.evaluate(stream=stream, model=h)


if __name__ == '__main__':
    demo()
