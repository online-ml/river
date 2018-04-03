import time
import numpy as np
from copy import copy

from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.lazy.sam_knn import SAMKNN
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream

def demo(): 

    # The classifier we will use (other options: SAMKNN, LeverageBagging, SGD)
    h = [HoeffdingTree(), SAMKNN(), LeverageBagging(), SGDClassifier()]

    # Setup Stream
    opt = FileOption("FILE", "OPT_NAME", "../datasets/sea_stream.csv", "CSV", False)
    stream = FileStream(opt, -1, 1)
    stream.prepare_for_use()

    # For each classifier, e...
    T_init = 100
    eval = EvaluatePrequential(pretrain_size=T_init, output_file='output.csv', max_instances=10000, batch_size=1, n_wait=1000, task_type='classification', show_plot=False, plot_options=['performance'])
    eval.eval(stream=stream, classifier=h)

if __name__ == '__main__':
    demo()


