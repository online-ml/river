import time
import numpy as np

from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.lazy.sam_knn import SAMKNN
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream

def demo_parameterized(h, dset="sea_stream.csv", show_plot=True): 
    # Setup Stream
    opt = FileOption("FILE", "OPT_NAME", "../datasets/"+dset, "CSV", False)
    stream = FileStream(opt, -1, 1)
    stream.prepare_for_use()

    # For each classifier, e...
    T_init = 100
    eval = EvaluatePrequential(pretrain_size=T_init, output_file='output.csv', max_samples=10000, batch_size=1, n_wait=1000, task_type='classification', show_plot=show_plot, plot_options=['performance'])
    eval.eval(stream=stream, classifier=h)

def demo():

    # The classifier we will use (other options: SAMKNN, LeverageBagging, SGD)
    h = [HoeffdingTree(), SAMKNN(), LeverageBagging(), SGDClassifier()]

    # Demo 1 -- plot should not fail
    demo_parameterized(h)

    # Demo 2 -- csv output should look nice
    demo_parameterized(h, "sea_stream.csv", False)

    # Demo 3 -- should not give "'NoneType' object is not iterable" error
    demo_parameterized(h, "covtype.csv", False)

if __name__ == '__main__':
    demo()


