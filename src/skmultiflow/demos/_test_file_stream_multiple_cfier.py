from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.lazy.sam_knn import SAMKNN
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream


def demo_parameterized(h, filename="covtype.csv", show_plot=True):
    # Setup Stream
    stream = FileStream("../datasets/" + filename, -1, 1)
    stream.prepare_for_use()

    # For each classifier, e...
    pretrain = 100
    evaluator = EvaluatePrequential(pretrain_size=pretrain, output_file='output.csv', max_samples=10000, batch_size=1,
                                    n_wait=1000, show_plot=show_plot, metrics=['performance'])
    evaluator.evaluate(stream=stream, model=h)


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
