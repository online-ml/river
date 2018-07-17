from skmultiflow.trees import HoeffdingTree
from skmultiflow.lazy import SAMKNN
from skmultiflow.meta import LeverageBagging
from sklearn.linear_model import SGDClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import FileStream


def demo_parameterized(h, filename="covtype.csv", show_plot=True):
    # Setup Stream
    stream = FileStream("../data/datasets/" + filename)
    stream.prepare_for_use()

    # For each classifier, e...
    pretrain = 100
    evaluator = EvaluatePrequential(pretrain_size=pretrain, output_file='test_parametrized.csv', max_samples=10000,
                                    batch_size=1, n_wait=1000, show_plot=show_plot, metrics=['performance'])
    evaluator.evaluate(stream=stream, model=h)


def demo():

    # The classifier we will use (other options: SAMKNN, LeverageBagging, SGD)
    h1 = [HoeffdingTree(), SAMKNN(), LeverageBagging(), SGDClassifier()]
    h2 = [HoeffdingTree(), SAMKNN(), LeverageBagging(), SGDClassifier()]
    h3 = [HoeffdingTree(), SAMKNN(), LeverageBagging(), SGDClassifier()]

    # Demo 1 -- plot should not fail
    demo_parameterized(h1)

    # Demo 2 -- csv output should look nice
    demo_parameterized(h2, "sea_stream.csv", False)

    # Demo 3 -- should not give "'NoneType' object is not iterable" error
    demo_parameterized(h3, "covtype.csv", False)


if __name__ == '__main__':
    demo()
