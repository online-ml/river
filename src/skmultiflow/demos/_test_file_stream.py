from skmultiflow.trees import HoeffdingTree

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import FileStream


def demo(): 

    # The classifier we will use (other options: SAMKNN, LeverageBagging, SGD)
    h = HoeffdingTree()

    # Setup Stream
    stream = FileStream("../data/datasets/sea_stream.csv")
    stream.prepare_for_use()

    pretrain = 100
    evaluator = EvaluatePrequential(pretrain_size=pretrain, output_file='test_filestream.csv', max_samples=10000,
                                    batch_size=1, n_wait=1000, show_plot=True)
    evaluator.evaluate(stream=stream, model=h)


if __name__ == '__main__':
    demo()
