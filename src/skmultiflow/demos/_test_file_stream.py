from skmultiflow.trees import HoeffdingTreeClassifier

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import FileStream


def demo(): 

    # The classifier we will use (other options: SAMKNNClassifier, LeverageBaggingClassifier, SGD)
    h = HoeffdingTreeClassifier()

    # Setup Stream
    stream = FileStream("../data/datasets/sea_stream.csv")

    pretrain = 100
    evaluator = EvaluatePrequential(pretrain_size=pretrain, output_file='test_filestream.csv', max_samples=10000,
                                    batch_size=1, n_wait=1000, show_plot=True)
    evaluator.evaluate(stream=stream, model=h)


if __name__ == '__main__':
    demo()
