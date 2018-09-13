from skmultiflow.lazy import SAMKNN
from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential


def demo(output_file=None, instances=50000):
    """ _test_sam_knn_prequential

    This demo shows how to produce a prequential evaluation.

    The first thing needed is a stream. For this case we use a file stream 
    which gets its samples from the moving_squares.csv file, inside the datasets 
    folder.

    Then we need to setup a classifier, which in this case is an instance 
    of scikit-multiflow's SAMKNN. Then, optionally we create a 
    pipeline structure, initialized on that classifier.

    The evaluation is then run.

    Parameters
    ----------
    output_file: string
        The name of the csv output file

    instances: int
        The evaluation's max number of instances

    """
    # Setup the File Stream
    stream = FileStream("../data/datasets/moving_squares.csv")
    # stream = WaveformGenerator()
    stream.prepare_for_use()

    # Setup the classifier
    classifier = SAMKNN(n_neighbors=5, weighting='distance', max_window_size=1000, stm_size_option='maxACCApprox',
                        use_ltm=False)

    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=0, max_samples=instances, batch_size=1, n_wait=100, max_time=1000,
                                    output_file=output_file, show_plot=True)

    # Evaluate
    evaluator.evaluate(stream=stream, model=classifier)


if __name__ == '__main__':
    demo()
