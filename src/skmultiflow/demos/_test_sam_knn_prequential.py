import numpy as np
from skmultiflow.classification.lazy.sam_knn import SAMKNN
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.core.pipeline import Pipeline

def demo(output_file=None, instances=50000):
    """ _test_sam_knn_prequential

    This demo shows how to produce a prequential evaluation.

    The first thing needed is a stream. For this case we use a file stream 
    which gets its samples from the movingSquares.csv file, inside the datasets 
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
    stream = FileStream("../datasets/movingSquares.csv", -1, 1)
    # stream = WaveformGenerator()
    stream.prepare_for_use()

    # Setup the classifier
    # classifier = SGDClassifier()
    # classifier = KNNAdwin(k=8, max_window_size=2000,leaf_size=40, categorical_list=None)
    # classifier = OzaBaggingAdwin(h=KNN(k=8, max_window_size=2000, leaf_size=30, categorical_list=None))
    classifier = SAMKNN(n_neighbors=5, knnWeights='distance', maxSize=1000, STMSizeAdaption='maxACCApprox',
                        useLTM=False)
    # classifier = SGDRegressor()
    # classifier = PerceptronMask()

    # Setup the pipeline
    # pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=0, max_samples=instances, batch_size=1, n_wait=100, max_time=1000,
                                    output_file=output_file, show_plot=True, metrics=['performance'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=classifier)


if __name__ == '__main__':
    demo()
