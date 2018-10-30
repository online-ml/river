from skmultiflow.data import WaveformGenerator
from sklearn.linear_model import SGDClassifier
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.lazy import KNNAdwin
from skmultiflow.trees import HoeffdingTree


def demo(output_file=None, instances=40000):
    """ _test_comparison_holdout
    
    This demo will test a holdout evaluation task when more than one learner is 
    evaluated, which makes it a comparison task. 
    
    Parameters
    ----------
    output_file: string, optional
        If passed this parameter indicates the output file name. If left blank, 
        no output file will be generated.
    
    instances: int (Default: 40000)
        The evaluation's maximum number of instances.
    
    """
    # Setup the File Stream
    # stream = FileStream("../data/datasets/covtype.csv", -1, 1)
    stream = WaveformGenerator()
    stream.prepare_for_use()

    # Setup the classifier
    clf_one = HoeffdingTree()
    # clf_two = KNNAdwin(n_neighbors=8, max_window_size=2000)
    # classifier = PassiveAggressiveClassifier()
    # classifier = SGDRegressor()
    # classifier = PerceptronMask()

    # Setup the pipeline
    classifier = [clf_one]

    # Setup the evaluator
    evaluator = EvaluateHoldout(test_size=500, dynamic_test_set=True, max_samples=instances, batch_size=1, n_wait=5000,
                                max_time=1000, output_file=output_file, show_plot=True, metrics=['kappa'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=classifier)


if __name__ == '__main__':
    demo(output_file='test_comparison_holdout.csv', instances=50000)
