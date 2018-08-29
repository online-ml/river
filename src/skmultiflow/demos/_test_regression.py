# from sklearn.linear_model.stochastic_gradient import SGDRegressor
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.regression_generator import RegressionGenerator
from skmultiflow.trees.regression_hoeffding_tree import RegressionHoeffdingTree


def demo(output_file=None, instances=40000):
    """ _test_regression

    This demo demonstrates how to evaluate a regressor. The data stream used
    is an instance of the RegressionGenerator, which feeds an instance from
    sklearn's SGDRegressor.

    Parameters
    ----------
    output_file: string
        The name of the csv output file

    instances: int
        The evaluation's max number of instances

    """
    # Setup the File Stream
    # stream = FileStream("../data/datasets/covtype.csv", -1, 1)
    # stream = WaveformGenerator()
    # stream.prepare_for_use()
    stream = RegressionGenerator(n_samples=40000)
    # Setup the classifier
    # classifier = SGDClassifier()
    # classifier = PassiveAggressiveClassifier()
    classifier = RegressionHoeffdingTree()
    # classifier = PerceptronMask()

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=1, max_samples=instances, batch_size=1, n_wait=200, max_time=1000,
                                    output_file=output_file, show_plot=False, metrics=['mean_square_error'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=pipe)


if __name__ == '__main__':
    demo('test_regression.csv', 40000)
