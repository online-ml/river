from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.classification.perceptron import PerceptronMask
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.generators.regression_generator import RegressionGenerator


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
    # stream = FileStream("../datasets/covtype.csv", -1, 1)
    # stream = WaveformGenerator()
    # stream.prepare_for_use()
    stream = RegressionGenerator(n_samples=40000)
    # Setup the classifier
    # classifier = SGDClassifier()
    # classifier = PassiveAggressiveClassifier()
    classifier = SGDRegressor()
    # classifier = PerceptronMask()

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=1, max_samples=instances, batch_size=1, n_wait=1, max_time=1000,
                                    output_file=output_file, show_plot=True, metrics=['true_vs_predicts'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=pipe)


if __name__ == '__main__':
    demo('log1.csv', 40000)
