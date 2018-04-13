from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.classification.perceptron import PerceptronMask
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_holdout import EvaluateHoldout


def demo(output_file=None, instances=40000):
    """ _test_holdout
    
    This demo runs a holdout evaluation task with one learner. The default 
    stream is a WaveformGenerator. The default learner is a SGDClassifier, 
    which is inserted into a Pipeline structure. All the default values can 
    be changing by uncommenting/commenting the code below.
    
    Parameters
    ----------
    output_file: string
        The name of the csv output file
    
    instances: int
        The evaluation's max number of instances
         
    """
    # Setup the File Stream
    # stream = FileStream("../datasets/covtype.csv", -1, 1)
    stream = WaveformGenerator()
    stream.prepare_for_use()

    # Setup the classifier
    classifier = SGDClassifier()
    # classifier = PassiveAggressiveClassifier()
    # classifier = SGDRegressor()
    # classifier = PerceptronMask()

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    evaluator = EvaluateHoldout(test_size=2000, dynamic_test_set=True, max_samples=instances, batch_size=1,
                                n_wait=15000, max_time=1000, output_file=output_file, show_plot=True,
                                metrics=['kappa', 'kappa_t', 'performance'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=pipe)


if __name__ == '__main__':
    demo('log1.csv', 400000)