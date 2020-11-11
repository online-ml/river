from sklearn.linear_model import SGDClassifier
from skmultiflow.core import Pipeline
from skmultiflow.data import WaveformGenerator
from skmultiflow.evaluation import EvaluateHoldout


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
    # stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
    #                     "master/covtype.csv")
    stream = WaveformGenerator()

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
    demo('test_holdout.csv', 400000)