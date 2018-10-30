from skmultiflow.core import Pipeline
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import WaveformGenerator
from skmultiflow.trees import HoeffdingTree


def demo():
    """ _test_pipeline
    
    This demo demonstrates the Pipeline structure seemingly working as a 
    learner, while being passed as parameter to an EvaluatePrequential 
    object.
     
    """
    # # Setup the stream
    # stream = FileStream("../data/datasets/covtype.csv", -1, 1)
    # stream.prepare_for_use()
    # # If used for Hoeffding Trees then need to pass indices for Nominal attributes

    # Test with RandomTreeGenerator
    # stream = RandomTreeGenerator(n_classes=2, n_numerical_attributes=5)
    # stream.prepare_for_use()

    # Test with WaveformGenerator
    stream = WaveformGenerator()
    stream.prepare_for_use()

    # Setup the classifier
    #classifier = PerceptronMask()
    #classifier = NaiveBayes()
    #classifier = PassiveAggressiveClassifier()
    classifier = HoeffdingTree()

    # Setup the pipeline
    pipe = Pipeline([('Hoeffding Tree', classifier)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_samples=100000)

    # Evaluate
    evaluator.evaluate(stream=stream, model=pipe)

if __name__ == '__main__':
    demo()