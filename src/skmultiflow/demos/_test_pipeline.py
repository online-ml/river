from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier

from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree


def demo():
    """ _test_pipeline
    
    This demo demonstrates the Pipeline structure seemingly working as a 
    learner, while being passed as parameter to an EvaluatePrequential 
    object.
     
    """
    # # Setup the stream
    # opt = FileOption("FILE", "OPT_NAME", "../datasets/covtype.csv", "CSV", False)
    # stream = FileStream(opt, -1, 1)
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
    eval = EvaluatePrequential(show_plot=True, pretrain_size=1000, max_instances=100000)

    # Evaluate
    eval.eval(stream=stream, classifier=pipe)

if __name__ == '__main__':
    demo()