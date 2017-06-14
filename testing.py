__author__ = 'Guilherme Matsumoto'

import skmultiflow.demos.streamPlusClassifier as spc
import skmultiflow.demos.evalWithPipeline as ewp
import skmultiflow.demos.evalStreamSpeed as ess
import skmultiflow.demos.evalPrequential as evp


def demoSCP():
    """ Demo for a stream plus a classifier - No evaluation module used
    
        Used mainly before the Pipeline and Prequential evaluation were created    
        Stream: FileStream with covtype.csv
        Classifier: Pipeline with Perceptron classifier
        Evaluator: No evaluator
        
    :return: No return
    """
    spc.demo()

def demo_pipeline():
    """ Demo for a Prequential Evaluator with a Pipeline as the main Classifier
    
        For this example, the Pipeline was initialized with only one classifier and no transforms.
        Stream: FileStream with covtype.csv
        Classifier: Pipeline with PassiveAgressiveClassifier classifier
        Evaluator: PrequentialEvaluator
    
    :return: No return
    """
    ewp.demo()

def demo_preq():
    """ Demo for a Prequential evaluation
        
        Stream: FileStream with covtype.csv
        Classifier: Perceptron classifier
        Evaluator: PrequentialEvaluator
    
    :return: No returns 
    """
    evp.demo()

def demo_stream_speed():
    """ Demo for stream generation speed
    
        Stream: Various option - change in skmultiflow.demos.evalStreamSpeed
        Evaluator: EvaluateStreamGenerationSpeed
    
    :return: 
    """
    ess.demo()

if __name__ == '__main__':
    # NEW tests

    # Stream plus classifier - no evaluation
    #demoSCP()

    # Demo Prequential evaluation - go to skmultiflow.demos.evalPrequential to change demo parameters
    demo_preq()

    # Demo Prequential eval with pipeline - go to skmultiflow.demos.evalWithPipeline to change demo parameters
    #demo_pipeline()

    # Demo eval stream speed - go to skmultiflow.demos.evalStreamSpeed to change demo parameters
    #demo_stream_speed()

