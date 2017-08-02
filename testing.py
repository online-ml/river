__author__ = 'Guilherme Matsumoto'

import skmultiflow.demos._test_stream_classifier as spc
import skmultiflow.demos._test_pipeline as ewp
import skmultiflow.demos._test_stream_speed as ess
import skmultiflow.demos._test_prequential as evp
import skmultiflow.demos._test_mol as emol
import skmultiflow.demos._test_comparison_prequential as comp
import skmultiflow.demos._test_kdtree_compare as kdc


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

def demo_multi_output():
    """ Demo for multi output learners
        
        Stream: from file music.csv
        Classifier: Various types, given by name to the MultiOutputLearner class
        Evaluator: Don't now yet
    
    :return: 
    """
    emol.demo()

def demo_compare():
    comp.demo(20000)

def demo_compare_kdtrees():
    kdc.demo()

if __name__ == '__main__':
    # NEW tests

    # Stream plus classifier - no evaluation
    #demoSCP()

    # Demo Prequential evaluation - go to skmultiflow.demos.evalPrequential to change demo parameters
    #demo_preq()

    # Demo Prequential eval with pipeline - go to skmultiflow.demos.evalWithPipeline to change demo parameters
    #demo_pipeline()

    # Demo eval stream speed - go to skmultiflow.demos.evalStreamSpeed to change demo parameters
    #demo_stream_speed()

    # Demo for multi output classification, trying with the music.csv from Jesse Read's repository
    #demo_multi_output()

    # Demo for learner comparison feature
    #demo_compare()

    # Demo for comparing kd tree efficiency
    demo_compare_kdtrees()

