from skmultiflow.data import RandomRBFGeneratorDrift
from skmultiflow.evaluation.evaluate_stream_gen_speed import EvaluateStreamGenerationSpeed


def demo():
    """ _test_stream_speed
    
    This demo tests the sample generation speed of the file stream.
    
    """
    # Setup the stream
    # stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
    #                     "master/covtype.csv")
    stream = RandomRBFGeneratorDrift()

    # Test with RandomTreeGenerator
    # opt_list = [['-c', '2'], ['-o', '0'], ['-u', '5'], ['-v', '4']]
    # stream = RandomTreeGenerator(opt_list)

    # Setup the evaluator
    evaluator = EvaluateStreamGenerationSpeed(100000, float("inf"), None, 5)

    # Evaluate
    evaluator.evaluate(stream)


if __name__ == '__main__':
    demo()
