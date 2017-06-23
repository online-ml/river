__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.file_stream import FileStream, FileOption
from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.generators.random_rbf_generator_drift import RandomRBFGeneratorDrift
from skmultiflow.data.generators.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.generators.sea_generator import SEAGenerator
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_stream_gen_speed import EvaluateStreamGenerationSpeed


def demo():
    # Setup the stream
    opt = FileOption("FILE", "OPT_NAME", "../datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    stream.prepare_for_use()

    # Test with RandomTreeGenerator
    #opt_list = [['-c', '2'], ['-o', '0'], ['-u', '5'], ['-v', '4']]
    #stream = RandomTreeGenerator(opt_list)
    #stream.prepare_for_use()

    # Setup the evaluator
    eval = EvaluateStreamGenerationSpeed(100000, float("inf"), None, 5)

    # Evaluate
    eval.eval(stream)

if __name__ == '__main__':
    demo()