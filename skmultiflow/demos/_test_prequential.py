__author__ = 'Guilherme Matsumoto'

from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.classification.perceptron import PerceptronMask
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream, FileOption
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential


def demo(output_file=None):
    # Setup the File Stream
    opt = FileOption("FILE", "OPT_NAME", "../datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    #stream = WaveformGenerator()
    stream.prepare_for_use()

    # Setup the classifier
    #classifier = SGDClassifier()
    classifier = PerceptronMask()

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    eval = EvaluatePrequential(show_performance=True, pretrain_size=1000, show_kappa=True, max_instances=40000, batch_size=1,
                               show_scatter_points=False, n_wait=200, max_time=1000, output_file=output_file, track_global_kappa=True)

    # Evaluate
    eval.eval(stream=stream, classifier=pipe)

if __name__ == '__main__':
    demo('log1.csv')