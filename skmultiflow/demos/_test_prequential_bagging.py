__author__= 'Guilherme Matsumoto'


from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.perceptron import Perceptron
from skmultiflow.classification.perceptron import PerceptronMask
from skmultiflow.classification.lazy.knn_adwin import KNNAdwin
from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.classification.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream, FileOption
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential


def demo(output_file=None, instances=40000):
    # Setup the File Stream
    opt = FileOption("FILE", "OPT_NAME", "../datasets/sea_big.csv", "CSV", False)
    stream = FileStream(opt, -1, 1)
    stream.prepare_for_use()

    # Setup the classifier
    #classifier = OzaBaggingAdwin(h=KNN(k=8, max_window_size=2000, leaf_size=30, categorical_list=None))
    classifier = LeverageBagging(h=KNN(k=8, max_window_size=2000, leaf_size=30), ensemble_length=1)

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    eval = EvaluatePrequential(pretrain_size=2000, max_instances=instances, batch_size=1, n_wait=200, max_time=1000,
                               output_file=output_file, task_type='classification', show_plot=True, plot_options=['kappa', 'kappa_t', 'performance'])

    # Evaluate
    eval.eval(stream=stream, classifier=pipe)

if __name__ == '__main__':
    demo('log1.csv', 20000)