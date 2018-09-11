from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees.stacked_single_target_regression_hoeffding_tree import \
    StackedSingleTargetRegressionHoeffdingTree

# from skmultiflow.trees.multi_target_regression_hoeffding_tree import \
#     MultiTargetRegressionHoeffdingTree


def demo(input_file, output_file=None):
    """ _test_mtr_regression

    This demo demonstrates how to evaluate a Multi-Target Regressor. The
    employed dataset is 'scm1d', which is contained in the data folder.

    Parameters
    ----------
    input_file: string
        A string describind the path for the input dataset

    output_file: string
        The name of the csv output file

    """
    # Setup the File Stream
    # stream = FileStream("../data/datasets/covtype.csv", -1, 1)
    # stream = WaveformGenerator()
    # stream.prepare_for_use()
    stream = FileStream(input_file, n_targets=16)
    stream.prepare_for_use()
    # Setup the classifier
    # classifier = SGDClassifier()
    # classifier = PassiveAggressiveClassifier()
    classifier = StackedSingleTargetRegressionHoeffdingTree(leaf_prediction='adaptive')
    # classifier = MultiTargetRegressionHoeffdingTree(leaf_prediction='adaptive')
    # classifier = PerceptronMask()

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=1, batch_size=1, n_wait=200,
                                    max_time=1000, output_file=output_file,
                                    show_plot=False,
                                    metrics=['average_mean_square_error',
                                             'average_mean_absolute_error',
                                             'average_root_mean_square_error'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=pipe)


if __name__ == '__main__':
    demo('../data/datasets/mtr/scm1d.csv',
         '/home/mastelini/Desktop/sst_mtr_test_adaptive.csv')
