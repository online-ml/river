from sklearn.linear_model.stochastic_gradient import SGDClassifier
from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.multilabel_generator import MultilabelGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential


def demo(output_file=None, instances=40000):
    """ _test_prequential_mol

    This demo shows the evaluation process of a MOL classifier, initialized 
    with sklearn's SGDClassifier.

    Parameters
    ----------
    output_file: string
        The name of the csv output file

    instances: int
        The evaluation's max number of instances

    """
    # Setup the File Stream
    # stream = FileStream("../data/datasets/music.csv", 0, 6)
    stream = MultilabelGenerator(n_samples=instances)
    # stream = WaveformGenerator()
    stream.prepare_for_use()

    # Setup the classifier
    classifier = MultiOutputLearner(SGDClassifier(n_iter=100))
    # classifier = SGDClassifier()
    # classifier = PassiveAggressiveClassifier()
    # classifier = SGDRegressor()
    # classifier = PerceptronMask()

    # Setup the pipeline
    pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=5000, max_samples=instances - 10000, batch_size=1, n_wait=200,
                                    max_time=1000, output_file=output_file, show_plot=True,
                                    metrics=['hamming_score', 'j_index', 'exact_match'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=pipe)


if __name__ == '__main__':
    demo('test_prequential_mol.csv', 50000)
