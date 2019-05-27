from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import FileStream
from skmultiflow.lazy import KNNAdwin
from sklearn.linear_model import SGDClassifier
from skmultiflow.core import Pipeline
from skmultiflow.transform import OneHotToCategorical


def demo(instances=2000):
    """ _test_comparison_prequential
    
    This demo will test a prequential evaluation when more than one learner is 
    passed, which makes it a comparison task.
    
    Parameters
    ----------
    instances: int
        The evaluation's maximum number of instances.
     
    """
    # Stream setup
    stream = FileStream("../data/datasets/covtype.csv", -1, 1)
    # stream = SEAGenerator(classification_function=2, sample_seed=53432, balance_classes=False)
    stream.prepare_for_use()
    # Setup the classifier
    clf = SGDClassifier()
    # classifier = KNNAdwin(n_neighbors=8, max_window_size=2000,leaf_size=40, nominal_attributes=None)
    # classifier = OzaBaggingAdwin(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30, categorical_list=None))
    clf_one = KNNAdwin(n_neighbors=8, max_window_size=1000, leaf_size=30)
    # clf_two = KNN(n_neighbors=8, max_window_size=1000, leaf_size=30)
    # clf_two = LeverageBagging(base_estimator=KNN(), n_estimators=2)

    t_one = OneHotToCategorical([[10, 11, 12, 13],
                            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    # t_two = OneHotToCategorical([[10, 11, 12, 13],
    #                        [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    #                        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])

    pipe_one = Pipeline([('one_hot_to_categorical', t_one), ('KNN', clf_one)])
    # pipe_two = Pipeline([('one_hot_to_categorical', t_two), ('KNN', clf_two)])

    classifier = [clf, pipe_one]
    # classifier = SGDRegressor()
    # classifier = PerceptronMask()

    # Setup the pipeline
    # pipe = Pipeline([('Classifier', classifier)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=2000, output_file='test_comparison_prequential.csv',
                                    max_samples=instances, batch_size=1, n_wait=200, max_time=1000, show_plot=True)

    # Evaluate
    evaluator.evaluate(stream=stream, model=classifier)


if __name__ == '__main__':
    demo(instances=10000)
