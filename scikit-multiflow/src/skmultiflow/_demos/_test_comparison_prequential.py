from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import FileStream
from skmultiflow.lazy import KNNADWINClassifier
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
    stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
                        "master/covtype.csv")
    # stream = SEAGenerator(classification_function=2, sample_seed=53432, balance_classes=False)
    # Setup the classifier
    clf = SGDClassifier()
    # classifier = KNNADWINClassifier(n_neighbors=8, max_window_size=2000,leaf_size=40, nominal_attributes=None)
    # classifier = OzaBaggingADWINClassifier(base_estimator=KNNClassifier(n_neighbors=8, max_window_size=2000,
    #                                                                     leaf_size=30))
    clf_one = KNNADWINClassifier(n_neighbors=8, max_window_size=1000, leaf_size=30)
    # clf_two = KNNClassifier(n_neighbors=8, max_window_size=1000, leaf_size=30)
    # clf_two = LeveragingBaggingClassifier(base_estimator=KNNClassifier(), n_estimators=2)

    t_one = OneHotToCategorical([[10, 11, 12, 13],
                                 [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                  27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    # t_two = OneHotToCategorical([[10, 11, 12, 13],
    #                             [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    #                              27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    #                              40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])

    pipe_one = Pipeline([('one_hot_to_categorical', t_one), ('KNNClassifier', clf_one)])
    # pipe_two = Pipeline([('one_hot_to_categorical', t_two), ('KNNClassifier', clf_two)])

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
