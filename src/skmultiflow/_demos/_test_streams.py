from skmultiflow.data import FileStream
from skmultiflow.data import RandomRBFGeneratorDrift
from skmultiflow.data import SEAGenerator


def demo():
    """ _test_streams
    
    This demo tests if the streams are correctly generating samples.
    
    :return: 
    """
    stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
                        "master/covtype.csv")

    rbf_drift = RandomRBFGeneratorDrift(change_speed=41.00, n_centroids=50, model_seed=32523423, instance_seed=5435,
                                        n_classes=2, n_features=10, num_drift_centroids=50)

    sea = SEAGenerator()

    print('1 instance:\n')

    X,y = stream.next_sample()
    print(X)
    print(y)

    X, y = sea.next_sample()
    print(X)
    print(y)

    print('\n\n10 instances:\n')
    X,y = stream.next_sample(10)
    print(X)
    print(y)

    X, y = sea.next_sample(10)
    print(X)
    print(y)


if __name__ == '__main__':
    demo()
