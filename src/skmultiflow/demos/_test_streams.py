from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.generators.random_rbf_generator_drift import RandomRBFGeneratorDrift
from skmultiflow.data.generators.sea_generator import SEAGenerator


def demo():
    """ _test_streams
    
    This demo tests if the streams are correctly generating samples.
    
    :return: 
    """
    stream = FileStream('../datasets/covtype.csv', -1, 1)
    stream.prepare_for_use()
    rbf_drift = RandomRBFGeneratorDrift(change_speed=41.00, n_centroids=50, model_seed=32523423, instance_seed=5435,
                                        n_classes=2, n_features=10, num_drift_centroids=50)
    rbf_drift.prepare_for_use()

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
