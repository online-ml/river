from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.generators.random_rbf_generator_drift import RandomRBFGeneratorDrift
from skmultiflow.data.generators.sea_generator import SEAGenerator
from skmultiflow.options.file_option import FileOption


def demo():
    """ _test_streams
    
    This demo tests if the streams are correctly generating samples.
    
    :return: 
    """
    opt = FileOption('FILE', 'OPT_NAME', '../datasets/covtype.csv', 'csv', False)
    stream = FileStream(opt, -1, 1)
    stream.prepare_for_use()
    rbf_drift = RandomRBFGeneratorDrift(change_speed=41.00, num_centroids=50, model_seed=32523423, instance_seed=5435,
                                     num_classes=2, num_att=10, num_drift_centroids=50)
    rbf_drift.prepare_for_use()

    sea = SEAGenerator()

    print('1 instance:\n')

    X,y = stream.next_instance()
    print(X)
    print(y)

    X, y = sea.next_instance()
    print(X)
    print(y)

    print('\n\n10 instances:\n')
    X,y = stream.next_instance(10)
    print(X)
    print(y)

    X, y = sea.next_instance(10)
    print(X)
    print(y)

if __name__ == '__main__':
    demo()