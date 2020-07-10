from skmultiflow.data.file_stream import FileStream
from skmultiflow.transform import MissingValuesCleaner


def demo():
    """ _test_filters
    
    This demo test the MissingValuesCleaner filter. The transform is set 
    to clean any value equal to -47, replacing it with the median value 
    of the last 10 samples, or less if there aren't 10 samples available. 
    
    The output will be the 10 instances used in the transform. The first 
    9 are kept untouched, as they don't have any feature value of -47. The 
    last samples has its first feature value equal to -47, so it's replaced 
    by the median of the 9 first samples.
    
    """
    stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
                        "master/covtype.csv")

    filter = MissingValuesCleaner(-47, 'median', 10)

    X, y = stream.next_sample(10)

    X[9, 0] = -47

    for i in range(10):
        temp = filter.partial_fit_transform([X[i].tolist()])
        print(temp)

if __name__ == '__main__':
    demo()
