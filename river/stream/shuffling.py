import itertools
import random
import types
import typing


def shuffle(stream: typing.Iterator, buffer_size: int, seed: int = None):
    """Shuffles a stream of data.

    This works by maintaining a buffer of elements. The first `buffer_size` elements are stored in
    memory. Once the buffer is full, a random element inside the buffer is yielded. Every time an
    element is yielded, the next element in the stream replaces it and the buffer is sampled again.
    Increasing `buffer_size` will improve the quality of the shuffling.

    If you really want to stream over your dataset in a "good" random order, the best way is to
    split your dataset into smaller datasets and loop over them in a round-robin fashion. You may
    do this by using the ``roundrobin`` recipe from the `itertools` module.

    Parameters
    ----------
    stream
        The stream to shuffle.
    buffer_size
        The size of the buffer which contains the elements help in memory. Increasing this will
        increase randomness but will incur more memory usage.
    seed
        Random seed used for sampling.

    Examples
    --------

    >>> from river import stream

    >>> for i in stream.shuffle(range(15), buffer_size=5, seed=42):
    ...     print(i)
    0
    5
    2
    1
    8
    9
    6
    4
    11
    12
    10
    7
    14
    13
    3

    References
    ----------
    [^1]: [Visualizing TensorFlow's streaming shufflers](http://www.moderndescartes.com/essays/shuffle_viz/)

    """

    rng = random.Random(seed)

    # If stream is not a generator, then we coerce it to one
    if not isinstance(stream, types.GeneratorType):
        stream = iter(stream)

    # Initialize the buffer with the first buffer_size elements of the stream
    buffer = list(itertools.islice(stream, buffer_size))

    # Deplete the stream until it is empty
    for element in stream:

        # Pick a random element from the buffer and yield it
        i = rng.randint(0, len(buffer) - 1)
        yield buffer[i]

        # Replace the yielded element from the buffer with the new element from the stream
        buffer[i] = element

    # Shuffle the remaining buffer elements and yield them one by one
    rng.shuffle(buffer)
    yield from buffer
