from typing import List

__all__ = ["iter_counts"]


def iter_counts(X: List[List[str]]):
    """
    Given lists of words, return vocabularies with counts. This is useful
    for the VariableVocabKMeans model that expects this input.

    Parameters
    ----------
    X
        A list of lists of words (str)

    Example
    -------

    >>> X = [
    ...    ["one", "two"],
    ...    ["one", "four"],
    ...    ["one", "zero"],
    ...    ["four", "two"],
    ...    ["four", "four"],
    ...    ["four", "zero"]
    ... ]

    >>> for i, vocab in enumerate(stream.iter_counts(X)):
    ...    print(vocab)

    ... {'one': 1, 'two': 1}
    ... {'one': 1, 'four': 1}
    ... {'one': 1, 'zero': 1}
    ... {'four': 1, 'two': 1}
    ... {'four': 2}
    ... {'four': 1, 'zero': 1}
    """
    # Convert to counts (vocabulary)
    counts = []
    for words in X:
        vocab = {}
        for word in words:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
        counts.append(vocab)
    return counts
