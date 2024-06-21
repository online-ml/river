from __future__ import annotations

import typing


from river import base

class SpaceSaving(base.Base):
    """Space-Saving algorithm for finding heavy hitters.[^1]

    The Space-Saving algorithm is designed to find the heavy hitters in a data stream using a
    hash map with a fixed amount of memory. It keeps track of the k most frequent items at any 
    given time, as well as their corresponding approximate frequency.

    Upon receiving a new item from the data stream, if it corresponds to a monitored element,
    the algorithm increments its counter. Conversely, if the received element does not match
    any monitored element, the algorithm finds the tuple with the smallest counter value and
    replaces its element with the new element, incrementing its counter.

    Parameters
    ----------
    k
        The maximum number of heavy hitters to store. The higher the value of k, the higher the 
        accuracy of the algorithm.

    Attributes
    ----------
    counts : dict
        A dictionary to store the counts of items. The keys correspond to the elements and the 
        values to their respective count.

    Methods
    -------
    update(x, w=1)
        Update the counts with the given element and weight.
    __getitem__(x) -> int
        Get the count of the given element.
    __len__() -> int
        Return the number of elements stored.
    total() -> int
        Return the total count.
    heavy_hitters() -> int
        Return the heavy hitters stored.

    Examples
    --------
    >>> from river import sketch

    >>> spacesaving = sketch.SpaceSaving(k=10)
    
    >>> for i in range(100):
    ...     spacesaving.update(i % 10)
    
    >>> print(len(spacesaving))
    10
    >>> print(spacesaving.total())
    100
    >>> print(spacesaving.heavy_hitters)
    {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10}
    >>> print(spacesaving[10])
    10
    

    References
    ----------
    - [^1]: Cormode, G., & Hadjieleftheriou, M. (2008). Finding Frequent Items in Data Streams. AT&T Labs–Research, Florham Park, NJ.
    """

    def __init__(self, k: int):
        self.k = k
        self.counts = {}

    def update(self, x: typing.Hashable, w: int = 1):
        """Update the counts with the given element."""
        if x in self.counts:
            self.counts[x] += w

        elif len(self.counts) >= self.k:
            min_count_key = min(self.counts, key=self.counts.get)
            self.counts[x] = self.counts.get(min_count_key) + 1
            del self.counts[min_count_key]

        else:
            self.counts[x] = w

    def __getitem__(self, x) -> int:
        """Get the count of the given element."""
        return self.counts.get(x, 0)
    
    def __len__(self):
        """Return the number of elements stored."""
        return len(self.counts)
    
    def total(self) -> int:
        """Return the total count."""
        return sum(self.counts.values())
    
    @property
    def heavy_hitters(self):
        """Return the heavy hitters stored."""
        return self.counts
