cimport cython

cdef VectorDict _add_const(VectorDict res, other):
    # add the constant other to res then return res
    for key, value in res._map.items():
        res._map[key] = value + other
    return res

cdef VectorDict _add_dict(VectorDict res, dict other):
    # add the dict other to res then return res
    for key, value in other.items():
        res._map[key] = res._map.get(key, 0) + value
    return res

cdef VectorDict _sub_const(VectorDict res, other, int rev):
    # subtract the constant other to res then return res
    if rev:
        for key, value in res._map.items():
            res._map[key] = other - value
    else:
        for key, value in res._map.items():
            res._map[key] = value - other
    return res

cdef VectorDict _sub_dict(VectorDict res, dict other, int rev):
    # subtract the dict other to res then return res
    if rev:
        for key, value in other.items():
            res._map[key] = value - res._map.get(key, 0)
    else:
        for key, value in other.items():
            res._map[key] = res._map.get(key, 0) - value
    return res


cdef VectorDict _mul_const(VectorDict res, other):
    # multiply all values in res by other and return res
    for key, value in res._map.items():
        res._map[key] = other * value
    return res


cdef VectorDict _div_const(VectorDict res, other):
    # divide all values in res by other and return res
    for key, value in res._map.items():
        res._map[key] = value / other
    return res


cdef _dot_dicts(dict x, dict y):
    # return the dot product of the dictionaries x and y
    if len(x) < len(y):
        x, y = y, x
    res = 0
    for i, xi in x.items():
        res += xi * y.get(i, 0)
    return res


cdef class VectorDict(dict):
    cdef dict _map

    def __init__(self, other=None):
        """A dictionary-like object that supports vector-like operations.

        Supports addition and subtraction with a VectorDict, a dict or a constant.
        Supports multiplication and division by a constant.
        Supports dot product with a VectorDict or dict through the @ operator.

        Parameters:
            other: a VectorDict or dict to copy key-values from, or None
        """
        if other is None:
            pass
        elif isinstance(other, dict):
            self._map = dict(other)
        elif isinstance(other, VectorDict):
            other_ = <VectorDict> other
            self._map = dict(other_._map)
        else:
            raise ValueError("Unsupported type: {}".format(type(other)))

    # pass-through methods to the underlying dict

    def __contains__(self, key):
        return self._map.__contains__(key)

    def __delitem__(self, key):
        self._map.__delitem__(key)

    def __format__(self, format_spec):
        return self._map.__format__(format_spec)

    def __getitem__(self, key):
        try:
            return self._map[key]
        except KeyError:
            return 0

    def __iter__(self):
        return self._map.__iter__()

    def __len__(self):
        return self._map.__len__()

    def __repr__(self):
        return self._map.__repr__()

    def __setitem__(self, key, value):
        self._map[key] = value

    def __str__(self):
        return self._map.__str__()

    def clear(self):
        return self._map.clear()

    def items(self):
        return self._map.items()

    def keys(self):
        return self._map.keys()

    def pop(self, *args, **kwargs):
        return self._map.pop(*args, **kwargs)

    def popitem(self):
        return self._map.popitem()

    def setdefault(self, *args, **kwargs):
        return self._map.setdefault(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self._map.update(*args, **kwargs)

    def values(self):
        return self._map.values()

    # export methods

    def copy(self):
        return VectorDict(self._map)

    def todict(self):
        return self._map.copy()

    # operator methods

    def __eq__(self, other):
        if isinstance(other, VectorDict):  # VD == VD
            other_ = <VectorDict> other
            return self._map.__eq__(other_._map)
        if isinstance(other, dict):  # VD == D
            return self._map.__eq__(other)
        return NotImplemented

    def __add__(left, right):
        if isinstance(left, VectorDict):
            if isinstance(right, VectorDict):  # VD + VD
                right_ = <VectorDict> right
                return _add_dict(VectorDict(left), right_._map)
            if isinstance(right, dict):  # VD + D
                return _add_dict(VectorDict(left), right)
            try:  # VD + ?, try it
                return _add_const(VectorDict(left), right)
            except TypeError:
                return NotImplemented
        if isinstance(left, dict):  # D + VD
            return _add_dict(VectorDict(right), left)
        try:  # ? + VD, try it
            return _add_const(VectorDict(right), left)
        except TypeError:
            return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, VectorDict):  # VD += VD
            other_ = <VectorDict> other
            return _add_dict(self, other_._map)
        if isinstance(other, dict): # VD += D
            return _add_dict(self, other)
        try:  # VD += ?, try it
            return _add_const(self, other)
        except TypeError:
            return NotImplemented

    def __sub__(left, right):
        if isinstance(left, VectorDict):
            if isinstance(right, VectorDict):  # VD - VD
                right_ = <VectorDict> right
                return _sub_dict(VectorDict(left), right_._map, False)
            if isinstance(right, dict):  # VD - D
                return _sub_dict(VectorDict(left), right, False)
            try:  # VD - ?, try it
                return _sub_const(VectorDict(left), right, False)
            except TypeError:
                return NotImplemented
        if isinstance(left, dict):  # D - VD
            return _sub_dict(VectorDict(right), left, True)
        try:  # ? - VD, try it
            return _sub_const(VectorDict(right), left, True)
        except TypeError:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, VectorDict):  # VD -= VD
            other_ = <VectorDict> other
            return _sub_dict(self, other_._map, False)
        if isinstance(other, dict):  # VD -= D
            return _sub_dict(self, other, False)
        try:  # VD -= ?, try it
            return _sub_const(self, other, False)
        except TypeError:
            return NotImplemented

    def __mul__(left, right):
        try:
            if isinstance(left, VectorDict):  # VD * ?, try it
                return _mul_const(VectorDict(left), right)
            return _mul_const(VectorDict(right), left)  # ? * VD, try it
        except TypeError:
            return NotImplemented

    def __imul__(self, other):
        try:
            return _mul_const(self, other)  # VD *= ?, try it
        except TypeError:
            return NotImplemented

    def __truediv__(left, right):
        try:
            if isinstance(left, VectorDict):  # VD / ?, try it
                return _div_const(VectorDict(left), right)
        except TypeError:
            pass
        return NotImplemented

    def __itruediv__(self, other):
        try:
            return _div_const(self, other)  # VD /= ?, try it
        except TypeError:
            return NotImplemented

    def __matmul__(left, right):
        if isinstance(left, VectorDict):
            left_ = <VectorDict> left
            if isinstance(right, VectorDict):  # VD @ VD
                right_ = <VectorDict> right
                return _dot_dicts(left_._map, right_._map)
            if isinstance(right, dict):  # VD @ D
                return _dot_dicts(left_._map, right)
            return NotImplemented
        if isinstance(left, dict):  # D @ VD
            right_ = <VectorDict> right
            return _dot_dicts(right_._map, left)
        return NotImplemented

    def __neg__(self):
        # -VD
        res = VectorDict(self)
        for key, value in res._map.items():
            res._map[key] = -value
        return res

    def __pos__(self):
        # +VD
        return VectorDict(self)
