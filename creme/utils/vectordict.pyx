cimport cython

import itertools
import numpy as np

missing = object()


cdef inline get_value(VectorDict vec, key):
    if vec._use_mask and key not in vec._mask:
        return 0
    value = vec._data.get(key, missing)
    if value is missing:
        if vec._use_factory:
            value = vec._default_factory()
            vec._data[key] = value
            return value
        return 0
    return value


cdef inline get_keys(VectorDict vec):
    if vec._lazy_mask:
        return filter(vec._mask.__contains__, vec._data.keys())
    return vec._data.keys()


cdef inline get_union_keys(VectorDict left, VectorDict right):
    left_keys = get_keys(left)
    right_keys = get_keys(right)
    right_only_keys = itertools.filterfalse(left._data.__contains__, right_keys)
    return itertools.chain(left_keys, right_only_keys)


cdef class VectorDict:
    cdef dict _data
    cdef object _mask
    cdef bint _use_mask
    cdef bint _lazy_mask
    cdef bint _use_factory
    cdef object _default_factory

    def __init__(self, data=None, default_factory=None, mask=None, copy=False):
        """A dictionary-like object that supports vector-like operations.

        Supports addition (+), subtraction (-), multiplication (*) and division
        (-) with a VectorDict or a scalar.
        Supports dot product (@) with a VectorDict.
        A scalar is any object that supports the four arithmetic operations
        with the dictionary's values.

        If mask is not None, any key which is contained in mask is said to be
        masked while other keys are said to be unmasked.
        If mask is None, any keys is said to be unmasked.
        If mask is not None and copy is True, only the key-values for keys in
        both data and mask will be used to initialized the dictionary.
        If mask is not None and copy is False, the mask will be applied lazily,
        which enables a faster initialization but potentially slower operations.

        If default_factory is not None, it is called whenever an unmasked
        missing key is accessed, either externally with __getitem__ or
        internally as part of an element-wise numeric operation such as
        addition, and the result is inserted as the value for that key.
        If a masked key, or an umasked missing key when default_factory is None,
        is accessed externally through __getitem___ a KeyError
        exception is raised, and if it is accessed internally as part of an
        operation, its value is taken as 0, but is not inserted for that key.

        Parameters:
            data: a VectorDict or dict to initialize key-values from, or None
            default_value: a scalar, or None
            default_factory: a callable returning a scalar, or None
            mask: a VectorDict or set-like object such that keys not in mask
                will not be considered in operations and will always result in
                a KeyError if accessed by __getitem__, or None
            copy: if data and/or mask are specified, whether to store a copy of
                the underlying dictionaries or references at initialization
        """
        if data is None:
            data = dict()
        elif isinstance(data, VectorDict):
            data_ = <VectorDict> data
            data = data_._data
        elif not isinstance(data, dict):
            raise ValueError(f"Unsupported type for data: {type(data)}")
        if mask is None:
            if copy:
                data = dict(data)
        else:
            if isinstance(mask, VectorDict):
                mask = <VectorDict>(mask)._data
            if copy:
                mask = set(mask)
                data = {key: value
                        for key, value in data.items() if key in mask}
            # TODO check if mask is "set-like"
        self._data = data
        self._mask = mask
        self._use_mask = mask is not None
        self._lazy_mask = mask is not None and not copy
        self._use_factory = default_factory is not None
        self._default_factory = default_factory

    cdef dict _apply_mask(self, force_copy=False):
        # NOTE this is potentially slow (makes a copy if lazy_mask is True),
        #      use with caution
        if not self._lazy_mask:
            if force_copy:
                return dict(self._data)
            return self._data
        return {key: value for key, value in self._data.items()
                if key in self._mask}

    def with_mask(self, mask, copy=False):
        return VectorDict(self._data, self._default_factory, mask, copy)

    def to_dict(self):
        return self._apply_mask(force_copy=True)

    def to_numpy(self, fields):
        return np.array([get_value(self, f) for f in fields])

    # pass-through methods to the underlying dict

    def __contains__(self, key):
        contains = self._data.__contains__(key)
        if not self._lazy_mask:
            return contains
        return contains and self._mask.__contains__(key)

    def __delitem__(self, key):
        if self._lazy_mask and key not in self._mask:
            raise KeyError(key)
        self._data.__delitem__(key)

    def __format__(self, format_spec):
        return self._apply_mask().__format__(format_spec)

    def __getitem__(self, key):
        if self._use_mask and key not in self._mask:
            raise KeyError(key)
        try:
            return self._data[key]
        except KeyError:
            if self._use_factory:
                value = self._default_factory()
                self._data[key] = value
                return value
            raise

    def __iter__(self):
        return self._apply_mask().__iter__()

    def __len__(self):
        if self._lazy_mask:
            return len(self._data.keys() - self._mask)
        return self._data.__len__()

    def __repr__(self):
        return self._apply_mask().__repr__()

    def __setitem__(self, key, value):
        if self._use_mask and key not in self._mask:
            raise KeyError(key)
        self._data[key] = value

    def __str__(self):
        return self._apply_mask().__str__()

    def clear(self):
        if self._lazy_mask:
            keep = {key: value for key, value in self._data.items()
                    if key in self._mask}
            self._data.clear()
            self._data.update(keep)
        else:
            self._data.clear()

    def get(self, key, *args, **kwargs):
        if self._lazy_mask and key not in self._mask:
            return dict().get(key, *args, **kwargs)
        return self._data.get(key, *args, **kwargs)

    def items(self):
        return self._apply_mask().items()

    def keys(self):
        return self._apply_mask().keys()

    def pop(self, *args, **kwargs):
        return self._data.pop(*args, **kwargs)

    def popitem(self):
        if self._lazy_mask:
            keep = []
            while True:
                (key, value) = self._data.popitem()
                if key in self._mask:
                    keep.append((key, value))
                else:
                    break
            for key_, value_ in keep:
                self._data[key_] = value_
            return key, value
        return self._data.popitem()

    def setdefault(self, key, *args, **kwargs):
        if self._lazy_mask and key not in self._mask:
            return dict().setdefault(key, *args, **kwargs)
        return self._data.setdefault(key, *args, **kwargs)

    def update(self, *args, **kwargs):
        if self._lazy_mask:
            keep1 = {key: value for key, value in self._data.items()
                     if key in self._mask}
            self._data.update(*args, **kwargs)
            keep2 = {key: value for key, value in self._data.items()
                     if key not in self._mask}
            self._data.clear()
            self._data.update(**keep1, **keep2)
        self._data.update(*args, **kwargs)

    def values(self):
        return self._apply_mask().values()

    # operator methods

    def __eq__(left, right):
        if isinstance(right, VectorDict):
            left, right = right, left
        left_ = <VectorDict> left
        if isinstance(right, VectorDict):
            right_ = <VectorDict> right
            return left_._apply_mask().__eq__(right_._apply_mask())
        elif isinstance(right, dict):
            return left_._apply_mask().__eq__(right)
        else:
            return NotImplemented

    def __add__(left, right):
        if isinstance(right, VectorDict):
            left, right = right, left
        left_ = <VectorDict> left
        if isinstance(right, VectorDict):  # vec + vec
            left_, right_ = <VectorDict> right, left_
            res = dict()
            for key in get_union_keys(left_, right_):
                res[key] = get_value(left_, key) + get_value(right_, key)
        else:  # vec + scalar
            res = left_._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = value + right
            except TypeError:
                return NotImplemented
        return VectorDict(res)

    def __iadd__(VectorDict self, other):
        if isinstance(other, VectorDict):  # vec += vec
            other_ = <VectorDict> other
            for key in get_union_keys(self, other_):
                self._data[key] = get_value(self, key) + get_value(other_, key)
        else:  # vec += scalar
            try:
                for key in get_keys(self):
                    self._data[key] += other
            except TypeError:
                return NotImplemented
        return self

    def __sub__(left, right):
        if isinstance(left, VectorDict) and isinstance(right, VectorDict):
            # vec - vec
            left_, right_ = <VectorDict> left, <VectorDict> right
            res = dict()
            for key in get_union_keys(left_, right_):
                res[key] = get_value(left_, key) - get_value(right_, key)
        elif isinstance(left, VectorDict):  # vec - scalar
            left_ = <VectorDict> left
            res = left_._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = value - right
            except TypeError:
                return NotImplemented
        else:  # scalar - vec
            right_ = <VectorDict> right
            res = right_._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = left - value
            except TypeError:
                return NotImplemented
        return VectorDict(res)

    def __isub__(VectorDict self, other):
        if isinstance(other, VectorDict):  # vec -= vec
            other_ = <VectorDict> other
            for key in get_union_keys(self, other_):
                self._data[key] = get_value(self, key) - get_value(other_, key)
        else:  # vec -= scalar
            try:
                for key in get_keys(self):
                    self._data[key] -= other
            except TypeError:
                return NotImplemented
        return self

    def __mul__(left, right):
        if isinstance(right, VectorDict):
            left, right = right, left
        left_ = <VectorDict> left
        if isinstance(right, VectorDict):  # vec * vec
            left_, right_ = <VectorDict> right, left_
            res = dict()
            for key in get_union_keys(left_, right_):
                res[key] = get_value(left_, key) * get_value(right_, key)
        else:  # vec * scalar
            res = left_._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = value * right
            except TypeError:
                return NotImplemented
        return VectorDict(res)

    def __imul__(VectorDict self, other):
        if isinstance(other, VectorDict):  # vec *= vec
            other_ = <VectorDict> other
            for key in get_union_keys(self, other_):
                self._data[key] = get_value(self, key) * get_value(other_, key)
        else:  # vec *= scalar
            try:
                for key in get_keys(self):
                    self._data[key] *= other
            except TypeError:
                return NotImplemented
        return self

    def __truediv__(left, right):
        if isinstance(left, VectorDict) and isinstance(right, VectorDict):
            # vec / vec
            left_, right_ = <VectorDict> left, <VectorDict> right
            res = dict()
            for key in get_union_keys(left_, right_):
                res[key] = get_value(left_, key) / get_value(right_, key)
        elif isinstance(left, VectorDict):  # vec / scalar
            left_ = <VectorDict> left
            res = left_._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = value / right
            except TypeError:
                return NotImplemented
        else:  # scalar / vec
            right_ = <VectorDict> right
            res = right_._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = left / value
            except TypeError:
                return NotImplemented
        return VectorDict(res)

    def __itruediv__(VectorDict self, other):
        if isinstance(other, VectorDict):  # vec /= vec
            other_ = <VectorDict> other
            for key in get_union_keys(self, other_):
                self._data[key] = get_value(self, key) / get_value(other_, key)
        else:  # vec /= scalar
            try:
                for key in get_keys(self):
                    self._data[key] /= other
            except TypeError:
                return NotImplemented
        return self

    def __pow__(left, right, modulo):
        if not isinstance(left, VectorDict) or modulo is not None:
            return NotImplemented
        left_ = <VectorDict> left
        res = left_._apply_mask(force_copy=True)
        for key, value in res.items():
            res[key] = value ** right
        return VectorDict(res)

    def __ipow__(VectorDict self, other):
        for key in get_keys(self):
            self._data[key] **= other
        return self

    def __matmul__(left, right):
        if not (isinstance(left, VectorDict) and isinstance(right, VectorDict)):
            return NotImplemented
        left_, right_ = <VectorDict> left, <VectorDict> right
        res = 0
        for key in get_union_keys(left_, right_):
            res += get_value(left_, key) * get_value(right_, key)
        return res

    def __neg__(self):
        # -vec
        res = self._apply_mask(force_copy=True)
        for key, value in res.items():
            res[key] = -value
        return VectorDict(res)

    def __pos__(self):
        # +vec
        return VectorDict(self._apply_mask(force_copy=True))

    def __abs__(self):
        # abs(vec)
        res = self._apply_mask(force_copy=True)
        for key, value in res.items():
            res[key] = abs(value)
        return VectorDict(res)

    # additional utilities

    def abs(self):
        return self.__abs__()

    def min(self):
        if self._lazy_mask:
            return min(value for key, value in self._data.items()
                       if key in self._mask)
        return min(self._data.values())

    def max(self):
        if self._lazy_mask:
            return max(value for key, value in self._data.items()
                       if key in self._mask)
        return max(self._data.values())

    def minimum(self, other):
        if isinstance(other, VectorDict):  # minimum(vec, vec)
            other_ = <VectorDict> other
            res = dict()
            for key in get_union_keys(self, other_):
                res[key] = min(get_value(self, key), get_value(other_, key))
        else:  # minimum(vec, scalar)
            res = self._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = min(value, other)
            except TypeError:
                return NotImplemented
        return VectorDict(res)

    def maximum(self, other):
        if isinstance(other, VectorDict):  # maximum(vec, vec)
            other_ = <VectorDict> other
            res = dict()
            for key in get_union_keys(self, other_):
                res[key] = max(get_value(self, key), get_value(other_, key))
        else:  # maximum(vec, scalar)
            res = self._apply_mask(force_copy=True)
            try:
                for key, value in res.items():
                    res[key] = max(value, other)
            except TypeError:
                return NotImplemented
        return VectorDict(res)
