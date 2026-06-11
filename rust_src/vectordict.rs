// VectorDict — port of river/utils/vectordict.pyx.
//
// A dict-like container with element-wise arithmetic that supports an optional
// `mask` (treat keys outside the mask as missing) and an optional
// `default_factory` (insert + return a factory value when an unmasked key is
// looked up).
//
// Storage is `Py<PyDict>` so keys/values stay arbitrary Python objects, exactly
// matching the Cython interface. Hot inner loops use `pyo3::ffi` directly so
// per-element overhead stays at the same level as Cython's compiled output.

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyTuple};
use pyo3::IntoPyObjectExt;

/// A dictionary-like object that supports vector-like operations.
///
/// Supports addition (+), subtraction (-), multiplication (*) and division (/)
/// with a VectorDict or a scalar.
/// Supports dot product (@) with a VectorDict.
/// A scalar is any object that supports the four arithmetic operations
/// with the dictionary's values.
///
/// If mask is not None, any key which is not contained in mask is said to be
/// masked while other keys are said to be unmasked. If mask is None, any key is
/// said to be unmasked.
///
/// If default_factory is not None, it is called whenever an unmasked missing
/// key is accessed, either externally with __getitem__ or internally as part of
/// an element-wise numeric operation such as addition, and the result is
/// inserted as the value for that key. If a masked key, or an unmasked missing
/// key when default_factory is None, is accessed externally through
/// __getitem__, a KeyError exception is raised, and if it is accessed
/// internally as part of an operation, its value is taken as 0, but is not
/// inserted for that key.
///
/// If copy is True, a copy of data and mask will be made if not None and these
/// arguments will not be modified. If copy is False, references to data and
/// mask will be used if not None. This means that the argument data may be
/// modified, although only on unmasked keys, and that external modifications
/// of data and mask will affect the internal operations.
///
/// Parameters
/// ----------
/// data
///     A VectorDict or dict to initialize key-values from, or None.
/// default_factory
///     A callable returning a scalar, or None.
/// mask
///     A VectorDict or set-like object such that keys not in mask will not be
///     considered in operations and will always result in a KeyError if
///     accessed by __getitem__, or None.
/// copy
///     If data and/or mask are specified, whether to store a copy of the
///     underlying dictionaries or references at initialization.
#[pyclass(name = "VectorDict", module = "river._river_rust.vectordict", subclass)]
pub struct VectorDict {
    data: Py<PyDict>,
    mask: Option<Py<PyAny>>,
    /// `true` whenever a mask was supplied. Operations that *write* (e.g.
    /// `__setitem__`, factory-driven `__getitem__`) must consult the mask to
    /// preserve its invariant, regardless of whether `data` was filtered at
    /// construction time, so they check this flag.
    use_mask: bool,
    /// `true` only when a mask was supplied *and* `copy=False` — meaning the
    /// underlying `data` dict still contains entries outside the mask, so the
    /// mask must be re-applied lazily on every *read* (`__contains__`,
    /// `__delitem__`, `__len__`, iteration, ...). When `copy=True` the data was
    /// already filtered at construction, so these reads can skip the mask
    /// lookup; this flag captures that optimization.
    ///
    /// Invariant: `lazy_mask => use_mask`.
    lazy_mask: bool,
    use_factory: bool,
    default_factory: Option<Py<PyAny>>,
}

// === Internal helpers ============================================================

impl VectorDict {
    fn is_simple(&self) -> bool {
        !self.use_mask && !self.use_factory
    }

    fn data_bound<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.data.bind(py).clone()
    }

    /// `_get`: 0 (or default_factory()) for missing or masked keys.
    fn get_value<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.use_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            if !mask.contains(key)? {
                return 0i64.into_bound_py_any(py);
            }
        }
        let data = self.data.bind(py);
        if let Some(v) = data.get_item(key)? {
            return Ok(v);
        }
        if self.use_factory {
            let factory = self.default_factory.as_ref().unwrap().bind(py);
            let v = factory.call0()?;
            data.set_item(key, &v)?;
            return Ok(v);
        }
        0i64.into_bound_py_any(py)
    }

    /// `_to_dict`: materialize as a (possibly newly-allocated) dict.
    fn to_dict_internal<'py>(
        &self,
        py: Python<'py>,
        force_copy: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let data = self.data.bind(py);
        if !self.lazy_mask {
            return Ok(if force_copy { data.copy()? } else { data.clone() });
        }
        let mask = self.mask.as_ref().unwrap().bind(py);
        let res = PyDict::new(py);
        for (k, v) in data.iter() {
            if mask.contains(&k)? {
                res.set_item(k, v)?;
            }
        }
        Ok(res)
    }

    /// Iterate keys (masked or not) — caller must drive the loop.
    fn iter_keys<'py>(
        &self,
        py: Python<'py>,
        f: &mut dyn FnMut(&Bound<'py, PyAny>) -> PyResult<()>,
    ) -> PyResult<()> {
        let data = self.data.bind(py);
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            for (k, _) in data.iter() {
                if mask.contains(&k)? {
                    f(&k)?;
                }
            }
        } else {
            for (k, _) in data.iter() {
                f(&k)?;
            }
        }
        Ok(())
    }

    /// `get_union_keys` from the Cython code, returned as a fresh Vec for simplicity.
    fn union_keys<'py>(
        &self,
        py: Python<'py>,
        other: &VectorDict,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut out = Vec::new();
        self.iter_keys(py, &mut |k| {
            out.push(k.clone());
            Ok(())
        })?;
        let other_data = other.data.bind(py);
        let self_data = self.data.bind(py);
        for (k, _) in other_data.iter() {
            // skip if in left (after mask)
            let in_left = if self.lazy_mask {
                let mask = self.mask.as_ref().unwrap().bind(py);
                self_data.contains(&k)? && mask.contains(&k)?
            } else {
                self_data.contains(&k)?
            };
            if in_left {
                continue;
            }
            // include only if in right's mask (when right is masked)
            if other.lazy_mask {
                let mask = other.mask.as_ref().unwrap().bind(py);
                if !mask.contains(&k)? {
                    continue;
                }
            }
            out.push(k);
        }
        Ok(out)
    }

    /// `get_intersection_keys` from the Cython code.
    fn intersection_keys<'py>(
        &self,
        py: Python<'py>,
        other: &VectorDict,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let (left, right) = if other.data.bind(py).len() < self.data.bind(py).len() {
            (other, self)
        } else {
            (self, other)
        };
        let left_data = left.data.bind(py);
        let right_data = right.data.bind(py);
        let mut out = Vec::new();
        for (k, _) in left_data.iter() {
            if left.lazy_mask {
                let m = left.mask.as_ref().unwrap().bind(py);
                if !m.contains(&k)? {
                    continue;
                }
            }
            if !right_data.contains(&k)? {
                continue;
            }
            if right.lazy_mask {
                let m = right.mask.as_ref().unwrap().bind(py);
                if !m.contains(&k)? {
                    continue;
                }
            }
            out.push(k);
        }
        Ok(out)
    }

    /// Build a fresh, mask-less, factory-less VectorDict from a dict.
    fn from_dict(data: Py<PyDict>) -> Self {
        VectorDict {
            data,
            mask: None,
            use_mask: false,
            lazy_mask: false,
            use_factory: false,
            default_factory: None,
        }
    }
}

// === Constructor + dict-protocol methods ==========================================

#[pymethods]
impl VectorDict {
    #[new]
    #[pyo3(signature = (data=None, default_factory=None, mask=None, copy=false))]
    fn new<'py>(
        py: Python<'py>,
        data: Option<Bound<'py, PyAny>>,
        default_factory: Option<Bound<'py, PyAny>>,
        mask: Option<Bound<'py, PyAny>>,
        copy: bool,
    ) -> PyResult<Self> {
        let mut mask = mask;
        let data: Bound<'py, PyDict> = match data {
            None => PyDict::new(py),
            // Common case (`VectorDict({...})`) — check for plain dict first to skip
            // the failed `VectorDict` downcast and a borrow.
            Some(d) => {
                if let Ok(d) = d.downcast::<PyDict>() {
                    if copy {
                        if mask.is_none() {
                            d.copy()?
                        } else {
                            let m = mask.take().unwrap();
                            let new_mask =
                                py.import("builtins")?.getattr("set")?.call1((&m,))?;
                            let res = PyDict::new(py);
                            for (k, v) in d.iter() {
                                if new_mask.contains(&k)? {
                                    res.set_item(k, v)?;
                                }
                            }
                            mask = Some(new_mask);
                            res
                        }
                    } else {
                        d.clone()
                    }
                } else if let Ok(vd) = d.downcast::<VectorDict>() {
                    let inner = vd.borrow();
                    if copy {
                        let dict = inner.to_dict_internal(py, true)?;
                        if let Some(m) = mask.take() {
                            let new_mask =
                                py.import("builtins")?.getattr("set")?.call1((m,))?;
                            mask = Some(new_mask);
                        }
                        dict
                    } else {
                        if inner.lazy_mask {
                            let outer_mask_is_inner_mask = match (&mask, &inner.mask) {
                                (Some(om), Some(im)) => om.is(im.bind(py)),
                                _ => false,
                            };
                            if !outer_mask_is_inner_mask {
                                return Err(PyValueError::new_err(
                                    "Cannot mask a masked VectorDict without copy",
                                ));
                            }
                        }
                        inner.data.bind(py).clone()
                    }
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported type for data: {}",
                        d.get_type().name()?
                    )));
                }
            }
        };

        let use_mask = mask.is_some();
        let lazy_mask = use_mask && !copy;
        let use_factory = default_factory.is_some();

        Ok(VectorDict {
            data: data.unbind(),
            mask: mask.map(|m| m.unbind()),
            use_mask,
            lazy_mask,
            use_factory,
            default_factory: default_factory.map(|f| f.unbind()),
        })
    }

    #[pyo3(signature = (mask, copy=false))]
    fn with_mask<'py>(
        &self,
        py: Python<'py>,
        mask: Option<Bound<'py, PyAny>>,
        copy: bool,
    ) -> PyResult<VectorDict> {
        VectorDict::new(
            py,
            Some(self.data.bind(py).clone().into_any()),
            self.default_factory
                .as_ref()
                .map(|f| f.bind(py).clone().into_any()),
            mask,
            copy,
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.to_dict_internal(py, true)
    }

    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        fields: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let values = pyo3::types::PyList::empty(py);
        for f in fields.try_iter()? {
            let f = f?;
            values.append(self.get_value(py, &f)?)?;
        }
        np.getattr("array")?.call1((values,))
    }

    // ---- pass-through dict methods ----

    fn __contains__<'py>(&self, py: Python<'py>, key: Bound<'py, PyAny>) -> PyResult<bool> {
        let in_data = self.data.bind(py).contains(&key)?;
        if !self.lazy_mask {
            return Ok(in_data);
        }
        let mask = self.mask.as_ref().unwrap().bind(py);
        Ok(in_data && mask.contains(&key)?)
    }

    fn __delitem__<'py>(&self, py: Python<'py>, key: Bound<'py, PyAny>) -> PyResult<()> {
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            if !mask.contains(&key)? {
                return Err(PyKeyError::new_err(key.unbind()));
            }
        }
        self.data.bind(py).del_item(key)
    }

    fn __format__<'py>(
        &self,
        py: Python<'py>,
        format_spec: Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let dict = self.to_dict_internal(py, false)?;
        let s = dict
            .as_any()
            .call_method1("__format__", (format_spec,))?;
        Ok(s.unbind())
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.use_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            if !mask.contains(&key)? {
                return Err(PyKeyError::new_err(key.unbind()));
            }
        }
        let data = self.data.bind(py);
        if let Some(v) = data.get_item(&key)? {
            return Ok(v);
        }
        if self.use_factory {
            let factory = self.default_factory.as_ref().unwrap().bind(py);
            let v = factory.call0()?;
            data.set_item(&key, &v)?;
            return Ok(v);
        }
        Err(PyKeyError::new_err(key.unbind()))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = self.to_dict_internal(py, false)?;
        dict.as_any().call_method0("__iter__")
    }

    fn __len__<'py>(&self, py: Python<'py>) -> PyResult<usize> {
        let data = self.data.bind(py);
        if self.lazy_mask {
            let keys = data.keys();
            let mask = self.mask.as_ref().unwrap().bind(py);
            // len(self._data.keys() - self._mask): set difference
            let diff = keys.as_any().sub(mask)?;
            return diff.len();
        }
        Ok(data.len())
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        let dict = self.to_dict_internal(py, false)?;
        dict.as_any().repr().map(|s| s.to_string())
    }

    fn __setitem__<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        if self.use_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            if !mask.contains(&key)? {
                return Err(PyKeyError::new_err(key.unbind()));
            }
        }
        self.data.bind(py).set_item(key, value)
    }

    fn __str__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        let dict = self.to_dict_internal(py, false)?;
        dict.as_any().str().map(|s| s.to_string())
    }

    fn clear<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let data = self.data.bind(py);
        if self.lazy_mask {
            // Keep masked-in items
            let mask = self.mask.as_ref().unwrap().bind(py);
            let keep = PyDict::new(py);
            for (k, v) in data.iter() {
                if mask.contains(&k)? {
                    keep.set_item(k, v)?;
                }
            }
            data.clear();
            data.update(keep.as_mapping())?;
        } else {
            data.clear();
        }
        Ok(())
    }

    #[pyo3(signature = (key, *args, **kwargs))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
        args: Bound<'py, PyTuple>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            if !mask.contains(&key)? {
                let empty = PyDict::new(py);
                let mut all_args: Vec<Bound<'py, PyAny>> = vec![key];
                for a in args.iter() {
                    all_args.push(a);
                }
                let tup = PyTuple::new(py, &all_args)?;
                return empty.as_any().call_method("get", tup, kwargs.as_ref());
            }
        }
        let data = self.data.bind(py);
        let mut all_args: Vec<Bound<'py, PyAny>> = vec![key];
        for a in args.iter() {
            all_args.push(a);
        }
        let tup = PyTuple::new(py, &all_args)?;
        data.as_any().call_method("get", tup, kwargs.as_ref())
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = self.to_dict_internal(py, false)?;
        dict.as_any().call_method0("items")
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = self.to_dict_internal(py, false)?;
        dict.as_any().call_method0("keys")
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn pop<'py>(
        &self,
        py: Python<'py>,
        args: Bound<'py, PyTuple>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.data
            .bind(py)
            .as_any()
            .call_method("pop", args, kwargs.as_ref())
    }

    fn popitem<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = self.data.bind(py);
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            let mut keep: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
            loop {
                let pair = data.as_any().call_method0("popitem")?;
                let tup = pair.downcast::<PyTuple>()?;
                let k = tup.get_item(0)?;
                let v = tup.get_item(1)?;
                if mask.contains(&k)? {
                    keep.push((k, v));
                } else {
                    // restore the unmasked items
                    for (kk, vv) in keep.into_iter() {
                        data.set_item(kk, vv)?;
                    }
                    return Ok(pair);
                }
            }
        }
        data.as_any().call_method0("popitem")
    }

    #[pyo3(signature = (key, *args, **kwargs))]
    fn setdefault<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
        args: Bound<'py, PyTuple>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            if !mask.contains(&key)? {
                let empty = PyDict::new(py);
                let mut all_args: Vec<Bound<'py, PyAny>> = vec![key];
                for a in args.iter() {
                    all_args.push(a);
                }
                let tup = PyTuple::new(py, &all_args)?;
                return empty
                    .as_any()
                    .call_method("setdefault", tup, kwargs.as_ref());
            }
        }
        let data = self.data.bind(py);
        let mut all_args: Vec<Bound<'py, PyAny>> = vec![key];
        for a in args.iter() {
            all_args.push(a);
        }
        let tup = PyTuple::new(py, &all_args)?;
        data.as_any()
            .call_method("setdefault", tup, kwargs.as_ref())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn update<'py>(
        &self,
        py: Python<'py>,
        args: Bound<'py, PyTuple>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        let data = self.data.bind(py);
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            // 1) snapshot masked-in current items
            let keep1 = PyDict::new(py);
            for (k, v) in data.iter() {
                if mask.contains(&k)? {
                    keep1.set_item(k, v)?;
                }
            }
            // 2) apply update
            data.as_any().call_method("update", &args, kwargs.as_ref())?;
            // 3) snapshot masked-out items (newly inserted ones we shouldn't keep visible —
            // but the original Cython actually keeps them in _data; we mirror that)
            let keep2 = PyDict::new(py);
            for (k, v) in data.iter() {
                if !mask.contains(&k)? {
                    keep2.set_item(k, v)?;
                }
            }
            data.clear();
            data.update(keep1.as_mapping())?;
            data.update(keep2.as_mapping())?;
        } else {
            data.as_any().call_method("update", args, kwargs.as_ref())?;
        }
        Ok(())
    }

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = self.to_dict_internal(py, false)?;
        dict.as_any().call_method0("values")
    }

    // ---- comparison ----

    fn __eq__<'py>(&self, py: Python<'py>, right: Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let left_dict = self.to_dict_internal(py, false)?;
        if let Ok(other_vd) = right.downcast::<VectorDict>() {
            let inner = other_vd.borrow();
            let other_dict = inner.to_dict_internal(py, false)?;
            return Ok(left_dict.as_any().eq(other_dict)?.into_py_any(py)?);
        }
        if right.downcast::<PyDict>().is_ok() {
            return Ok(left_dict.as_any().eq(&right)?.into_py_any(py)?);
        }
        Ok(py.NotImplemented())
    }
    // ---- arithmetic: addition ----

    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        binop_add(py, self, &other, /*reverse=*/ false)
    }

    fn __radd__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        // Addition is commutative.
        binop_add(py, self, &other, false)
    }

    fn __iadd__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<()> {
        iadd(py, self, &other)
    }

    // ---- arithmetic: subtraction ----

    fn __sub__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        binop_sub(py, self, &other, /*reverse=*/ false)
    }

    fn __rsub__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        // other - self
        binop_sub(py, self, &other, /*reverse=*/ true)
    }

    fn __isub__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<()> {
        isub(py, self, &other)
    }

    // ---- arithmetic: multiplication ----

    fn __mul__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        binop_mul(py, self, &other)
    }

    fn __rmul__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        binop_mul(py, self, &other)
    }

    fn __imul__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<()> {
        imul(py, self, &other)
    }

    // ---- arithmetic: true division ----

    fn __truediv__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        binop_div(py, self, &other, /*reverse=*/ false)
    }

    fn __rtruediv__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        binop_div(py, self, &other, /*reverse=*/ true)
    }

    fn __itruediv__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<()> {
        idiv(py, self, &other)
    }

    // ---- arithmetic: power ----

    fn __pow__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
        modulo: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<VectorDict>> {
        let _ = modulo; // ignored: VectorDict ** k doesn't take a modulus
        let res = self.to_dict_internal(py, true)?;
        unsafe {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut val: *mut ffi::PyObject = std::ptr::null_mut();
            let other_ptr = other.as_ptr();
            let none_ptr = ffi::Py_None();
            while ffi::PyDict_Next(res.as_ptr(), &mut pos, &mut key, &mut val) != 0 {
                let new_val = ffi_check(py, ffi::PyNumber_Power(val, other_ptr, none_ptr))?;
                let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
                ffi::Py_DECREF(new_val);
                ffi_status(py, r)?;
            }
        }
        Py::new(py, VectorDict::from_dict(res.unbind()))
    }

    fn __ipow__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
        modulo: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let _ = modulo;
        let data = self.data.bind(py);
        let mask_ptr = if self.lazy_mask {
            Some(self.mask.as_ref().unwrap().bind(py).as_ptr())
        } else {
            None
        };
        unsafe {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut val: *mut ffi::PyObject = std::ptr::null_mut();
            let other_ptr = other.as_ptr();
            let none_ptr = ffi::Py_None();
            while ffi::PyDict_Next(data.as_ptr(), &mut pos, &mut key, &mut val) != 0 {
                if let Some(mp) = mask_ptr {
                    let in_mask = ffi::PySequence_Contains(mp, key);
                    if in_mask < 0 {
                        return Err(PyErr::fetch(py));
                    }
                    if in_mask == 0 {
                        continue;
                    }
                }
                let new_val = ffi_check(py, ffi::PyNumber_Power(val, other_ptr, none_ptr))?;
                let r = ffi::PyDict_SetItem(data.as_ptr(), key, new_val);
                ffi::Py_DECREF(new_val);
                ffi_status(py, r)?;
            }
        }
        Ok(())
    }

    // ---- arithmetic: matmul (dot product) ----

    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        right: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let other_vd = match right.downcast::<VectorDict>() {
            Ok(o) => o,
            Err(_) => return Ok(py.NotImplemented().bind(py).clone()),
        };
        let right_inner = other_vd.borrow();
        matmul(py, self, &right_inner)
    }

    // ---- unary operators ----

    fn __neg__<'py>(&self, py: Python<'py>) -> PyResult<Py<VectorDict>> {
        let res = unary_into_new(py, self, ffi::PyNumber_Negative)?;
        Py::new(py, VectorDict::from_dict(res))
    }

    fn __pos__<'py>(&self, py: Python<'py>) -> PyResult<Py<VectorDict>> {
        let res = self.to_dict_internal(py, true)?;
        Py::new(py, VectorDict::from_dict(res.unbind()))
    }

    fn __abs__<'py>(&self, py: Python<'py>) -> PyResult<Py<VectorDict>> {
        let res = unary_into_new(py, self, ffi::PyNumber_Absolute)?;
        Py::new(py, VectorDict::from_dict(res))
    }

    // ---- additional utilities ----

    fn iadd_scaled<'py>(
        slf: Bound<'py, VectorDict>,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
        scalar: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        if let Ok(other_vd) = other.downcast::<VectorDict>() {
            let inner = slf.borrow();
            let other_inner = other_vd.borrow();
            if inner.is_simple() && other_inner.is_simple() {
                let self_data = inner.data.bind(py);
                let other_data = other_inner.data.bind(py);
                unsafe {
                    let mut pos: ffi::Py_ssize_t = 0;
                    let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                    let mut other_val: *mut ffi::PyObject = std::ptr::null_mut();
                    while ffi::PyDict_Next(other_data.as_ptr(), &mut pos, &mut key, &mut other_val)
                        != 0
                    {
                        let scaled =
                            ffi_check(py, ffi::PyNumber_Multiply(scalar.as_ptr(), other_val))?;
                        let self_val = dict_get_or_null(self_data.as_ptr(), key);
                        let new_val = if !self_val.is_null() {
                            let nv = ffi::PyNumber_Add(self_val, scaled);
                            ffi::Py_DECREF(scaled);
                            ffi_check(py, nv)?
                        } else {
                            // Missing key — the new value is just `scaled`. Ownership of the
                            // single ref now belongs to `new_val`; the SetItem below INCREFs and
                            // we DECREF after.
                            scaled
                        };
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, new_val);
                        ffi::Py_DECREF(new_val);
                        ffi_status(py, r)?;
                    }
                }
                drop(other_inner);
                drop(inner);
                return Ok(slf.unbind());
            }
        }
        // Fallback: self += scalar * other
        let py_scaled = scalar.mul(&other)?;
        let inner_borrow = slf.borrow();
        iadd(py, &*inner_borrow, &py_scaled)?;
        drop(inner_borrow);
        Ok(slf.unbind())
    }

    fn isub_scaled<'py>(
        slf: Bound<'py, VectorDict>,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
        scalar: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        if let Ok(other_vd) = other.downcast::<VectorDict>() {
            let inner = slf.borrow();
            let other_inner = other_vd.borrow();
            if inner.is_simple() && other_inner.is_simple() {
                let self_data = inner.data.bind(py);
                let other_data = other_inner.data.bind(py);
                unsafe {
                    let mut pos: ffi::Py_ssize_t = 0;
                    let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                    let mut other_val: *mut ffi::PyObject = std::ptr::null_mut();
                    while ffi::PyDict_Next(other_data.as_ptr(), &mut pos, &mut key, &mut other_val)
                        != 0
                    {
                        let scaled = ffi_check(
                            py,
                            ffi::PyNumber_Multiply(scalar.as_ptr(), other_val),
                        )?;
                        let self_val = dict_get_or_null(self_data.as_ptr(), key);
                        let new_val = if !self_val.is_null() {
                            let nv = ffi::PyNumber_Subtract(self_val, scaled);
                            ffi::Py_DECREF(scaled);
                            ffi_check(py, nv)?
                        } else {
                            // missing key: result is -scaled
                            let nv = ffi::PyNumber_Negative(scaled);
                            ffi::Py_DECREF(scaled);
                            ffi_check(py, nv)?
                        };
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, new_val);
                        ffi::Py_DECREF(new_val);
                        ffi_status(py, r)?;
                    }
                }
                drop(other_inner);
                drop(inner);
                return Ok(slf.unbind());
            }
        }
        // Fallback: self -= scalar * other
        let py_scaled = scalar.mul(&other)?;
        let inner_borrow = slf.borrow();
        isub(py, &*inner_borrow, &py_scaled)?;
        drop(inner_borrow);
        Ok(slf.unbind())
    }

    fn abs<'py>(&self, py: Python<'py>) -> PyResult<Py<VectorDict>> {
        self.__abs__(py)
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = self.data.bind(py);
        let builtins = py.import("builtins")?;
        let min_fn = builtins.getattr("min")?;
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            let vals = pyo3::types::PyList::empty(py);
            for (k, v) in data.iter() {
                if mask.contains(&k)? {
                    vals.append(v)?;
                }
            }
            return min_fn.call1((vals,));
        }
        min_fn.call1((data.values(),))
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = self.data.bind(py);
        let builtins = py.import("builtins")?;
        let max_fn = builtins.getattr("max")?;
        if self.lazy_mask {
            let mask = self.mask.as_ref().unwrap().bind(py);
            let vals = pyo3::types::PyList::empty(py);
            for (k, v) in data.iter() {
                if mask.contains(&k)? {
                    vals.append(v)?;
                }
            }
            return max_fn.call1((vals,));
        }
        max_fn.call1((data.values(),))
    }

    fn minimum<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        per_element_min_max(py, self, &other, /*want_min=*/ true)
    }

    fn maximum<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        per_element_min_max(py, self, &other, /*want_min=*/ false)
    }

    fn __reduce__<'py>(slf: Bound<'py, VectorDict>) -> PyResult<Py<PyAny>> {
        // (constructor, (data, default_factory, mask, copy=True))
        //
        // We pull the concrete type off `slf` (rather than hard-coding
        // `VectorDict`) so subclasses unpickle as themselves.
        let py = slf.py();
        let cls = slf.get_type();
        let inner = slf.borrow();
        let data = inner.data.bind(py).clone().into_any();
        let factory = match &inner.default_factory {
            Some(f) => f.bind(py).clone(),
            None => py.None().into_bound(py),
        };
        let mask = match &inner.mask {
            Some(m) => m.bind(py).clone(),
            None => py.None().into_bound(py),
        };
        let args = PyTuple::new(py, [data, factory, mask, true.into_bound_py_any(py)?])?;
        let result = PyTuple::new(py, [cls.into_any(), args.into_any()])?;
        Ok(result.unbind().into_any())
    }

    fn __deepcopy__<'py>(
        &self,
        py: Python<'py>,
        _memo: Bound<'py, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        let copy_mod = py.import("copy")?;
        let deepcopy = copy_mod.getattr("deepcopy")?;
        let data_copy = deepcopy.call1((self.data.bind(py),))?;
        let factory = self.default_factory.as_ref().map(|f| f.bind(py).clone());
        let mask = self.mask.as_ref().map(|m| {
            let m = m.bind(py).clone();
            deepcopy.call1((m,))
        });
        let mask = match mask {
            Some(r) => Some(r?),
            None => None,
        };
        let new = VectorDict::new(py, Some(data_copy), factory, mask, true)?;
        Py::new(py, new)
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Py<VectorDict>> {
        let data_copy = self.data.bind(py).copy()?;
        Py::new(
            py,
            VectorDict {
                data: data_copy.unbind(),
                mask: self.mask.as_ref().map(|m| m.clone_ref(py)),
                use_mask: self.use_mask,
                lazy_mask: self.lazy_mask,
                use_factory: self.use_factory,
                default_factory: self.default_factory.as_ref().map(|f| f.clone_ref(py)),
            },
        )
    }

    fn euclidean_distance<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<f64> {
        let other_dict_ptr = if let Ok(vd) = other.downcast::<VectorDict>() {
            vd.borrow().data.clone_ref(py)
        } else if let Ok(d) = other.downcast::<PyDict>() {
            d.clone().unbind()
        } else {
            return Err(PyTypeError::new_err(format!(
                "Unsupported type: {}",
                other.get_type().name()?
            )));
        };
        let other_d = other_dict_ptr.bind(py);
        euclidean_distance_dict_dict(py, self.data.bind(py), other_d)
    }
}

// === Arithmetic operators =========================================================
//
// The `__add__`, `__sub__`, etc. variants dispatch on whether `right` is a
// VectorDict or a scalar. PyO3 routes `vec + X` to `__add__` and `X + vec` to
// `__radd__`. For commutative ops (+, *), `__radd__` uses the same logic with
// args swapped; for non-commutative ops (-, /), it differs.
//
// Per-element loops use `pyo3::ffi` directly so PyO3's safe wrappers don't add
// a Bound creation per iteration on the dict-iteration / set-item hot path.
// All `unsafe` blocks share the same shape: borrow the GIL-bound dict pointers,
// call CPython's number protocol, then check for the standard
// "null result + PyErr_Occurred" failure mode before returning.

unsafe fn ffi_check(py: Python<'_>, ptr: *mut ffi::PyObject) -> PyResult<*mut ffi::PyObject> {
    if ptr.is_null() {
        Err(PyErr::fetch(py))
    } else {
        Ok(ptr)
    }
}

// `PyDict_GetItem`: dict lookup that returns NULL on miss without setting an
// error. Safe to call for keys obtained from `PyDict_Next`, which were already
// successfully hashed. Faster than the error-propagating variant because it
// skips a `PyErr_Occurred` check on every miss.
#[inline]
unsafe fn dict_get_or_null(
    dict: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    ffi::PyDict_GetItem(dict, key)
}

unsafe fn ffi_status(py: Python<'_>, code: i32) -> PyResult<()> {
    if code != 0 {
        Err(PyErr::fetch(py))
    } else {
        Ok(())
    }
}

type BinOp = unsafe extern "C" fn(*mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject;

/// Apply `res[k] = a[k] OP scalar` (or `scalar OP a[k]` when `REV=true`) for
/// every key in `a`. `REV` is a const generic so the operand-swap branch is
/// monomorphized away at compile time — both directions get the same codegen
/// as the previous two specialized `dict_op_scalar` / `dict_op_scalar_rev`
/// functions, but the body lives in one place.
unsafe fn dict_op_scalar<const REV: bool>(
    py: Python<'_>,
    a: *mut ffi::PyObject,
    scalar: *mut ffi::PyObject,
    op: BinOp,
    res: *mut ffi::PyObject,
) -> PyResult<()> {
    let mut pos: ffi::Py_ssize_t = 0;
    let mut key: *mut ffi::PyObject = std::ptr::null_mut();
    let mut val: *mut ffi::PyObject = std::ptr::null_mut();
    while ffi::PyDict_Next(a, &mut pos, &mut key, &mut val) != 0 {
        let raw = if REV { op(scalar, val) } else { op(val, scalar) };
        let new_val = ffi_check(py, raw)?;
        let r = ffi::PyDict_SetItem(res, key, new_val);
        ffi::Py_DECREF(new_val);
        ffi_status(py, r)?;
    }
    Ok(())
}

/// In-place `a[k] = a[k] OP scalar` over every key in `a` (or, when `mask` is
/// provided, every masked-in key).
unsafe fn dict_iop_scalar(
    py: Python<'_>,
    a: *mut ffi::PyObject,
    scalar: *mut ffi::PyObject,
    op: BinOp,
    mask: Option<*mut ffi::PyObject>,
) -> PyResult<()> {
    let mut pos: ffi::Py_ssize_t = 0;
    let mut key: *mut ffi::PyObject = std::ptr::null_mut();
    let mut val: *mut ffi::PyObject = std::ptr::null_mut();
    while ffi::PyDict_Next(a, &mut pos, &mut key, &mut val) != 0 {
        if let Some(m) = mask {
            let in_mask = ffi::PySequence_Contains(m, key);
            if in_mask < 0 {
                return Err(PyErr::fetch(py));
            }
            if in_mask == 0 {
                continue;
            }
        }
        let new_val = ffi_check(py, op(val, scalar))?;
        let r = ffi::PyDict_SetItem(a, key, new_val);
        ffi::Py_DECREF(new_val);
        ffi_status(py, r)?;
    }
    Ok(())
}


type UnaryOp = unsafe extern "C" fn(*mut ffi::PyObject) -> *mut ffi::PyObject;

// Build a fresh PyDict. The previous implementation called the private
// CPython _PyDict_NewPresized to preallocate `n` slots, but pyo3 0.28
// stopped re-exporting that symbol. PyDict::new() falls back to the public
// PyDict_New, which doesn't preallocate; the amortised rehash cost is
// O(n) and dominated by the per-key set_item work the caller does next.
unsafe fn fresh_dict_for<'py>(py: Python<'py>, _n: usize) -> PyResult<Bound<'py, PyDict>> {
    Ok(PyDict::new(py))
}

// Single-pass unary op (`__neg__`, `__abs__`): read source, write fresh dict.
#[inline]
fn unary_into_new<'py>(
    py: Python<'py>,
    left: &VectorDict,
    op: UnaryOp,
) -> PyResult<Py<PyDict>> {
    let n = left.data.bind(py).len();
    let res = unsafe { fresh_dict_for(py, n)? };
    let src = left.data.bind(py);
    let mask_ptr = if left.lazy_mask {
        Some(left.mask.as_ref().unwrap().bind(py).as_ptr())
    } else {
        None
    };
    unsafe {
        let mut pos: ffi::Py_ssize_t = 0;
        let mut key: *mut ffi::PyObject = std::ptr::null_mut();
        let mut val: *mut ffi::PyObject = std::ptr::null_mut();
        while ffi::PyDict_Next(src.as_ptr(), &mut pos, &mut key, &mut val) != 0 {
            if let Some(m) = mask_ptr {
                let in_mask = ffi::PySequence_Contains(m, key);
                if in_mask < 0 {
                    return Err(PyErr::fetch(py));
                }
                if in_mask == 0 {
                    continue;
                }
            }
            let new_val = ffi_check(py, op(val))?;
            let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
            ffi::Py_DECREF(new_val);
            ffi_status(py, r)?;
        }
    }
    Ok(res.unbind())
}

// Convenience: read self's data (filtered by mask if any), apply `value OP scalar`
// (or `scalar OP value` when `REV=true`), write into a fresh PyDict. Single-pass
// — avoids the dict.copy()+overwrite that the Cython baseline does. The result
// dict is pre-sized via `_PyDict_NewPresized` so large outputs don't pay the
// resize/rehash cost. Direction is a const generic so monomorphization gives
// each direction its own specialized loop body.
fn scalar_op_into_new<'py, const REV: bool>(
    py: Python<'py>,
    left: &VectorDict,
    scalar: *mut ffi::PyObject,
    op: BinOp,
) -> PyResult<Py<PyDict>> {
    let src = left.data.bind(py);
    let n = src.len();
    let res = unsafe { fresh_dict_for(py, n)? };
    if !left.lazy_mask {
        unsafe {
            dict_op_scalar::<REV>(py, src.as_ptr(), scalar, op, res.as_ptr())?;
        }
    } else {
        let mask = left.mask.as_ref().unwrap().bind(py);
        unsafe {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut val: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(src.as_ptr(), &mut pos, &mut key, &mut val) != 0 {
                let in_mask = ffi::PySequence_Contains(mask.as_ptr(), key);
                if in_mask < 0 {
                    return Err(PyErr::fetch(py));
                }
                if in_mask == 0 {
                    continue;
                }
                let raw = if REV { op(scalar, val) } else { op(val, scalar) };
                let new_val = ffi_check(py, raw)?;
                let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
                ffi::Py_DECREF(new_val);
                ffi_status(py, r)?;
            }
        }
    }
    Ok(res.unbind())
}

// === Free helpers for arithmetic operators =======================================

#[inline]
fn binop_add<'py>(
    py: Python<'py>,
    left: &VectorDict,
    right: &Bound<'py, PyAny>,
    _reverse: bool,
) -> PyResult<Py<VectorDict>> {
    if let Ok(right_vd) = right.downcast::<VectorDict>() {
        let right_inner = right_vd.borrow();
        return Py::new(
            py,
            VectorDict::from_dict(add_dict_dict(py, left, &right_inner)?),
        );
    }
    // vec + scalar — single-pass: read source, write fresh result. Saves the
    // dict.copy()+overwrite double-pass that the Cython does.
    let res = scalar_op_into_new::<false>(py, left, right.as_ptr(), ffi::PyNumber_Add)?;
    Py::new(py, VectorDict::from_dict(res))
}

#[inline]
fn add_dict_dict<'py>(
    py: Python<'py>,
    left: &VectorDict,
    right: &VectorDict,
) -> PyResult<Py<PyDict>> {
    if left.is_simple() && right.is_simple() {
        // Fast path: copy left, then merge right with `+` on collisions
        let res = left.data.bind(py).copy()?;
        let right_data = right.data.bind(py);
        unsafe {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut right_val: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(right_data.as_ptr(), &mut pos, &mut key, &mut right_val) != 0 {
                let left_val = dict_get_or_null(res.as_ptr(), key);
                if !left_val.is_null() {
                    let new_val = ffi_check(py, ffi::PyNumber_Add(left_val, right_val))?;
                    let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
                    ffi::Py_DECREF(new_val);
                    ffi_status(py, r)?;
                } else {
                    // key only in right — copy the value
                    let r = ffi::PyDict_SetItem(res.as_ptr(), key, right_val);
                    ffi_status(py, r)?;
                }
            }
        }
        return Ok(res.unbind());
    }
    // Slow path with mask/factory: iterate union keys via _get
    let res = PyDict::new(py);
    for k in left.union_keys(py, right)? {
        let lv = left.get_value(py, &k)?;
        let rv = right.get_value(py, &k)?;
        res.set_item(k, lv.add(rv)?)?;
    }
    Ok(res.unbind())
}

fn iadd<'py>(py: Python<'py>, inner: &VectorDict, other: &Bound<'py, PyAny>) -> PyResult<()> {
    if let Ok(other_vd) = other.downcast::<VectorDict>() {
        let other_inner = other_vd.borrow();
        let self_data = inner.data.bind(py);
        let other_data = other_inner.data.bind(py);
        if inner.is_simple() && other_inner.is_simple() {
            unsafe {
                let mut pos: ffi::Py_ssize_t = 0;
                let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                let mut other_val: *mut ffi::PyObject = std::ptr::null_mut();
                while ffi::PyDict_Next(other_data.as_ptr(), &mut pos, &mut key, &mut other_val)
                    != 0
                {
                    let self_val = dict_get_or_null(self_data.as_ptr(), key);
                    if !self_val.is_null() {
                        let new_val =
                            ffi_check(py, ffi::PyNumber_Add(self_val, other_val))?;
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, new_val);
                        ffi::Py_DECREF(new_val);
                        ffi_status(py, r)?;
                    } else {
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, other_val);
                        ffi_status(py, r)?;
                    }
                }
            }
            return Ok(());
        }
        // mask/factory: walk union keys
        for k in inner.union_keys(py, &other_inner)? {
            let lv = inner.get_value(py, &k)?;
            let rv = other_inner.get_value(py, &k)?;
            inner.data.bind(py).set_item(k, lv.add(rv)?)?;
        }
        return Ok(());
    }
    // vec += scalar
    let mask_ptr = if inner.lazy_mask {
        Some(inner.mask.as_ref().unwrap().bind(py).as_ptr())
    } else {
        None
    };
    unsafe {
        dict_iop_scalar(
            py,
            inner.data.bind(py).as_ptr(),
            other.as_ptr(),
            ffi::PyNumber_InPlaceAdd,
            mask_ptr,
        )?;
    }
    Ok(())
}

#[inline]
fn binop_sub<'py>(
    py: Python<'py>,
    left: &VectorDict,
    other: &Bound<'py, PyAny>,
    reverse: bool,
) -> PyResult<Py<VectorDict>> {
    if let Ok(right_vd) = other.downcast::<VectorDict>() {
        // Both VectorDicts. `reverse` only happens if the LHS isn't a VectorDict.
        let right_inner = right_vd.borrow();
        let (l, r) = if reverse {
            (&*right_inner, left)
        } else {
            (left, &*right_inner)
        };
        return Py::new(py, VectorDict::from_dict(sub_dict_dict(py, l, r)?));
    }
    let res = if reverse {
        scalar_op_into_new::<true>(py, left, other.as_ptr(), ffi::PyNumber_Subtract)?
    } else {
        scalar_op_into_new::<false>(py, left, other.as_ptr(), ffi::PyNumber_Subtract)?
    };
    Py::new(py, VectorDict::from_dict(res))
}

#[inline]
fn sub_dict_dict<'py>(
    py: Python<'py>,
    left: &VectorDict,
    right: &VectorDict,
) -> PyResult<Py<PyDict>> {
    if left.is_simple() && right.is_simple() {
        // Copy left; iterate right
        let res = left.data.bind(py).copy()?;
        let right_data = right.data.bind(py);
        unsafe {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut right_val: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(right_data.as_ptr(), &mut pos, &mut key, &mut right_val) != 0 {
                let left_val = dict_get_or_null(res.as_ptr(), key);
                let new_val = if !left_val.is_null() {
                    ffi_check(py, ffi::PyNumber_Subtract(left_val, right_val))?
                } else {
                    ffi_check(py, ffi::PyNumber_Negative(right_val))?
                };
                let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
                ffi::Py_DECREF(new_val);
                ffi_status(py, r)?;
            }
        }
        return Ok(res.unbind());
    }
    let res = PyDict::new(py);
    for k in left.union_keys(py, right)? {
        let lv = left.get_value(py, &k)?;
        let rv = right.get_value(py, &k)?;
        res.set_item(k, lv.sub(rv)?)?;
    }
    Ok(res.unbind())
}

fn isub<'py>(py: Python<'py>, inner: &VectorDict, other: &Bound<'py, PyAny>) -> PyResult<()> {
    if let Ok(other_vd) = other.downcast::<VectorDict>() {
        let other_inner = other_vd.borrow();
        if inner.is_simple() && other_inner.is_simple() {
            let self_data = inner.data.bind(py);
            let other_data = other_inner.data.bind(py);
            unsafe {
                let mut pos: ffi::Py_ssize_t = 0;
                let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                let mut other_val: *mut ffi::PyObject = std::ptr::null_mut();
                while ffi::PyDict_Next(other_data.as_ptr(), &mut pos, &mut key, &mut other_val)
                    != 0
                {
                    let self_val = dict_get_or_null(self_data.as_ptr(), key);
                    let new_val = if !self_val.is_null() {
                        ffi_check(py, ffi::PyNumber_Subtract(self_val, other_val))?
                    } else {
                        ffi_check(py, ffi::PyNumber_Negative(other_val))?
                    };
                    let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, new_val);
                    ffi::Py_DECREF(new_val);
                    ffi_status(py, r)?;
                }
            }
            return Ok(());
        }
        for k in inner.union_keys(py, &other_inner)? {
            let lv = inner.get_value(py, &k)?;
            let rv = other_inner.get_value(py, &k)?;
            inner.data.bind(py).set_item(k, lv.sub(rv)?)?;
        }
        return Ok(());
    }
    let mask_ptr = if inner.lazy_mask {
        Some(inner.mask.as_ref().unwrap().bind(py).as_ptr())
    } else {
        None
    };
    unsafe {
        dict_iop_scalar(
            py,
            inner.data.bind(py).as_ptr(),
            other.as_ptr(),
            ffi::PyNumber_InPlaceSubtract,
            mask_ptr,
        )?;
    }
    Ok(())
}

#[inline]
fn binop_mul<'py>(
    py: Python<'py>,
    left: &VectorDict,
    other: &Bound<'py, PyAny>,
) -> PyResult<Py<VectorDict>> {
    if let Ok(other_vd) = other.downcast::<VectorDict>() {
        let other_inner = other_vd.borrow();
        return Py::new(
            py,
            VectorDict::from_dict(mul_dict_dict(py, left, &other_inner)?),
        );
    }
    let res = scalar_op_into_new::<false>(py, left, other.as_ptr(), ffi::PyNumber_Multiply)?;
    Py::new(py, VectorDict::from_dict(res))
}

#[inline]
fn mul_dict_dict<'py>(
    py: Python<'py>,
    left: &VectorDict,
    right: &VectorDict,
) -> PyResult<Py<PyDict>> {
    if left.is_simple() && right.is_simple() {
        let res = PyDict::new(py);
        let left_data = left.data.bind(py);
        let right_data = right.data.bind(py);
        let zero = 0i64.into_bound_py_any(py)?;
        unsafe {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut left_val: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(left_data.as_ptr(), &mut pos, &mut key, &mut left_val) != 0 {
                let right_val = dict_get_or_null(right_data.as_ptr(), key);
                if !right_val.is_null() {
                    let new_val = ffi_check(py, ffi::PyNumber_Multiply(left_val, right_val))?;
                    let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
                    ffi::Py_DECREF(new_val);
                    ffi_status(py, r)?;
                } else {
                    let r = ffi::PyDict_SetItem(res.as_ptr(), key, zero.as_ptr());
                    ffi_status(py, r)?;
                }
            }
            // Fill in keys present only in right with zero
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut _v: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(right_data.as_ptr(), &mut pos, &mut key, &mut _v) != 0 {
                let in_res = ffi::PyDict_Contains(res.as_ptr(), key);
                if in_res < 0 {
                    return Err(PyErr::fetch(py));
                }
                if in_res == 0 {
                    let r = ffi::PyDict_SetItem(res.as_ptr(), key, zero.as_ptr());
                    ffi_status(py, r)?;
                }
            }
        }
        return Ok(res.unbind());
    }
    let res = PyDict::new(py);
    for k in left.union_keys(py, right)? {
        let lv = left.get_value(py, &k)?;
        let rv = right.get_value(py, &k)?;
        res.set_item(k, lv.mul(rv)?)?;
    }
    Ok(res.unbind())
}

fn imul<'py>(py: Python<'py>, inner: &VectorDict, other: &Bound<'py, PyAny>) -> PyResult<()> {
    if let Ok(other_vd) = other.downcast::<VectorDict>() {
        let other_inner = other_vd.borrow();
        if inner.is_simple() && other_inner.is_simple() {
            let self_data = inner.data.bind(py);
            let other_data = other_inner.data.bind(py);
            let zero = 0i64.into_bound_py_any(py)?;
            unsafe {
                // Walk self's keys
                let mut pos: ffi::Py_ssize_t = 0;
                let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                let mut self_val: *mut ffi::PyObject = std::ptr::null_mut();
                while ffi::PyDict_Next(self_data.as_ptr(), &mut pos, &mut key, &mut self_val) != 0
                {
                    let other_val = dict_get_or_null(other_data.as_ptr(), key);
                    if !other_val.is_null() {
                        let new_val =
                            ffi_check(py, ffi::PyNumber_Multiply(self_val, other_val))?;
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, new_val);
                        ffi::Py_DECREF(new_val);
                        ffi_status(py, r)?;
                    } else {
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, zero.as_ptr());
                        ffi_status(py, r)?;
                    }
                }
                // Add zero entries for keys only in other
                let mut pos: ffi::Py_ssize_t = 0;
                let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                let mut _v: *mut ffi::PyObject = std::ptr::null_mut();
                while ffi::PyDict_Next(other_data.as_ptr(), &mut pos, &mut key, &mut _v) != 0 {
                    let in_self = ffi::PyDict_Contains(self_data.as_ptr(), key);
                    if in_self < 0 {
                        return Err(PyErr::fetch(py));
                    }
                    if in_self == 0 {
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, zero.as_ptr());
                        ffi_status(py, r)?;
                    }
                }
            }
            return Ok(());
        }
        for k in inner.union_keys(py, &other_inner)? {
            let lv = inner.get_value(py, &k)?;
            let rv = other_inner.get_value(py, &k)?;
            inner.data.bind(py).set_item(k, lv.mul(rv)?)?;
        }
        return Ok(());
    }
    let mask_ptr = if inner.lazy_mask {
        Some(inner.mask.as_ref().unwrap().bind(py).as_ptr())
    } else {
        None
    };
    unsafe {
        dict_iop_scalar(
            py,
            inner.data.bind(py).as_ptr(),
            other.as_ptr(),
            ffi::PyNumber_InPlaceMultiply,
            mask_ptr,
        )?;
    }
    Ok(())
}

#[inline]
fn binop_div<'py>(
    py: Python<'py>,
    left: &VectorDict,
    other: &Bound<'py, PyAny>,
    reverse: bool,
) -> PyResult<Py<VectorDict>> {
    if let Ok(other_vd) = other.downcast::<VectorDict>() {
        let other_inner = other_vd.borrow();
        let (l, r) = if reverse {
            (&*other_inner, left)
        } else {
            (left, &*other_inner)
        };
        return Py::new(py, VectorDict::from_dict(div_dict_dict(py, l, r)?));
    }
    let res = if reverse {
        scalar_op_into_new::<true>(py, left, other.as_ptr(), ffi::PyNumber_TrueDivide)?
    } else {
        scalar_op_into_new::<false>(py, left, other.as_ptr(), ffi::PyNumber_TrueDivide)?
    };
    Py::new(py, VectorDict::from_dict(res))
}

#[inline]
fn div_dict_dict<'py>(
    py: Python<'py>,
    left: &VectorDict,
    right: &VectorDict,
) -> PyResult<Py<PyDict>> {
    if left.is_simple() && right.is_simple() {
        let res = PyDict::new(py);
        let left_data = left.data.bind(py);
        let right_data = right.data.bind(py);
        let zero = 0i64.into_bound_py_any(py)?;
        unsafe {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut left_val: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(left_data.as_ptr(), &mut pos, &mut key, &mut left_val) != 0 {
                let right_val = dict_get_or_null(right_data.as_ptr(), key);
                let new_val = if !right_val.is_null() {
                    ffi_check(py, ffi::PyNumber_TrueDivide(left_val, right_val))?
                } else {
                    // matches Cython: `left_value / 0` raises ZeroDivisionError
                    ffi_check(py, ffi::PyNumber_TrueDivide(left_val, zero.as_ptr()))?
                };
                let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
                ffi::Py_DECREF(new_val);
                ffi_status(py, r)?;
            }
            // For keys only in right: 0 / right_val
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: *mut ffi::PyObject = std::ptr::null_mut();
            let mut right_val: *mut ffi::PyObject = std::ptr::null_mut();
            while ffi::PyDict_Next(right_data.as_ptr(), &mut pos, &mut key, &mut right_val) != 0 {
                let in_left = ffi::PyDict_Contains(left_data.as_ptr(), key);
                if in_left < 0 {
                    return Err(PyErr::fetch(py));
                }
                if in_left == 0 {
                    let new_val =
                        ffi_check(py, ffi::PyNumber_TrueDivide(zero.as_ptr(), right_val))?;
                    let r = ffi::PyDict_SetItem(res.as_ptr(), key, new_val);
                    ffi::Py_DECREF(new_val);
                    ffi_status(py, r)?;
                }
            }
        }
        return Ok(res.unbind());
    }
    let res = PyDict::new(py);
    for k in left.union_keys(py, right)? {
        let lv = left.get_value(py, &k)?;
        let rv = right.get_value(py, &k)?;
        res.set_item(k, lv.div(rv)?)?;
    }
    Ok(res.unbind())
}

fn idiv<'py>(py: Python<'py>, inner: &VectorDict, other: &Bound<'py, PyAny>) -> PyResult<()> {
    if let Ok(other_vd) = other.downcast::<VectorDict>() {
        let other_inner = other_vd.borrow();
        if inner.is_simple() && other_inner.is_simple() {
            let self_data = inner.data.bind(py);
            let other_data = other_inner.data.bind(py);
            let zero = 0i64.into_bound_py_any(py)?;
            unsafe {
                let mut pos: ffi::Py_ssize_t = 0;
                let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                let mut self_val: *mut ffi::PyObject = std::ptr::null_mut();
                while ffi::PyDict_Next(self_data.as_ptr(), &mut pos, &mut key, &mut self_val) != 0
                {
                    let other_val = dict_get_or_null(other_data.as_ptr(), key);
                    let new_val = if !other_val.is_null() {
                        ffi_check(py, ffi::PyNumber_TrueDivide(self_val, other_val))?
                    } else {
                        ffi_check(py, ffi::PyNumber_TrueDivide(self_val, zero.as_ptr()))?
                    };
                    let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, new_val);
                    ffi::Py_DECREF(new_val);
                    ffi_status(py, r)?;
                }
                // Keys only in other
                let mut pos: ffi::Py_ssize_t = 0;
                let mut key: *mut ffi::PyObject = std::ptr::null_mut();
                let mut other_val: *mut ffi::PyObject = std::ptr::null_mut();
                while ffi::PyDict_Next(other_data.as_ptr(), &mut pos, &mut key, &mut other_val)
                    != 0
                {
                    let in_self = ffi::PyDict_Contains(self_data.as_ptr(), key);
                    if in_self < 0 {
                        return Err(PyErr::fetch(py));
                    }
                    if in_self == 0 {
                        let new_val =
                            ffi_check(py, ffi::PyNumber_TrueDivide(zero.as_ptr(), other_val))?;
                        let r = ffi::PyDict_SetItem(self_data.as_ptr(), key, new_val);
                        ffi::Py_DECREF(new_val);
                        ffi_status(py, r)?;
                    }
                }
            }
            return Ok(());
        }
        for k in inner.union_keys(py, &other_inner)? {
            let lv = inner.get_value(py, &k)?;
            let rv = other_inner.get_value(py, &k)?;
            inner.data.bind(py).set_item(k, lv.div(rv)?)?;
        }
        return Ok(());
    }
    let mask_ptr = if inner.lazy_mask {
        Some(inner.mask.as_ref().unwrap().bind(py).as_ptr())
    } else {
        None
    };
    unsafe {
        dict_iop_scalar(
            py,
            inner.data.bind(py).as_ptr(),
            other.as_ptr(),
            ffi::PyNumber_InPlaceTrueDivide,
            mask_ptr,
        )?;
    }
    Ok(())
}

fn matmul<'py>(
    py: Python<'py>,
    left: &VectorDict,
    right: &VectorDict,
) -> PyResult<Bound<'py, PyAny>> {
    let mut acc: Bound<'py, PyAny> = 0i64.into_bound_py_any(py)?;
    if left.use_factory || right.use_factory {
        for k in left.union_keys(py, right)? {
            let lv = left.get_value(py, &k)?;
            let rv = right.get_value(py, &k)?;
            acc = acc.add(lv.mul(rv)?)?;
        }
        return Ok(acc);
    }
    if left.use_mask || right.use_mask {
        for k in left.intersection_keys(py, right)? {
            let lv = left.data.bind(py).get_item(&k)?.unwrap();
            let rv = right.data.bind(py).get_item(&k)?.unwrap();
            acc = acc.add(lv.mul(rv)?)?;
        }
        return Ok(acc);
    }
    // Pure simple: iterate the smaller dict
    let left_data = left.data.bind(py);
    let right_data = right.data.bind(py);
    let (small_data, large_data) = if right_data.len() < left_data.len() {
        (&right_data, &left_data)
    } else {
        (&left_data, &right_data)
    };
    unsafe {
        let mut pos: ffi::Py_ssize_t = 0;
        let mut key: *mut ffi::PyObject = std::ptr::null_mut();
        let mut sv: *mut ffi::PyObject = std::ptr::null_mut();
        while ffi::PyDict_Next(small_data.as_ptr(), &mut pos, &mut key, &mut sv) != 0 {
            let lv = dict_get_or_null(large_data.as_ptr(), key);
            if lv.is_null() {
                continue; // missing in large = 0, contributes nothing
            }
            let prod = ffi_check(py, ffi::PyNumber_Multiply(sv, lv))?;
            // PyNumber_Add returns new ref; replace acc, drop the old
            let new_acc = ffi_check(py, ffi::PyNumber_Add(acc.as_ptr(), prod))?;
            ffi::Py_DECREF(prod);
            // Build a fresh Bound from new_acc (steal ref)
            acc = Bound::from_owned_ptr(py, new_acc);
        }
    }
    Ok(acc)
}

fn per_element_min_max<'py>(
    py: Python<'py>,
    left: &VectorDict,
    other: &Bound<'py, PyAny>,
    want_min: bool,
) -> PyResult<Py<VectorDict>> {
    let res = PyDict::new(py);
    if let Ok(other_vd) = other.downcast::<VectorDict>() {
        let right = other_vd.borrow();
        if left.is_simple() && right.is_simple() {
            let zero = 0i64.into_bound_py_any(py)?;
            // Iterate left
            for (k, lv) in left.data.bind(py).iter() {
                let rv_opt = right.data.bind(py).get_item(&k)?;
                let rv = rv_opt.unwrap_or_else(|| zero.clone());
                let chosen = pick(&lv, &rv, want_min)?;
                res.set_item(k, chosen)?;
            }
            // Add right-only entries
            for (k, rv) in right.data.bind(py).iter() {
                if !left.data.bind(py).contains(&k)? {
                    let chosen = pick(&zero, &rv, want_min)?;
                    res.set_item(k, chosen)?;
                }
            }
            return Py::new(py, VectorDict::from_dict(res.unbind()));
        }
        for k in left.union_keys(py, &right)? {
            let lv = left.get_value(py, &k)?;
            let rv = right.get_value(py, &k)?;
            let chosen = pick(&lv, &rv, want_min)?;
            res.set_item(k, chosen)?;
        }
        return Py::new(py, VectorDict::from_dict(res.unbind()));
    }
    // vec OP scalar
    let dict = left.to_dict_internal(py, true)?;
    for (k, v) in dict.iter() {
        let chosen = pick(&v, other, want_min)?;
        dict.set_item(k, chosen)?;
    }
    Py::new(py, VectorDict::from_dict(dict.unbind()))
}

fn pick<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    want_min: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let cmp = if want_min {
        b.lt(a)?
    } else {
        b.gt(a)?
    };
    Ok(if cmp { b.clone() } else { a.clone() })
}

fn euclidean_distance_dict_dict<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyDict>,
    b: &Bound<'py, PyDict>,
) -> PyResult<f64> {
    let mut total: f64 = 0.0;
    unsafe {
        let mut pos: ffi::Py_ssize_t = 0;
        let mut key: *mut ffi::PyObject = std::ptr::null_mut();
        let mut a_val: *mut ffi::PyObject = std::ptr::null_mut();
        while ffi::PyDict_Next(a.as_ptr(), &mut pos, &mut key, &mut a_val) != 0 {
            let av = ffi::PyFloat_AsDouble(a_val);
            if av == -1.0 && !ffi::PyErr_Occurred().is_null() {
                return Err(PyErr::fetch(py));
            }
            let b_val = dict_get_or_null(b.as_ptr(), key);
            let bv = if !b_val.is_null() {
                let v = ffi::PyFloat_AsDouble(b_val);
                if v == -1.0 && !ffi::PyErr_Occurred().is_null() {
                    return Err(PyErr::fetch(py));
                }
                v
            } else {
                0.0
            };
            let d = av - bv;
            total += d * d;
        }
        let mut pos: ffi::Py_ssize_t = 0;
        let mut key: *mut ffi::PyObject = std::ptr::null_mut();
        let mut b_val: *mut ffi::PyObject = std::ptr::null_mut();
        while ffi::PyDict_Next(b.as_ptr(), &mut pos, &mut key, &mut b_val) != 0 {
            let in_a = ffi::PyDict_Contains(a.as_ptr(), key);
            if in_a < 0 {
                return Err(PyErr::fetch(py));
            }
            if in_a == 0 {
                let bv = ffi::PyFloat_AsDouble(b_val);
                if bv == -1.0 && !ffi::PyErr_Occurred().is_null() {
                    return Err(PyErr::fetch(py));
                }
                total += bv * bv;
            }
        }
    }
    Ok(total.sqrt())
}

// === Module-level utility functions ==============================================

#[pyfunction]
pub fn euclidean_distance_dict<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyDict>,
    b: &Bound<'py, PyDict>,
) -> PyResult<f64> {
    euclidean_distance_dict_dict(py, a, b)
}

#[pyfunction]
pub fn euclidean_distance_tuple<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyTuple>,
    b: &Bound<'py, PyTuple>,
) -> PyResult<f64> {
    let da = a.get_item(0)?;
    let db = b.get_item(0)?;
    let da = da.downcast::<PyDict>()?;
    let db = db.downcast::<PyDict>()?;
    euclidean_distance_dict_dict(py, da, db)
}

/// `lazy_search_euclidean(query, window, n_neighbors)` — KNN over a deque.
/// Window entries are `(item, ...)` tuples where `item[0]` is a feature dict.
#[pyfunction]
pub fn lazy_search_euclidean<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyTuple>,
    window: Bound<'py, PyAny>,
    n_neighbors: i32,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let qx = query.get_item(0)?;
    let qx = qx.downcast::<PyDict>()?;
    let k = n_neighbors.max(0) as usize;

    // The Cython path uses the C-level `_heapify_max` / `_heapreplace_max` from
    // `heapq`. We mirror that exactly for byte-identical behavior on ties.
    let heapq = py.import("heapq")?;
    let heapify_max = heapq.getattr("_heapify_max")?;
    let heapreplace_max = heapq.getattr("_heapreplace_max")?;

    let heap = pyo3::types::PyList::empty(py);
    let mut i: usize = 0;
    let iter = window.try_iter()?;
    for entry_res in iter {
        let entry = entry_res?;
        let item_tuple = entry.get_item(0)?;
        let item_tuple = item_tuple.downcast::<PyTuple>()?;
        let px = item_tuple.get_item(0)?;
        let px = px.downcast::<PyDict>()?;
        let dist_sq = squared_euclid(py, qx, px)?;
        let neg = -dist_sq;
        let triple = PyTuple::new(py, [
            neg.into_bound_py_any(py)?,
            i.into_bound_py_any(py)?,
            entry.clone(),
        ])?;
        if i < k {
            heap.append(triple)?;
            if i + 1 == k {
                heapify_max.call1((&heap,))?;
            }
        } else {
            // Compare against current max (heap[0])
            let top = heap.get_item(0)?;
            let top_neg = top.get_item(0)?.extract::<f64>()?;
            if dist_sq < -top_neg {
                heapreplace_max.call1((&heap, triple))?;
            }
        }
        i += 1;
    }

    // Sort by neg-distance descending so the smallest distance is last popped
    heap.sort()?;
    heap.reverse()?;

    let items = pyo3::types::PyList::empty(py);
    let distances = pyo3::types::PyList::empty(py);
    for triple in heap.iter() {
        let neg = triple.get_item(0)?.extract::<f64>()?;
        let entry = triple.get_item(2)?;
        let item = entry.get_item(0)?;
        items.append(item)?;
        distances.append((-neg).sqrt())?;
    }
    Ok((items.unbind().into_any(), distances.unbind().into_any()))
}

fn squared_euclid<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyDict>,
    b: &Bound<'py, PyDict>,
) -> PyResult<f64> {
    let mut total = 0.0_f64;
    unsafe {
        let mut pos: ffi::Py_ssize_t = 0;
        let mut key: *mut ffi::PyObject = std::ptr::null_mut();
        let mut a_val: *mut ffi::PyObject = std::ptr::null_mut();
        while ffi::PyDict_Next(a.as_ptr(), &mut pos, &mut key, &mut a_val) != 0 {
            let av = ffi::PyFloat_AsDouble(a_val);
            if av == -1.0 && !ffi::PyErr_Occurred().is_null() {
                return Err(PyErr::fetch(py));
            }
            let b_val = dict_get_or_null(b.as_ptr(), key);
            let bv = if !b_val.is_null() {
                let v = ffi::PyFloat_AsDouble(b_val);
                if v == -1.0 && !ffi::PyErr_Occurred().is_null() {
                    return Err(PyErr::fetch(py));
                }
                v
            } else {
                0.0
            };
            let d = av - bv;
            total += d * d;
        }
        let mut pos: ffi::Py_ssize_t = 0;
        let mut key: *mut ffi::PyObject = std::ptr::null_mut();
        let mut b_val: *mut ffi::PyObject = std::ptr::null_mut();
        while ffi::PyDict_Next(b.as_ptr(), &mut pos, &mut key, &mut b_val) != 0 {
            let in_a = ffi::PyDict_Contains(a.as_ptr(), key);
            if in_a < 0 {
                return Err(PyErr::fetch(py));
            }
            if in_a == 0 {
                let bv = ffi::PyFloat_AsDouble(b_val);
                if bv == -1.0 && !ffi::PyErr_Occurred().is_null() {
                    return Err(PyErr::fetch(py));
                }
                total += bv * bv;
            }
        }
    }
    Ok(total)
}
