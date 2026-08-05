use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyFloat, PyList, PyTuple};
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
    use_mask: bool,
    lazy_mask: bool,
    use_factory: bool,
    default_factory: Option<Py<PyAny>>,
}

impl VectorDict {
    fn from_dict(data: Py<PyDict>) -> Self {
        Self {
            data,
            mask: None,
            use_mask: false,
            lazy_mask: false,
            use_factory: false,
            default_factory: None,
        }
    }

    fn is_simple(&self) -> bool {
        !self.use_mask && !self.use_factory
    }

    fn visible<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(!self.use_mask || self.mask.as_ref().unwrap().bind(py).contains(key)?)
    }

    fn value<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<f64> {
        if !self.visible(py, key)? {
            return Ok(0.0);
        }
        self.value_unchecked(py, key)
    }

    fn value_unchecked<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<f64> {
        if let Some(value) = self.data.bind(py).get_item(key)? {
            return number(&value);
        }
        if self.use_factory {
            let value = number(&self.default_factory.as_ref().unwrap().bind(py).call0()?)?;
            self.data.bind(py).set_item(key, value)?;
            return Ok(value);
        }
        Ok(0.0)
    }

    fn dict<'py>(&self, py: Python<'py>, copy: bool) -> PyResult<Bound<'py, PyDict>> {
        let data = self.data.bind(py);
        if !self.lazy_mask {
            return if copy { data.copy() } else { Ok(data.clone()) };
        }
        let out = PyDict::new(py);
        for (key, value) in data.iter() {
            if self.visible(py, &key)? {
                out.set_item(key, value)?;
            }
        }
        Ok(out)
    }

    fn keys_vec<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut keys = Vec::new();
        if self.lazy_mask {
            let data = self.data.bind(py);
            let mask = self.mask.as_ref().unwrap().bind(py);
            if matches!(mask.len(), Ok(mask_len) if mask_len < data.len()) {
                for key in mask.try_iter()? {
                    let key = key?;
                    if data.contains(&key)? {
                        keys.push(key);
                    }
                }
                return Ok(keys);
            }
        }
        for (key, _) in self.data.bind(py).iter() {
            if self.visible(py, &key)? {
                keys.push(key);
            }
        }
        Ok(keys)
    }

    fn union<'py>(&self, py: Python<'py>, other: &Self) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut keys = self.keys_vec(py)?;
        let seen = self.dict(py, false)?;
        for key in other.keys_vec(py)? {
            if !seen.contains(&key)? {
                keys.push(key);
            }
        }
        Ok(keys)
    }
}

fn number(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(value) = value.cast::<PyFloat>() {
        return Ok(value.value());
    }
    value
        .extract::<f64>()
        .map_err(|_| PyTypeError::new_err("VectorDict values must be real numbers"))
}

fn scalar(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    number(value)
}

fn normalized<'py>(source: &Bound<'py, PyDict>, copy: bool) -> PyResult<Bound<'py, PyDict>> {
    let out = if copy { source.copy()? } else { source.clone() };
    for (key, value) in out.iter() {
        if value.cast::<PyFloat>().is_err() {
            out.set_item(key, number(&value)?)?;
        }
    }
    Ok(out)
}

fn new_result(py: Python<'_>, data: Bound<'_, PyDict>) -> PyResult<Py<VectorDict>> {
    Py::new(py, VectorDict::from_dict(data.unbind()))
}

fn scalar_new(
    py: Python<'_>,
    this: &VectorDict,
    other: f64,
    op: fn(f64, f64) -> f64,
) -> PyResult<Py<VectorDict>> {
    let out = PyDict::new(py);
    if this.is_simple() {
        for (key, value) in this.data.bind(py).iter() {
            out.set_item(key, op(number(&value)?, other))?;
        }
        return new_result(py, out);
    }
    for key in this.keys_vec(py)? {
        out.set_item(&key, op(this.value(py, &key)?, other))?;
    }
    new_result(py, out)
}

fn vector_new(
    py: Python<'_>,
    left: &VectorDict,
    right: &VectorDict,
    op: fn(f64, f64) -> f64,
) -> PyResult<Py<VectorDict>> {
    let out = PyDict::new(py);
    if left.is_simple() && right.is_simple() {
        let left_data = left.data.bind(py);
        let right_data = right.data.bind(py);
        let mut matched = 0;
        for (key, left_value) in left_data.iter() {
            let right_value = match right_data.get_item(&key)? {
                Some(value) => {
                    matched += 1;
                    number(&value)?
                }
                None => 0.0,
            };
            out.set_item(key, op(number(&left_value)?, right_value))?;
        }
        if matched < right_data.len() {
            for (key, right_value) in right_data.iter() {
                if !left_data.contains(&key)? {
                    out.set_item(key, op(0.0, number(&right_value)?))?;
                }
            }
        }
        return new_result(py, out);
    }
    for key in left.union(py, right)? {
        out.set_item(&key, op(left.value(py, &key)?, right.value(py, &key)?))?;
    }
    new_result(py, out)
}

fn apply_in_place(
    py: Python<'_>,
    this: &VectorDict,
    other: &Bound<'_, PyAny>,
    op: fn(f64, f64) -> PyResult<f64>,
) -> PyResult<()> {
    if let Ok(vd) = other.cast::<VectorDict>() {
        let vd = vd.borrow();
        if this.is_simple() && vd.is_simple() {
            let data = this.data.bind(py);
            let other_data = vd.data.bind(py);
            for (key, value) in data.iter() {
                let right = match other_data.get_item(&key)? {
                    Some(value) => number(&value)?,
                    None => 0.0,
                };
                data.set_item(key, op(number(&value)?, right)?)?;
            }
            for (key, value) in other_data.iter() {
                if !data.contains(&key)? {
                    data.set_item(key, op(0.0, number(&value)?)?)?;
                }
            }
            return Ok(());
        }
        for key in this.union(py, &vd)? {
            if !this.visible(py, &key)? {
                continue;
            }
            this.data.bind(py).set_item(
                &key,
                op(this.value_unchecked(py, &key)?, vd.value(py, &key)?)?,
            )?;
        }
    } else {
        let rhs = scalar(other)?;
        if this.is_simple() {
            let data = this.data.bind(py);
            for (key, value) in data.iter() {
                let value = number(&value)?;
                data.set_item(key, op(value, rhs)?)?;
            }
            return Ok(());
        }
        for key in this.keys_vec(py)? {
            this.data
                .bind(py)
                .set_item(&key, op(this.value(py, &key)?, rhs)?)?;
        }
    }
    Ok(())
}

fn div(a: f64, b: f64) -> PyResult<f64> {
    if b == 0.0 {
        Err(PyZeroDivisionError::new_err("float division by zero"))
    } else {
        Ok(a / b)
    }
}

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
        let data = match data {
            None => PyDict::new(py),
            Some(value) => {
                if let Ok(dict) = value.cast::<PyDict>() {
                    normalized(dict, copy)?
                } else if let Ok(vd) = value.cast::<VectorDict>() {
                    vd.borrow().dict(py, copy)?
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported type for data: {}",
                        value.get_type().name()?
                    )));
                }
            }
        };
        let mask = if copy {
            match mask {
                Some(value) => Some(py.import("builtins")?.getattr("set")?.call1((value,))?),
                None => None,
            }
        } else {
            mask
        };
        if copy {
            if let Some(ref selected) = mask {
                let remove: Vec<_> = data
                    .iter()
                    .filter_map(|(key, _)| match selected.contains(&key) {
                        Ok(false) => Some(Ok(key)),
                        Ok(true) => None,
                        Err(err) => Some(Err(err)),
                    })
                    .collect::<PyResult<_>>()?;
                for key in remove {
                    data.del_item(key)?;
                }
            }
        }
        Ok(Self {
            data: data.unbind(),
            use_mask: mask.is_some(),
            lazy_mask: mask.is_some() && !copy,
            mask: mask.map(Bound::unbind),
            use_factory: default_factory.is_some(),
            default_factory: default_factory.map(Bound::unbind),
        })
    }

    #[pyo3(signature = (mask, copy=false))]
    fn with_mask<'py>(
        &self,
        py: Python<'py>,
        mask: Option<Bound<'py, PyAny>>,
        copy: bool,
    ) -> PyResult<Self> {
        Self::new(
            py,
            Some(self.data.bind(py).clone().into_any()),
            self.default_factory.as_ref().map(|x| x.bind(py).clone()),
            mask,
            copy,
        )
    }

    #[staticmethod]
    fn from_scaled(
        py: Python<'_>,
        values: &Bound<'_, PyDict>,
        scalar: f64,
    ) -> PyResult<Py<VectorDict>> {
        let out = PyDict::new(py);
        for (key, value) in values.iter() {
            out.set_item(key, number(&value)? * scalar)?;
        }
        new_result(py, out)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.dict(py, true)
    }

    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        fields: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let values = PyList::empty(py);
        for key in fields.try_iter()? {
            values.append(self.value(py, &key?)?)?;
        }
        py.import("numpy")?.getattr("array")?.call1((values,))
    }

    fn __contains__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.visible(py, &key)? && self.data.bind(py).contains(key)?)
    }
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.keys_vec(py)?.len())
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.dict(py, false)?.as_any().call_method0("__iter__")
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.dict(py, false)?.repr()?.to_string())
    }
    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.dict(py, false)?.str()?.to_string())
    }
    fn __format__<'py>(&self, py: Python<'py>, spec: Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(self
            .dict(py, false)?
            .as_any()
            .call_method1("__format__", (spec,))?
            .unbind())
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !self.visible(py, &key)? {
            return Err(PyKeyError::new_err(key.unbind()));
        }
        if let Some(value) = self.data.bind(py).get_item(&key)? {
            return Ok(value);
        }
        if self.use_factory {
            let value = number(&self.default_factory.as_ref().unwrap().bind(py).call0()?)?;
            self.data.bind(py).set_item(&key, value)?;
            return value.into_bound_py_any(py);
        }
        Err(PyKeyError::new_err(key.unbind()))
    }

    fn __setitem__(
        &self,
        py: Python<'_>,
        key: Bound<'_, PyAny>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if !self.visible(py, &key)? {
            return Err(PyKeyError::new_err(key.unbind()));
        }
        self.data.bind(py).set_item(key, number(&value)?)
    }
    fn __delitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<()> {
        if !self.visible(py, &key)? {
            return Err(PyKeyError::new_err(key.unbind()));
        }
        self.data.bind(py).del_item(key)
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        let keys = self.keys_vec(py)?;
        for key in keys {
            self.data.bind(py).del_item(key)?;
        }
        Ok(())
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.dict(py, false)?.as_any().call_method0("items")
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.dict(py, false)?.as_any().call_method0("keys")
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.dict(py, false)?.as_any().call_method0("values")
    }
    #[pyo3(signature = (key, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.visible(py, &key)? {
            if let Some(value) = self.data.bind(py).get_item(key)? {
                return Ok(value);
            }
        }
        Ok(default.unwrap_or_else(|| py.None().into_bound(py)))
    }
    #[pyo3(signature = (key, default=None))]
    fn pop<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !self.visible(py, &key)? {
            return match default {
                Some(v) => Ok(v),
                None => Err(PyKeyError::new_err(key.unbind())),
            };
        }
        match self.data.bind(py).get_item(&key)? {
            Some(value) => {
                self.data.bind(py).del_item(key)?;
                Ok(value)
            }
            None => match default {
                Some(v) => Ok(v),
                None => Err(PyKeyError::new_err(key.unbind())),
            },
        }
    }
    fn popitem<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let key = self
            .keys_vec(py)?
            .pop()
            .ok_or_else(|| PyKeyError::new_err("popitem(): dictionary is empty"))?;
        let value = self.data.bind(py).get_item(&key)?.unwrap();
        self.data.bind(py).del_item(&key)?;
        PyTuple::new(py, [key, value]).map(|x| x.into_any())
    }
    #[pyo3(signature = (key, default=None))]
    fn setdefault<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !self.visible(py, &key)? {
            return Err(PyKeyError::new_err(key.unbind()));
        }
        if let Some(value) = self.data.bind(py).get_item(&key)? {
            return Ok(value);
        }
        let value = match default {
            Some(value) => number(&value)?,
            None => 0.0,
        };
        self.data.bind(py).set_item(&key, value)?;
        value.into_bound_py_any(py)
    }
    #[pyo3(signature = (other))]
    fn update(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<()> {
        let incoming = other
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("update requires a dict"))?;
        for (key, value) in incoming.iter() {
            if self.visible(py, &key)? {
                self.data.bind(py).set_item(key, number(&value)?)?;
            }
        }
        Ok(())
    }

    fn __eq__(&self, py: Python<'_>, right: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let left = self.dict(py, false)?;
        if let Ok(vd) = right.cast::<VectorDict>() {
            return Ok(left.eq(vd.borrow().dict(py, false)?)?.into_py_any(py)?);
        }
        if right.cast::<PyDict>().is_ok() {
            return Ok(left.eq(right)?.into_py_any(py)?);
        }
        Ok(py.NotImplemented())
    }

    fn __add__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        if let Ok(vd) = other.cast::<VectorDict>() {
            vector_new(py, self, &vd.borrow(), |a, b| a + b)
        } else {
            scalar_new(py, self, scalar(&other)?, |a, b| a + b)
        }
    }
    fn __radd__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        self.__add__(py, other)
    }
    fn __sub__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        if let Ok(vd) = other.cast::<VectorDict>() {
            vector_new(py, self, &vd.borrow(), |a, b| a - b)
        } else {
            scalar_new(py, self, scalar(&other)?, |a, b| a - b)
        }
    }
    fn __rsub__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        scalar_new(py, self, scalar(&other)?, |a, b| b - a)
    }
    fn __mul__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        if let Ok(vd) = other.cast::<VectorDict>() {
            vector_new(py, self, &vd.borrow(), |a, b| a * b)
        } else {
            scalar_new(py, self, scalar(&other)?, |a, b| a * b)
        }
    }
    fn __rmul__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        self.__mul__(py, other)
    }
    fn __truediv__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        if let Ok(vd) = other.cast::<VectorDict>() {
            let vd = vd.borrow();
            let out = PyDict::new(py);
            for key in self.union(py, &vd)? {
                out.set_item(&key, div(self.value(py, &key)?, vd.value(py, &key)?)?)?;
            }
            new_result(py, out)
        } else {
            let rhs = scalar(&other)?;
            let out = PyDict::new(py);
            for key in self.keys_vec(py)? {
                out.set_item(&key, div(self.value(py, &key)?, rhs)?)?;
            }
            new_result(py, out)
        }
    }
    fn __rtruediv__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        let lhs = scalar(&other)?;
        let out = PyDict::new(py);
        for key in self.keys_vec(py)? {
            out.set_item(&key, div(lhs, self.value(py, &key)?)?)?;
        }
        new_result(py, out)
    }
    fn __pow__(
        &self,
        py: Python<'_>,
        other: Bound<'_, PyAny>,
        _modulo: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<VectorDict>> {
        scalar_new(py, self, scalar(&other)?, f64::powf)
    }
    fn __iadd__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<()> {
        apply_in_place(py, self, &other, |a, b| Ok(a + b))
    }
    fn __isub__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<()> {
        apply_in_place(py, self, &other, |a, b| Ok(a - b))
    }
    fn __imul__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<()> {
        apply_in_place(py, self, &other, |a, b| Ok(a * b))
    }
    fn __itruediv__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<()> {
        apply_in_place(py, self, &other, div)
    }
    fn __ipow__(
        &self,
        py: Python<'_>,
        other: Bound<'_, PyAny>,
        _modulo: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let rhs = scalar(&other)?;
        for key in self.keys_vec(py)? {
            self.data
                .bind(py)
                .set_item(&key, self.value(py, &key)?.powf(rhs))?;
        }
        Ok(())
    }

    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        right: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let vd = match right.cast::<VectorDict>() {
            Ok(v) => v,
            Err(_) => return Ok(py.NotImplemented().bind(py).clone()),
        };
        let vd = vd.borrow();
        let mut total = 0.0;
        if self.is_simple() && vd.is_simple() {
            let left = self.data.bind(py);
            let right = vd.data.bind(py);
            let (small, large) = if left.len() <= right.len() {
                (&left, &right)
            } else {
                (&right, &left)
            };
            for (key, value) in small.iter() {
                if let Some(other) = large.get_item(key)? {
                    total += number(&value)? * number(&other)?;
                }
            }
            return total.into_bound_py_any(py);
        }
        if self.use_factory || vd.use_factory {
            for key in self.union(py, &vd)? {
                total += self.value(py, &key)? * vd.value(py, &key)?;
            }
        } else {
            let (small, large) = if self.keys_vec(py)?.len() <= vd.keys_vec(py)?.len() {
                (self, &*vd)
            } else {
                (&*vd, self)
            };
            for key in small.keys_vec(py)? {
                if large.visible(py, &key)? {
                    if let Some(value) = large.data.bind(py).get_item(&key)? {
                        total += small.value(py, &key)? * number(&value)?;
                    }
                }
            }
        }
        total.into_bound_py_any(py)
    }

    fn dot(&self, py: Python<'_>, values: &Bound<'_, PyDict>) -> PyResult<f64> {
        let mut total = 0.0;
        let mask_matches = self
            .mask
            .as_ref()
            .is_some_and(|mask| mask.bind(py).is(values));
        for (key, value) in values.iter() {
            let weight = if !self.use_mask || mask_matches {
                self.value_unchecked(py, &key)?
            } else {
                self.value(py, &key)?
            };
            total += weight * number(&value)?;
        }
        Ok(total)
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<VectorDict>> {
        scalar_new(py, self, -1.0, |a, b| a * b)
    }
    fn __pos__(&self, py: Python<'_>) -> PyResult<Py<VectorDict>> {
        new_result(py, self.dict(py, true)?)
    }
    fn __abs__(&self, py: Python<'_>) -> PyResult<Py<VectorDict>> {
        scalar_new(py, self, 0.0, |a, _| a.abs())
    }
    fn abs(&self, py: Python<'_>) -> PyResult<Py<VectorDict>> {
        self.__abs__(py)
    }
    fn min(&self, py: Python<'_>) -> PyResult<f64> {
        self.keys_vec(py)?
            .iter()
            .map(|k| self.value(py, k))
            .collect::<PyResult<Vec<_>>>()?
            .into_iter()
            .reduce(f64::min)
            .ok_or_else(|| PyValueError::new_err("min() arg is an empty sequence"))
    }
    fn max(&self, py: Python<'_>) -> PyResult<f64> {
        self.keys_vec(py)?
            .iter()
            .map(|k| self.value(py, k))
            .collect::<PyResult<Vec<_>>>()?
            .into_iter()
            .reduce(f64::max)
            .ok_or_else(|| PyValueError::new_err("max() arg is an empty sequence"))
    }
    fn minimum(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        if let Ok(vd) = other.cast::<VectorDict>() {
            vector_new(py, self, &vd.borrow(), f64::min)
        } else {
            scalar_new(py, self, scalar(&other)?, f64::min)
        }
    }
    fn maximum(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        if let Ok(vd) = other.cast::<VectorDict>() {
            vector_new(py, self, &vd.borrow(), f64::max)
        } else {
            scalar_new(py, self, scalar(&other)?, f64::max)
        }
    }

    fn iadd_scaled(
        slf: Bound<'_, VectorDict>,
        py: Python<'_>,
        other: Bound<'_, PyAny>,
        scale: Bound<'_, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        let s = scalar(&scale)?;
        {
            let this = slf.borrow();
            if let Ok(vd) = other.cast::<VectorDict>() {
                let vd = vd.borrow();
                if vd.is_simple() {
                    for (key, other_value) in vd.data.bind(py).iter() {
                        if !this.visible(py, &key)? {
                            continue;
                        }
                        let value = this.value_unchecked(py, &key)? + s * number(&other_value)?;
                        this.data.bind(py).set_item(key, value)?;
                    }
                } else {
                    for key in vd.keys_vec(py)? {
                        if !this.visible(py, &key)? {
                            continue;
                        }
                        let value = this.value_unchecked(py, &key)? + s * vd.value(py, &key)?;
                        this.data.bind(py).set_item(key, value)?;
                    }
                }
            } else {
                return Err(PyTypeError::new_err("expected VectorDict"));
            }
        }
        Ok(slf.unbind())
    }
    fn isub_scaled(
        slf: Bound<'_, VectorDict>,
        py: Python<'_>,
        other: Bound<'_, PyAny>,
        scale: Bound<'_, PyAny>,
    ) -> PyResult<Py<VectorDict>> {
        let s = scalar(&scale)?;
        {
            let this = slf.borrow();
            if let Ok(vd) = other.cast::<VectorDict>() {
                let vd = vd.borrow();
                if vd.is_simple() {
                    for (key, other_value) in vd.data.bind(py).iter() {
                        if !this.visible(py, &key)? {
                            continue;
                        }
                        let value = this.value_unchecked(py, &key)? - s * number(&other_value)?;
                        this.data.bind(py).set_item(key, value)?;
                    }
                } else {
                    for key in vd.keys_vec(py)? {
                        if !this.visible(py, &key)? {
                            continue;
                        }
                        let value = this.value_unchecked(py, &key)? - s * vd.value(py, &key)?;
                        this.data.bind(py).set_item(key, value)?;
                    }
                }
            } else {
                return Err(PyTypeError::new_err("expected VectorDict"));
            }
        }
        Ok(slf.unbind())
    }

    #[pyo3(signature = (other, decay, square=false))]
    fn update_ema(
        slf: Bound<'_, VectorDict>,
        py: Python<'_>,
        other: Bound<'_, VectorDict>,
        decay: f64,
        square: bool,
    ) -> PyResult<Py<VectorDict>> {
        {
            let this = slf.borrow();
            let other = other.borrow();
            if other.is_simple() {
                for (key, incoming) in other.data.bind(py).iter() {
                    if !this.visible(py, &key)? {
                        continue;
                    }
                    let incoming = number(&incoming)?;
                    let incoming = if square {
                        incoming * incoming
                    } else {
                        incoming
                    };
                    let value = decay * this.value_unchecked(py, &key)? + (1.0 - decay) * incoming;
                    this.data.bind(py).set_item(key, value)?;
                }
                return Ok(slf.unbind());
            }
            for key in other.keys_vec(py)? {
                if !this.visible(py, &key)? {
                    continue;
                }
                let incoming = other.value(py, &key)?;
                let incoming = if square {
                    incoming * incoming
                } else {
                    incoming
                };
                let value = decay * this.value_unchecked(py, &key)? + (1.0 - decay) * incoming;
                this.data.bind(py).set_item(key, value)?;
            }
        }
        Ok(slf.unbind())
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Py<VectorDict>> {
        Py::new(
            py,
            Self {
                data: self.data.bind(py).copy()?.unbind(),
                mask: self.mask.as_ref().map(|x| x.clone_ref(py)),
                use_mask: self.use_mask,
                lazy_mask: self.lazy_mask,
                use_factory: self.use_factory,
                default_factory: self.default_factory.as_ref().map(|x| x.clone_ref(py)),
            },
        )
    }
    fn __deepcopy__(&self, py: Python<'_>, _memo: Bound<'_, PyAny>) -> PyResult<Py<VectorDict>> {
        self.__copy__(py)
    }
    fn __reduce__<'py>(slf: Bound<'py, VectorDict>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let cls = slf.get_type();
        let this = slf.borrow();
        let args = PyTuple::new(
            py,
            [
                this.dict(py, true)?.into_any(),
                this.default_factory
                    .as_ref()
                    .map(|x| x.bind(py).clone())
                    .unwrap_or_else(|| py.None().into_bound(py)),
                this.mask
                    .as_ref()
                    .map(|x| x.bind(py).clone())
                    .unwrap_or_else(|| py.None().into_bound(py)),
                true.into_bound_py_any(py)?,
            ],
        )?;
        Ok(PyTuple::new(py, [cls.into_any(), args.into_any()])?
            .unbind()
            .into_any())
    }
    fn euclidean_distance(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<f64> {
        let dict = if let Ok(vd) = other.cast::<VectorDict>() {
            vd.borrow().dict(py, false)?
        } else {
            other.cast::<PyDict>()?.clone()
        };
        euclidean_distance_dict_dict(self.data.bind(py), &dict)
    }
}

fn euclidean_distance_dict_dict(a: &Bound<'_, PyDict>, b: &Bound<'_, PyDict>) -> PyResult<f64> {
    let mut total = 0.0;
    let mut matched = 0;
    for (key, value) in a.iter() {
        let av = number(&value)?;
        let bv = match b.get_item(&key)? {
            Some(value) => {
                matched += 1;
                number(&value)?
            }
            None => 0.0,
        };
        total += (av - bv) * (av - bv);
    }
    if matched < b.len() {
        for (key, value) in b.iter() {
            if !a.contains(&key)? {
                let value = number(&value)?;
                total += value * value;
            }
        }
    }
    Ok(total.sqrt())
}

#[pyfunction]
pub fn euclidean_distance_dict(a: &Bound<'_, PyDict>, b: &Bound<'_, PyDict>) -> PyResult<f64> {
    euclidean_distance_dict_dict(a, b)
}

#[pyfunction]
pub fn euclidean_distance_tuple(a: &Bound<'_, PyTuple>, b: &Bound<'_, PyTuple>) -> PyResult<f64> {
    euclidean_distance_dict_dict(
        a.get_item(0)?.cast::<PyDict>()?,
        b.get_item(0)?.cast::<PyDict>()?,
    )
}

#[pyfunction]
pub fn lazy_search_euclidean<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyTuple>,
    window: Bound<'py, PyAny>,
    n_neighbors: i32,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let query_item = query.get_item(0)?;
    let qx = query_item.cast::<PyDict>()?;
    let mut found = Vec::new();
    for (index, entry) in window.try_iter()?.enumerate() {
        let entry = entry?;
        let item = entry.get_item(0)?;
        let point = item.get_item(0)?;
        let px = point.cast::<PyDict>()?;
        found.push((euclidean_distance_dict_dict(qx, px)?, index, entry));
    }
    let keep = n_neighbors.max(0) as usize;
    if keep < found.len() {
        found.select_nth_unstable_by(keep, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        found.truncate(keep);
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let items = PyList::empty(py);
    let distances = PyList::empty(py);
    for (distance, _, entry) in found {
        items.append(entry.get_item(0)?)?;
        distances.append(distance)?;
    }
    Ok((items.unbind().into_any(), distances.unbind().into_any()))
}
