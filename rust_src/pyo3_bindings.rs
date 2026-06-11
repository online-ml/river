use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::{Deserialize, Serialize};
use crate::{
    adwin::AdaptiveWindowing, ewmean::EWMean, ewvariance::EWVariance,
    expected_mutual_info::expected_mutual_info as compute_expected_mutual_info,
    iqr::RollingIQR, iqr::IQR, kurtosis::Kurtosis,
    mondrian::{
        go_downwards_classifier, go_downwards_regressor, go_upwards, log_sum_2_exp,
        predict_one_regressor, predict_proba_classifier, predict_proba_upward, predict_scores,
        range_extension, update_ranges,
    },
    ptp::PeakToPeak, quantile::Quantile,
    quantile::RollingQuantile, rolling_pr_auc::RollingPRAUC, rolling_roc_auc::RollingROCAUC,
    skew::Skew, stats::Univariate,
    vectordict::{
        euclidean_distance_dict, euclidean_distance_tuple, lazy_search_euclidean, VectorDict,
    },
};

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsQuantile {
    pub quantile: Quantile<f64>,
}

#[pymethods]
impl RsQuantile {
    #[new]
    #[pyo3(signature = (q=None))]
    pub fn new(q: Option<f64>) -> RsQuantile {
        match q {
            Some(q) => RsQuantile {
                quantile: Quantile::new(q).expect("q should be between 0 and 1"),
            },
            None => RsQuantile {
                quantile: Quantile::default(),
            },
        }
    }
    pub fn update(&mut self, x: f64) {
        self.quantile.update(x);
    }
    pub fn get(&self) -> f64 {
        self.quantile.get()
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsEWMean {
    ewmean: EWMean<f64>,
    alpha: f64,
}
#[pymethods]
impl RsEWMean {
    #[new]
    pub fn new(alpha: f64) -> RsEWMean {
        RsEWMean {
            ewmean: EWMean::new(alpha),
            alpha,
        }
    }
    pub fn update(&mut self, x: f64) {
        self.ewmean.update(x);
    }
    pub fn get(&self) -> f64 {
        self.ewmean.get()
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64,)> {
        Ok((self.alpha,))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsEWVar {
    ewvar: EWVariance<f64>,
    alpha: f64,
}
#[pymethods]
impl RsEWVar {
    #[new]
    pub fn new(alpha: f64) -> RsEWVar {
        RsEWVar {
            ewvar: EWVariance::new(alpha),
            alpha,
        }
    }
    pub fn update(&mut self, x: f64) {
        self.ewvar.update(x);
    }
    pub fn get(&self) -> f64 {
        self.ewvar.get()
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64,)> {
        Ok((self.alpha,))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsIQR {
    iqr: IQR<f64>,
    q_inf: f64,
    q_sup: f64,
}

#[pymethods]
impl RsIQR {
    #[new]
    pub fn new(q_inf: f64, q_sup: f64) -> RsIQR {
        RsIQR {
            iqr: IQR::new(q_inf, q_sup).expect("TODO"),
            q_inf,
            q_sup,
        }
    }
    pub fn update(&mut self, x: f64) {
        self.iqr.update(x);
    }
    pub fn get(&self) -> f64 {
        self.iqr.get()
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64, f64)> {
        Ok((self.q_inf, self.q_sup))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsKurtosis {
    kurtosis: Kurtosis<f64>,
    bias: bool,
}
#[pymethods]
impl RsKurtosis {
    #[new]
    pub fn new(bias: bool) -> RsKurtosis {
        RsKurtosis {
            kurtosis: Kurtosis::new(bias),
            bias,
        }
    }
    pub fn update(&mut self, x: f64) {
        self.kurtosis.update(x);
    }
    pub fn get(&self) -> f64 {
        self.kurtosis.get()
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(bool,)> {
        Ok((self.bias,))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsPeakToPeak {
    ptp: PeakToPeak<f64>,
}

#[pymethods]
impl RsPeakToPeak {
    #[new]
    pub fn new() -> RsPeakToPeak {
        RsPeakToPeak {
            ptp: PeakToPeak::new(),
        }
    }

    pub fn update(&mut self, x: f64) {
        self.ptp.update(x);
    }
    pub fn get(&self) -> f64 {
        self.ptp.get()
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsSkew {
    skew: Skew<f64>,
    bias: bool,
}
#[pymethods]
impl RsSkew {
    #[new]
    pub fn new(bias: bool) -> RsSkew {
        RsSkew {
            skew: Skew::new(bias),
            bias,
        }
    }
    pub fn update(&mut self, x: f64) {
        self.skew.update(x);
    }
    pub fn get(&self) -> f64 {
        self.skew.get()
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(bool,)> {
        Ok((self.bias,))
    }
}
#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsRollingQuantile {
    stat: RollingQuantile<f64>,
    q: f64,
    window_size: usize,
}

#[pymethods]
impl RsRollingQuantile {
    #[new]
    pub fn new(q: f64, window_size: usize) -> RsRollingQuantile {
        RsRollingQuantile {
            stat: RollingQuantile::new(q, window_size).unwrap(),
            q,
            window_size,
        }
    }
    pub fn update(&mut self, x: f64) {
        self.stat.update(x);
    }
    pub fn get(&self) -> f64 {
        self.stat.get()
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64, usize)> {
        Ok((self.q, self.window_size))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsRollingIQR {
    stat: RollingIQR<f64>,
    q_inf: f64,
    q_sup: f64,
    window_size: usize,
}

#[pymethods]
impl RsRollingIQR {
    #[new]
    pub fn new(q_inf: f64, q_sup: f64, window_size: usize) -> RsRollingIQR {
        RsRollingIQR {
            stat: RollingIQR::new(q_inf, q_sup, window_size).unwrap(),
            q_inf,
            q_sup,
            window_size,
        }
    }
    pub fn update(&mut self, x: f64) {
        self.stat.update(x);
    }
    pub fn get(&self) -> f64 {
        self.stat.get()
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64, f64, usize)> {
        Ok((self.q_inf, self.q_sup, self.window_size))
    }
}

#[pyfunction]
#[pyo3(name = "expected_mutual_info")]
fn rs_expected_mutual_info(n_samples: f64, a: Vec<i64>, b: Vec<i64>) -> f64 {
    compute_expected_mutual_info(n_samples, &a, &b)
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsRollingROCAUC {
    inner: RollingROCAUC,
    positive_label: i32,
    window_size: usize,
}

#[pymethods]
impl RsRollingROCAUC {
    #[new]
    pub fn new(positive_label: i32, window_size: usize) -> RsRollingROCAUC {
        RsRollingROCAUC {
            inner: RollingROCAUC::new(positive_label, window_size),
            positive_label,
            window_size,
        }
    }
    pub fn update(&mut self, label: i32, score: f64) {
        self.inner.update(label, score);
    }
    /// Apply many `(label, score)` updates in a single FFI call. The PyO3
    /// per-call cost is several hundred ns, so batching is significantly
    /// faster than calling `update` in a Python loop.
    pub fn update_many(&mut self, labels: Vec<i32>, scores: Vec<f64>) -> PyResult<()> {
        if labels.len() != scores.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "labels and scores must be the same length",
            ));
        }
        for (label, score) in labels.into_iter().zip(scores.into_iter()) {
            self.inner.update(label, score);
        }
        Ok(())
    }
    pub fn revert(&mut self, label: i32, score: f64) {
        self.inner.revert(label, score);
    }
    pub fn get(&self) -> f64 {
        self.inner.get()
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(i32, usize)> {
        Ok((self.positive_label, self.window_size))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsRollingPRAUC {
    inner: RollingPRAUC,
    positive_label: i32,
    window_size: usize,
}

#[pymethods]
impl RsRollingPRAUC {
    #[new]
    pub fn new(positive_label: i32, window_size: usize) -> RsRollingPRAUC {
        RsRollingPRAUC {
            inner: RollingPRAUC::new(positive_label, window_size),
            positive_label,
            window_size,
        }
    }
    pub fn update(&mut self, label: i32, score: f64) {
        self.inner.update(label, score);
    }
    pub fn update_many(&mut self, labels: Vec<i32>, scores: Vec<f64>) -> PyResult<()> {
        if labels.len() != scores.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "labels and scores must be the same length",
            ));
        }
        for (label, score) in labels.into_iter().zip(scores.into_iter()) {
            self.inner.update(label, score);
        }
        Ok(())
    }
    pub fn revert(&mut self, label: i32, score: f64) {
        self.inner.revert(label, score);
    }
    pub fn get(&self) -> f64 {
        self.inner.get()
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(i32, usize)> {
        Ok((self.positive_label, self.window_size))
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(name = "AdaptiveWindowing", module = "river._river_rust.drift")]
pub struct RsAdaptiveWindowing {
    inner: AdaptiveWindowing,
    delta: f64,
    clock: i32,
    max_buckets: usize,
    min_window_length: i32,
    grace_period: i32,
}

#[pymethods]
impl RsAdaptiveWindowing {
    #[new]
    #[pyo3(signature = (delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10))]
    pub fn new(
        delta: f64,
        clock: i32,
        max_buckets: usize,
        min_window_length: i32,
        grace_period: i32,
    ) -> RsAdaptiveWindowing {
        RsAdaptiveWindowing {
            inner: AdaptiveWindowing::new(
                delta,
                clock,
                max_buckets,
                min_window_length,
                grace_period,
            ),
            delta,
            clock,
            max_buckets,
            min_window_length,
            grace_period,
        }
    }
    pub fn update(&mut self, value: f64) -> bool {
        self.inner.update(value)
    }
    pub fn get_n_detections(&self) -> i32 {
        self.inner.n_detections()
    }
    pub fn get_width(&self) -> f64 {
        self.inner.width()
    }
    pub fn get_total(&self) -> f64 {
        self.inner.total()
    }
    pub fn get_variance(&self) -> f64 {
        self.inner.variance()
    }
    #[getter]
    pub fn variance_in_window(&self) -> f64 {
        self.inner.variance_in_window()
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64, i32, usize, i32, i32)> {
        Ok((
            self.delta,
            self.clock,
            self.max_buckets,
            self.min_window_length,
            self.grace_period,
        ))
    }
}

/// Top-level Rust extension. Builds four semantic submodules and registers each
/// in `sys.modules` so dotted imports like `from river._river_rust.stats import X`
/// resolve without first importing the parent.
#[pymodule]
fn _river_rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_submodule(py, m, "stats", register_stats)?;
    register_submodule(py, m, "drift", register_drift)?;
    register_submodule(py, m, "tree", register_tree)?;
    register_submodule(py, m, "vectordict", register_vectordict)?;
    Ok(())
}

fn register_submodule(
    py: Python<'_>,
    parent: &Bound<'_, PyModule>,
    name: &str,
    builder: impl FnOnce(Python<'_>, &Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let qualname = format!("river._river_rust.{}", name);
    let child = PyModule::new(py, name)?;
    // Pin __name__ to the full dotted path before populating so that functions
    // registered via `m.add_function(...)` inherit `__module__` = qualname.
    // PyO3 reads the module's __name__ at the moment a function is attached.
    child.setattr("__name__", &qualname)?;
    builder(py, &child)?;

    let sys = PyModule::import(py, "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;
    sys_modules.set_item(&qualname, &child)?;

    parent.add_submodule(&child)?;
    Ok(())
}

fn register_stats(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RsQuantile>()?;
    m.add_class::<RsEWMean>()?;
    m.add_class::<RsEWVar>()?;
    m.add_class::<RsIQR>()?;
    m.add_class::<RsKurtosis>()?;
    m.add_class::<RsPeakToPeak>()?;
    m.add_class::<RsSkew>()?;
    m.add_class::<RsRollingQuantile>()?;
    m.add_class::<RsRollingIQR>()?;
    m.add_class::<RsRollingROCAUC>()?;
    m.add_class::<RsRollingPRAUC>()?;
    m.add_function(wrap_pyfunction!(rs_expected_mutual_info, m)?)?;
    Ok(())
}

fn register_drift(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RsAdaptiveWindowing>()?;
    Ok(())
}

fn register_tree(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(log_sum_2_exp, m)?)?;
    m.add_function(wrap_pyfunction!(update_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(range_extension, m)?)?;
    m.add_function(wrap_pyfunction!(predict_scores, m)?)?;
    m.add_function(wrap_pyfunction!(go_downwards_classifier, m)?)?;
    m.add_function(wrap_pyfunction!(go_downwards_regressor, m)?)?;
    m.add_function(wrap_pyfunction!(go_upwards, m)?)?;
    m.add_function(wrap_pyfunction!(predict_proba_upward, m)?)?;
    m.add_function(wrap_pyfunction!(predict_proba_classifier, m)?)?;
    m.add_function(wrap_pyfunction!(predict_one_regressor, m)?)?;
    Ok(())
}

fn register_vectordict(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<VectorDict>()?;
    m.add_function(wrap_pyfunction!(euclidean_distance_dict, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean_distance_tuple, m)?)?;
    m.add_function(wrap_pyfunction!(lazy_search_euclidean, m)?)?;
    Ok(())
}
