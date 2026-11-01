use crate::{
    adwin::AdaptiveWindowing,
    ewmean::EWMean,
    ewvariance::EWVariance,
    expected_mutual_info::expected_mutual_info as compute_expected_mutual_info,
    feature_hashing::feature_hash,
    gaussian::Gaussian as NativeGaussian,
    hoeffding_tree::{
        HoeffdingTreeClassifier as NativeHoeffdingTreeClassifier, LeafPrediction, SplitCriterion,
    },
    iqr::RollingIQR,
    iqr::IQR,
    kurtosis::Kurtosis,
    mondrian::{
        MondrianTreeClassifier as NativeMondrianTreeClassifier,
        MondrianTreeRegressor as NativeMondrianTreeRegressor,
    },
    ptp::PeakToPeak,
    quantile::Quantile,
    quantile::RollingQuantile,
    rolling_pr_auc::RollingPRAUC,
    rolling_roc_auc::RollingROCAUC,
    skew::Skew,
    stats::Univariate,
    vectordict::{
        euclidean_distance_dict, euclidean_distance_squared_dict, euclidean_distance_tuple,
        lazy_search_euclidean, VectorDict,
    },
};
use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river._river_rust.stats")]
pub struct RsGaussian {
    inner: NativeGaussian,
}

#[pymethods]
impl RsGaussian {
    #[new]
    #[pyo3(signature = (ddof=1, n=0.0, mean=0.0, variance=0.0))]
    pub fn new(ddof: u32, n: f64, mean: f64, variance: f64) -> PyResult<Self> {
        if !n.is_finite() || n < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n must be finite and non-negative",
            ));
        }
        if !mean.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mean must be finite",
            ));
        }
        if !variance.is_finite() || variance < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "variance must be finite and non-negative",
            ));
        }
        Ok(Self {
            inner: NativeGaussian::from_state(n, mean, variance, ddof),
        })
    }

    #[pyo3(signature = (x, w=1.0))]
    pub fn update(&mut self, x: f64, w: f64) -> PyResult<()> {
        validate_gaussian_observation(x, w)?;
        self.inner.update(x, w);
        Ok(())
    }

    pub fn update_batch(&mut self, n: f64, mean: f64, state: f64) -> PyResult<()> {
        if !n.is_finite() || n < 0.0 || !mean.is_finite() || !state.is_finite() || state < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch statistics must be finite and non-negative",
            ));
        }
        self.inner.update_batch(n, mean, state);
        Ok(())
    }

    #[pyo3(signature = (x, w=1.0))]
    pub fn revert(&mut self, x: f64, w: f64) -> PyResult<()> {
        validate_gaussian_observation(x, w)?;
        self.inner
            .revert(x, w)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        self.inner.log_pdf(x)
    }

    pub fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    #[getter]
    pub fn n_samples(&self) -> f64 {
        self.inner.n_samples()
    }

    #[getter]
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    #[getter]
    pub fn variance(&self) -> f64 {
        self.inner.variance()
    }

    #[getter]
    pub fn standard_deviation(&self) -> f64 {
        self.inner.standard_deviation()
    }

    #[getter]
    pub fn ddof(&self) -> u32 {
        self.inner.ddof()
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "could not deserialize Gaussian: {error}"
            ))
        })?;
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let state = serialize(self).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "could not serialize Gaussian: {error}"
            ))
        })?;
        Ok(PyBytes::new(py, &state))
    }

    pub fn __getnewargs__(&self) -> (u32,) {
        (self.inner.ddof(),)
    }
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

#[pyclass(
    name = "NativeHoeffdingTreeClassifier",
    module = "river._river_rust.tree"
)]
pub struct RsNativeHoeffdingTreeClassifier {
    inner: NativeHoeffdingTreeClassifier,
    n_features: usize,
    n_classes: usize,
    values: Vec<f64>,
}

#[pymethods]
impl RsNativeHoeffdingTreeClassifier {
    #[new]
    #[pyo3(signature = (n_features, n_classes=2, max_features=None, grace_period=200.0, delta=1e-7, tau=0.05, n_splits=10, max_depth=None, split_criterion="info_gain", leaf_prediction="nba", nb_threshold=0.0, min_branch_fraction=0.01, max_share_to_split=0.99, merit_preprune=true, seed=42))]
    fn new(
        n_features: usize,
        n_classes: usize,
        max_features: Option<usize>,
        grace_period: f64,
        delta: f64,
        tau: f64,
        n_splits: usize,
        max_depth: Option<usize>,
        split_criterion: &str,
        leaf_prediction: &str,
        nb_threshold: f64,
        min_branch_fraction: f64,
        max_share_to_split: f64,
        merit_preprune: bool,
        seed: u64,
    ) -> PyResult<Self> {
        validate_tree_parameters(
            n_features,
            n_classes,
            max_features,
            grace_period,
            delta,
            tau,
            n_splits,
            nb_threshold,
            min_branch_fraction,
            max_share_to_split,
        )?;
        let split_criterion = parse_split_criterion(split_criterion)?;
        validate_split_criterion(split_criterion, n_classes)?;
        let leaf_prediction = parse_leaf_prediction(leaf_prediction)?;
        Ok(Self {
            inner: NativeHoeffdingTreeClassifier::new(
                n_features,
                n_classes,
                max_features.unwrap_or(n_features),
                max_features.is_none(),
                grace_period,
                delta,
                tau,
                n_splits,
                max_depth.unwrap_or(usize::MAX),
                seed,
                split_criterion,
                leaf_prediction,
                nb_threshold,
                min_branch_fraction,
                max_share_to_split,
                merit_preprune,
            ),
            n_features,
            n_classes,
            values: vec![f64::NAN; n_features],
        })
    }

    fn predict_one(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<usize> {
        self.validate_x(&x)?;
        Ok(py.detach(|| self.inner.predict(&x)))
    }

    fn predict_proba_one(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<Vec<f64>> {
        self.validate_x(&x)?;
        Ok(py.detach(|| self.inner.predict_proba(&x)))
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_proba_mapping<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        output_labels: &Bound<'_, PyList>,
        output_indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.encode_mapping(x, feature_indices, None)?;
        let probabilities = py.detach(|| self.inner.predict_proba(&self.values));
        if output_labels.len() != output_indices.len()
            || output_indices
                .iter()
                .any(|class_index| *class_index >= probabilities.len())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "class output order is invalid",
            ));
        }
        let output = PyDict::new(py);
        for (label, class_index) in output_labels.iter().zip(output_indices) {
            output.set_item(label, probabilities[class_index])?;
        }
        Ok(output)
    }

    fn predict_mapping(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        class_order: Vec<usize>,
    ) -> PyResult<usize> {
        self.encode_mapping(x, feature_indices, None)?;
        if class_order.is_empty()
            || class_order
                .iter()
                .any(|class_index| *class_index >= self.n_classes)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "class order contains an invalid index",
            ));
        }
        Ok(py.detach(|| self.inner.predict_ordered(&self.values, &class_order)))
    }

    #[pyo3(signature = (x, y, w=1.0))]
    fn learn_one(&mut self, py: Python<'_>, x: Vec<f64>, y: usize, w: f64) -> PyResult<()> {
        self.validate_xy(&x, y)?;
        validate_weight(w)?;
        py.detach(|| self.inner.learn(&x, y, w));
        Ok(())
    }

    #[pyo3(signature = (x, feature_indices, feature_names, y, w=1.0))]
    fn learn_one_mapping(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        feature_names: &Bound<'_, PyList>,
        y: usize,
        w: f64,
    ) -> PyResult<()> {
        self.encode_mapping(x, feature_indices, Some(feature_names))?;
        self.validate_y(y)?;
        validate_weight(w)?;
        py.detach(|| self.inner.learn(&self.values, y, w));
        Ok(())
    }

    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    fn n_branches(&self) -> usize {
        self.inner.n_branches()
    }

    fn n_leaves(&self) -> usize {
        self.inner.n_leaves()
    }

    fn n_active_leaves(&self) -> usize {
        self.inner.n_active_leaves()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }

    fn total_weight_seen(&self) -> f64 {
        self.inner.total_weight_seen()
    }

    fn nodes(&self) -> Vec<crate::hoeffding_tree::NodeSnapshot> {
        self.inner.nodes()
    }

    fn add_features(&mut self, count: usize) {
        self.n_features += count;
        self.values.resize(self.n_features, f64::NAN);
        self.inner.add_features(count);
    }

    fn add_class(&mut self) {
        self.n_classes += 1;
        self.inner.add_class();
    }

    fn insert_class(&mut self, index: usize, grow: bool) -> PyResult<()> {
        if index >= self.n_classes + usize::from(grow) {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "class index out of range",
            ));
        }
        if grow {
            self.n_classes += 1;
        }
        self.inner.insert_class(index, grow);
        Ok(())
    }

    fn set_mutable_parameters(&mut self, grace_period: f64, delta: f64, tau: f64) {
        self.inner.set_mutable_parameters(grace_period, delta, tau);
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let state = serialize(&(1_u8, &self.inner)).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("could not serialize tree: {error}"))
        })?;
        Ok(PyBytes::new(py, &state))
    }

    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        let (version, inner): (u8, NativeHoeffdingTreeClassifier) = deserialize(state.as_bytes())
            .map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("could not deserialize tree: {error}"))
        })?;
        if version != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported tree state version: {version}"
            )));
        }
        self.n_features = inner.n_features();
        self.n_classes = inner.n_classes();
        self.values = vec![f64::NAN; self.n_features];
        self.inner = inner;
        Ok(())
    }

    fn __getnewargs__(&self) -> (usize, usize) {
        (self.n_features, self.n_classes)
    }
}

impl RsNativeHoeffdingTreeClassifier {
    fn encode_mapping(
        &mut self,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        feature_names: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        self.values.fill(f64::NAN);
        for (feature, value) in x.iter() {
            let index = match feature_indices.get_item(&feature)? {
                Some(index) => index.extract::<usize>()?,
                None => match feature_names {
                    Some(feature_names) => {
                        let index = self.n_features;
                        feature_indices.set_item(&feature, index)?;
                        feature_names.append(&feature)?;
                        self.n_features += 1;
                        self.values.push(f64::NAN);
                        self.inner.add_features(1);
                        index
                    }
                    None => continue,
                },
            };
            if index >= self.n_features {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "feature index out of range",
                ));
            }
            let value = value.extract::<f64>()?;
            if value.is_infinite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "features must not contain infinite values",
                ));
            }
            self.values[index] = value;
        }
        Ok(())
    }

    fn validate_y(&self, y: usize) -> PyResult<()> {
        if y >= self.n_classes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected a class below {}, got {}",
                self.n_classes, y
            )));
        }
        Ok(())
    }

    fn validate_x(&self, x: &[f64]) -> PyResult<()> {
        if x.len() != self.n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} features, got {}",
                self.n_features,
                x.len()
            )));
        }
        validate_finite_features(x)?;
        Ok(())
    }

    fn validate_xy(&self, x: &[f64], y: usize) -> PyResult<()> {
        self.validate_x(x)?;
        self.validate_y(y)
    }
}

struct MondrianMapping {
    values: Vec<f64>,
    present: Vec<bool>,
    feature_order: Vec<usize>,
}

impl MondrianMapping {
    fn new(n_features: usize) -> Self {
        Self {
            values: vec![0.0; n_features],
            present: vec![false; n_features],
            feature_order: (0..n_features).collect(),
        }
    }

    fn encode(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        feature_names: Option<&Bound<'_, PyList>>,
    ) -> PyResult<usize> {
        self.present.fill(false);
        let previous_features = self.values.len();
        for (feature, value) in x.iter() {
            let index = match feature_indices.get_item(&feature)? {
                Some(index) => index.extract::<usize>()?,
                None => match feature_names {
                    Some(feature_names) => {
                        let index = self.values.len();
                        feature_indices.set_item(&feature, index)?;
                        feature_names.append(&feature)?;
                        self.values.push(0.0);
                        self.present.push(false);
                        index
                    }
                    None => continue,
                },
            };
            if index >= self.values.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "feature index out of range",
                ));
            }
            let value = value.extract::<f64>()?;
            if value.is_infinite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "features must not contain infinite values",
                ));
            }
            self.values[index] = value;
            self.present[index] = true;
        }
        let added = self.values.len() - previous_features;
        if added > 0 {
            let feature_names = feature_names.expect("feature names are present during discovery");
            let sorted = PyList::new(py, feature_names.iter())?;
            sorted.sort()?;
            self.feature_order = sorted
                .iter()
                .map(|feature| {
                    feature_indices
                        .get_item(feature)?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyKeyError::new_err("feature index is missing")
                        })?
                        .extract::<usize>()
                })
                .collect::<PyResult<_>>()?;
        }
        Ok(added)
    }
}

#[pyclass(
    name = "NativeMondrianTreeClassifier",
    module = "river._river_rust.tree"
)]
pub struct RsNativeMondrianTreeClassifier {
    inner: NativeMondrianTreeClassifier,
    mapping: MondrianMapping,
}

#[pymethods]
impl RsNativeMondrianTreeClassifier {
    #[new]
    #[pyo3(signature = (n_features, n_classes, seed=0))]
    fn new(n_features: usize, n_classes: usize, seed: u64) -> PyResult<Self> {
        if n_classes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_classes must be positive",
            ));
        }
        let inner = NativeMondrianTreeClassifier::seeded(n_features, n_classes, seed);
        Ok(Self {
            inner,
            mapping: MondrianMapping::new(n_features),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, feature_indices, feature_names, y, iteration, step, use_aggregation, dirichlet, split_pure, max_nodes=None))]
    fn learn_one_mapping(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        feature_names: &Bound<'_, PyList>,
        y: usize,
        iteration: usize,
        step: f64,
        use_aggregation: bool,
        dirichlet: f64,
        split_pure: bool,
        max_nodes: Option<usize>,
    ) -> PyResult<usize> {
        let added = self
            .mapping
            .encode(py, x, feature_indices, Some(feature_names))?;
        if added > 0 {
            self.inner.add_features(added);
        }
        if y >= self.inner.n_classes() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "class index out of range",
            ));
        }
        Ok(py.detach(|| {
            self.inner.learn(
                &self.mapping.values,
                &self.mapping.present,
                &self.mapping.feature_order,
                y,
                iteration,
                step,
                use_aggregation,
                dirichlet,
                split_pure,
                max_nodes,
            )
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_proba_mapping<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        output_labels: &Bound<'_, PyList>,
        output_indices: Vec<usize>,
        dirichlet: f64,
        use_aggregation: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.mapping.encode(py, x, feature_indices, None)?;
        let probabilities = py.detach(|| {
            self.inner.predict_proba(
                &self.mapping.values,
                &self.mapping.present,
                dirichlet,
                use_aggregation,
            )
        });
        if output_labels.len() != output_indices.len()
            || output_indices
                .iter()
                .any(|class_index| *class_index >= probabilities.len())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "class output order is invalid",
            ));
        }
        let output = PyDict::new(py);
        for (label, class_index) in output_labels.iter().zip(output_indices) {
            output.set_item(label, probabilities[class_index])?;
        }
        Ok(output)
    }

    fn add_class(&mut self) {
        self.inner.add_class();
    }

    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let state =
            serialize(&(1_u8, &self.inner, &self.mapping.feature_order)).map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "could not serialize tree: {error}"
                ))
            })?;
        Ok(PyBytes::new(py, &state))
    }

    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        let (version, inner, feature_order): (u8, NativeMondrianTreeClassifier, Vec<usize>) =
            deserialize(state.as_bytes()).map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "could not deserialize tree: {error}"
                ))
            })?;
        if version != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported tree state version: {version}"
            )));
        }
        self.mapping = MondrianMapping::new(inner.n_features());
        self.mapping.feature_order = feature_order;
        self.inner = inner;
        Ok(())
    }

    fn __getnewargs__(&self) -> (usize, usize) {
        (self.inner.n_features(), self.inner.n_classes())
    }
}

#[pyclass(
    name = "NativeMondrianTreeRegressor",
    module = "river._river_rust.tree"
)]
pub struct RsNativeMondrianTreeRegressor {
    inner: NativeMondrianTreeRegressor,
    mapping: MondrianMapping,
}

#[pymethods]
impl RsNativeMondrianTreeRegressor {
    #[new]
    #[pyo3(signature = (n_features, seed=0))]
    fn new(n_features: usize, seed: u64) -> PyResult<Self> {
        let inner = NativeMondrianTreeRegressor::seeded(n_features, seed);
        Ok(Self {
            inner,
            mapping: MondrianMapping::new(n_features),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, feature_indices, feature_names, y, iteration, step, use_aggregation, max_nodes=None))]
    fn learn_one_mapping(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        feature_names: &Bound<'_, PyList>,
        y: f64,
        iteration: usize,
        step: f64,
        use_aggregation: bool,
        max_nodes: Option<usize>,
    ) -> PyResult<usize> {
        let added = self
            .mapping
            .encode(py, x, feature_indices, Some(feature_names))?;
        if added > 0 {
            self.inner.add_features(added);
        }
        Ok(py.detach(|| {
            self.inner.learn(
                &self.mapping.values,
                &self.mapping.present,
                &self.mapping.feature_order,
                y,
                iteration,
                step,
                use_aggregation,
                max_nodes,
            )
        }))
    }

    fn predict_mapping(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyDict>,
        feature_indices: &Bound<'_, PyDict>,
        use_aggregation: bool,
    ) -> PyResult<f64> {
        self.mapping.encode(py, x, feature_indices, None)?;
        Ok(py.detach(|| {
            self.inner
                .predict(&self.mapping.values, &self.mapping.present, use_aggregation)
        }))
    }

    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let state =
            serialize(&(1_u8, &self.inner, &self.mapping.feature_order)).map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "could not serialize tree: {error}"
                ))
            })?;
        Ok(PyBytes::new(py, &state))
    }

    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        let (version, inner, feature_order): (u8, NativeMondrianTreeRegressor, Vec<usize>) =
            deserialize(state.as_bytes()).map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "could not deserialize tree: {error}"
                ))
            })?;
        if version != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported tree state version: {version}"
            )));
        }
        self.mapping = MondrianMapping::new(inner.n_features());
        self.mapping.feature_order = feature_order;
        self.inner = inner;
        Ok(())
    }

    fn __getnewargs__(&self) -> (usize,) {
        (self.inner.n_features(),)
    }
}

fn validate_tree_parameters(
    n_features: usize,
    n_classes: usize,
    max_features: Option<usize>,
    grace_period: f64,
    delta: f64,
    tau: f64,
    n_splits: usize,
    nb_threshold: f64,
    min_branch_fraction: f64,
    max_share_to_split: f64,
) -> PyResult<()> {
    if n_features == 0 || n_classes < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_features must be positive and n_classes must be at least two",
        ));
    }
    if max_features.is_some_and(|value| value == 0 || value > n_features) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_features must be between one and n_features",
        ));
    }
    if !grace_period.is_finite()
        || grace_period <= 0.0
        || !(0.0..1.0).contains(&delta)
        || !(0.0..=1.0).contains(&tau)
        || n_splits == 0
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grace_period and n_splits must be positive and delta and tau must be between zero and one",
        ));
    }
    if !nb_threshold.is_finite()
        || nb_threshold < 0.0
        || !(0.0..0.5).contains(&min_branch_fraction)
        || !(0.0..=1.0).contains(&max_share_to_split)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "nb_threshold must be non-negative, min_branch_fraction must be between zero and one half, and max_share_to_split must be between zero and one",
        ));
    }
    Ok(())
}

fn validate_weight(weight: f64) -> PyResult<()> {
    if !weight.is_finite() || weight <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample weights must be finite and greater than zero",
        ));
    }
    Ok(())
}

fn validate_gaussian_observation(value: f64, weight: f64) -> PyResult<()> {
    if !value.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "observations must be finite",
        ));
    }
    if !weight.is_finite() || weight < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_finite_features(values: &[f64]) -> PyResult<()> {
    if values.iter().any(|value| value.is_infinite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "features must not contain infinite values",
        ));
    }
    Ok(())
}

fn parse_split_criterion(value: &str) -> PyResult<SplitCriterion> {
    match value {
        "info_gain" => Ok(SplitCriterion::InfoGain),
        "gini" => Ok(SplitCriterion::Gini),
        "hellinger" => Ok(SplitCriterion::Hellinger),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown split criterion: {value}"
        ))),
    }
}

fn validate_split_criterion(criterion: SplitCriterion, n_classes: usize) -> PyResult<()> {
    if matches!(criterion, SplitCriterion::Hellinger) && n_classes != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the Hellinger split criterion requires exactly two classes",
        ));
    }
    Ok(())
}

fn parse_leaf_prediction(value: &str) -> PyResult<LeafPrediction> {
    match value {
        "mc" => Ok(LeafPrediction::MajorityClass),
        "nb" => Ok(LeafPrediction::NaiveBayes),
        "nba" => Ok(LeafPrediction::NaiveBayesAdaptive),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown leaf prediction: {value}"
        ))),
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
    register_submodule(py, m, "feature_hashing", register_feature_hashing)?;
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
    m.add_class::<RsGaussian>()?;
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
    m.add_class::<RsNativeHoeffdingTreeClassifier>()?;
    m.add_class::<RsNativeMondrianTreeClassifier>()?;
    m.add_class::<RsNativeMondrianTreeRegressor>()?;
    Ok(())
}

fn register_vectordict(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<VectorDict>()?;
    m.add_function(wrap_pyfunction!(euclidean_distance_dict, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean_distance_squared_dict, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean_distance_tuple, m)?)?;
    m.add_function(wrap_pyfunction!(lazy_search_euclidean, m)?)?;
    Ok(())
}

fn register_feature_hashing(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(feature_hash, m)?)?;
    Ok(())
}
