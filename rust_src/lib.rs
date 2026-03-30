// The purpose of this is to make the binding between watermill.rs and Python.
use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use watermill::{
    ewmean::EWMean, ewvariance::EWVariance, iqr::RollingIQR, iqr::IQR, kurtosis::Kurtosis,
    ptp::PeakToPeak, quantile::Quantile, quantile::RollingQuantile, skew::Skew, stats::Univariate,
};

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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
#[pyclass(module = "river.stats._rust_stats")]
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

/// A Python module implemented in Rust.
#[pymodule]
fn _rust_stats(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RsQuantile>()?;
    m.add_class::<RsEWMean>()?;
    m.add_class::<RsEWVar>()?;
    m.add_class::<RsIQR>()?;
    m.add_class::<RsKurtosis>()?;
    m.add_class::<RsPeakToPeak>()?;
    m.add_class::<RsSkew>()?;
    m.add_class::<RsRollingQuantile>()?;
    m.add_class::<RsRollingIQR>()?;
    Ok(())
}
