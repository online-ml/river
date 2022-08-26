use bincode::{deserialize, serialize};
use online_statistics::{
    ewmean::EWMean, ewvariance::EWVariance, iqr::IQR, kurtosis::Kurtosis, ptp::PeakToPeak,
    quantile::Quantile, quantile::RollingQuantile, skew::Skew, stats::Univariate,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PyQuantile {
    pub quantile: Quantile<f64>,
}

#[pymethods]
impl PyQuantile {
    #[new]
    pub fn new(q: Option<f64>) -> PyQuantile {
        match q {
            Some(q) => {
                return PyQuantile {
                    quantile: Quantile::new(q).expect("q should between 0 and 1"),
                }
            }
            None => PyQuantile {
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
}

#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PyEWMean {
    ewmean: EWMean<f64>,
}
#[pymethods]
impl PyEWMean {
    #[new]
    pub fn new(alpha: f64) -> PyEWMean {
        PyEWMean {
            ewmean: EWMean::new(alpha),
        }
    }
    pub fn update(&mut self, x: f64) {
        self.ewmean.update(x);
    }
    pub fn get(&self) -> f64 {
        self.ewmean.get()
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PyEWVar {
    ewvar: EWVariance<f64>,
    alpha: f64,
}
#[pymethods]
impl PyEWVar {
    #[new]
    pub fn new(alpha: f64) -> PyEWVar {
        PyEWVar {
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

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64,)> {
        Ok((self.alpha,))
    }
}

#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PyIQR {
    pub iqr: IQR<f64>,
}

#[pymethods]
impl PyIQR {
    #[new]
    pub fn new(q_inf: f64, q_sup: f64) -> PyIQR {
        PyIQR {
            iqr: IQR::new(q_inf, q_sup).expect("TODO"),
        }
    }
    pub fn update(&mut self, x: f64) {
        self.iqr.update(x);
    }
    pub fn get(&self) -> f64 {
        self.iqr.get()
    }
}

#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PyKurtosis {
    kurtosis: Kurtosis<f64>,
}
#[pymethods]
impl PyKurtosis {
    #[new]
    pub fn new(bias: bool) -> PyKurtosis {
        PyKurtosis {
            kurtosis: Kurtosis::new(bias),
        }
    }
    pub fn update(&mut self, x: f64) {
        self.kurtosis.update(x);
    }
    pub fn get(&self) -> f64 {
        self.kurtosis.get()
    }
}

#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PyPeakToPeak {
    ptp: PeakToPeak<f64>,
}

#[pymethods]
impl PyPeakToPeak {
    #[new]
    pub fn new() -> PyPeakToPeak {
        PyPeakToPeak {
            ptp: PeakToPeak::new(),
        }
    }

    pub fn update(&mut self, x: f64) {
        self.ptp.update(x);
    }
    pub fn get(&self) -> f64 {
        self.ptp.get()
    }
}

#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PySkew {
    skew: Skew<f64>,
}
#[pymethods]
impl PySkew {
    #[new]
    pub fn new(bias: bool) -> PySkew {
        PySkew {
            skew: Skew::new(bias),
        }
    }
    pub fn update(&mut self, x: f64) {
        self.skew.update(x);
    }
    pub fn get(&self) -> f64 {
        self.skew.get()
    }
}

#[pyclass(module = "river_rust_stats.river_rust_stats")]
pub struct PyRollingQuantile {
    stat: RollingQuantile<f64>,
}

#[pymethods]
impl PyRollingQuantile {
    #[new]
    pub fn new(q: f64, window_size: usize) -> PyRollingQuantile {
        PyRollingQuantile {
            stat: RollingQuantile::new(q, window_size).unwrap(),
        }
    }
    pub fn update(&mut self, x: f64) {
        self.stat.update(x);
    }
    pub fn get(&self) -> f64 {
        self.stat.get()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn river_rust_stats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyQuantile>()?;
    m.add_class::<PyEWMean>()?;
    m.add_class::<PyEWVar>()?;
    m.add_class::<PyIQR>()?;
    m.add_class::<PyKurtosis>()?;
    m.add_class::<PyPeakToPeak>()?;
    m.add_class::<PySkew>()?;
    m.add_class::<PyRollingQuantile>()?;
    Ok(())
}
