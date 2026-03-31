use crate::quantile::Quantile;
use crate::sorted_window::SortedWindow;

use crate::stats::Univariate;
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};
/// Computes the interquartile range.
/// # Arguments
/// * `q_inf` - Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
/// * `q_sup` -  Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.
/// # Examples
/// ```
/// use watermill::iqr::IQR;
/// use watermill::stats::Univariate;
/// let mut running_iqr: IQR<f64> = IQR::default();
/// for i in 1..=100{
///     running_iqr.update(i as f64);
/// }
/// assert_eq!(running_iqr.get(), 50.0);
/// ```
///
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IQR<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub q_inf: Quantile<F>,
    pub q_sup: Quantile<F>,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> IQR<F> {
    pub fn new(q_inf: F, q_sup: F) -> Result<Self, &'static str> {
        if q_inf >= q_sup {
            return Err("q_inf must be strictly less than q_sup");
        }

        Ok(Self {
            q_inf: Quantile::new(q_inf).unwrap(),
            q_sup: Quantile::new(q_sup).unwrap(),
        })
    }
}

impl<F> Default for IQR<F>
where
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            q_inf: Quantile::new(F::from_f64(0.25).unwrap()).unwrap(),
            q_sup: Quantile::new(F::from_f64(0.75).unwrap()).unwrap(),
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for IQR<F> {
    fn update(&mut self, x: F) {
        self.q_inf.update(x);
        self.q_sup.update(x);
    }
    fn get(&self) -> F {
        self.q_sup.get() - self.q_inf.get()
    }
}

/// Rolling interquartile range.
/// # Arguments
/// * `q_inf` - Desired inferior quantile, must be between 0 and 1.
/// * `q_sup` -  Desired superior quantile, must be between 0 and 1.
/// * `window_size` - Size of the rolling window.
/// # Examples
/// ```
/// use watermill::iqr::RollingIQR;
/// use watermill::stats::Univariate;
/// let mut rolling_iqr: RollingIQR<f64> = RollingIQR::new(0.25_f64, 0.75_f64, 101).unwrap();
/// for i in 0..=100{
///     rolling_iqr.update(i as f64);
///     //println!("{}", rolling_iqr.get());
///     rolling_iqr.get();
/// }
/// assert_eq!(rolling_iqr.get(), 50.0);
/// ```
///

#[derive(Serialize, Deserialize)]
pub struct RollingIQR<F: Float + FromPrimitive + AddAssign + SubAssign> {
    sorted_window: SortedWindow<F>,
    q_inf: F,
    q_sup: F,
    window_size: usize,
    lower_inf: usize,
    higher_inf: usize,
    frac_inf: F,
    lower_sup: usize,
    higher_sup: usize,
    frac_sup: F,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollingIQR<F> {
    pub fn new(q_inf: F, q_sup: F, window_size: usize) -> Result<Self, &'static str> {
        if F::from_f64(0.).unwrap() > q_inf && F::from_f64(1.).unwrap() < q_inf {
            return Err("q_inf should be betweek 0 and 1");
        }

        if F::from_f64(0.).unwrap() > q_sup && F::from_f64(1.).unwrap() < q_sup {
            return Err("q_sup should be betweek 0 and 1");
        }
        if q_inf >= q_sup {
            return Err("q_inf must be strictly less than q_sup");
        }

        let idx_inf = q_inf * (F::from_usize(window_size).unwrap() - F::from_f64(1.).unwrap());
        let lower_inf = idx_inf.floor().to_usize().unwrap();
        let mut higher_inf = lower_inf + 1;
        if higher_inf > window_size - 1 {
            higher_inf = lower_inf.saturating_sub(1); // Avoid attempt to subtract with overflow
        }

        let frac_inf = idx_inf - F::from_usize(lower_inf).unwrap();

        let idx_sup = q_sup * (F::from_usize(window_size).unwrap() - F::from_f64(1.).unwrap());
        let lower_sup = idx_sup.floor().to_usize().unwrap();
        let mut higher_sup = lower_sup + 1;
        if higher_sup > window_size - 1 {
            higher_sup = lower_sup.saturating_sub(1); // Avoid attempt to subtract with overflow
        }

        let frac_sup = idx_sup - F::from_usize(lower_sup).unwrap();
        Ok(Self {
            sorted_window: SortedWindow::new(window_size),
            q_inf,
            q_sup,
            window_size,
            lower_inf,
            higher_inf,
            frac_inf,
            lower_sup,
            higher_sup,
            frac_sup,
        })
    }
    fn prepare(&self, q: F, is_inf: bool) -> (usize, usize, F) {
        if self.sorted_window.len() < self.window_size {
            let idx =
                q * (F::from_usize(self.sorted_window.len()).unwrap() - F::from_f64(1.).unwrap());
            let lower = idx.floor().to_usize().unwrap();
            let mut higher = lower + 1;
            if higher > self.sorted_window.len() - 1 {
                higher = self.sorted_window.len().saturating_sub(1); // Avoid attempt to subtract with overflow
            }

            let frac = idx - F::from_usize(lower).unwrap();
            return (lower, higher, frac);
        }
        if is_inf {
            return (self.lower_inf, self.higher_inf, self.frac_inf);
        }
        (self.lower_sup, self.higher_sup, self.frac_sup)
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for RollingIQR<F> {
    fn update(&mut self, x: F) {
        self.sorted_window.push_back(x);
    }
    fn get(&self) -> F {
        let (lower_inf, higher_inf, frac_inf) = self.prepare(self.q_inf, true);
        let (lower_sup, higher_sup, frac_sup) = self.prepare(self.q_sup, false);

        let quantile_inf = self.sorted_window[lower_inf]
            + (self.sorted_window[higher_inf] - self.sorted_window[lower_inf]) * frac_inf;
        let quantile_sup = self.sorted_window[lower_sup]
            + (self.sorted_window[higher_sup] - self.sorted_window[lower_sup]) * frac_sup;

        quantile_sup - quantile_inf
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn rolling_iqr_edge_case() {
        use crate::iqr::RollingIQR;
        use crate::stats::Univariate;
        let mut rolling_iqr: RollingIQR<f64> = RollingIQR::new(0.99_f64, 1.0_f64, 1).unwrap();
        for i in 0..=1000 {
            rolling_iqr.update(i as f64);
            //println!("{}", rolling_iqr.get());
            rolling_iqr.get();
        }
        assert_eq!(rolling_iqr.get(), 0.0);
    }
}
