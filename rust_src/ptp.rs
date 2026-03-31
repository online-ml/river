use crate::maximum::{Max, RollingMax};
use crate::minimum::{Min, RollingMin};
use crate::stats::Univariate;
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};
/// Running peak to peak (max - min).
/// # Examples
/// ```
/// use watermill::ptp::PeakToPeak;
/// use watermill::stats::Univariate;
/// let mut running_peak_to_peak: PeakToPeak<f64> = PeakToPeak::new();
/// for i in 1..10{
///     running_peak_to_peak.update(i as f64);
/// }
/// assert_eq!(running_peak_to_peak.get(), 8.0);
/// ```
///
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct PeakToPeak<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub min: Min<F>,
    pub max: Max<F>,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Default for PeakToPeak<F> {
    fn default() -> Self {
        Self {
            min: Min::new(),
            max: Max::new(),
        }
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> PeakToPeak<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for PeakToPeak<F> {
    fn update(&mut self, x: F) {
        self.min.update(x);
        self.max.update(x);
    }
    fn get(&self) -> F {
        self.max.get() - self.min.get()
    }
}

/// Rolling peak to peak (max - min).
/// # Arguments
/// * `window_size` - Size of the rolling window.
/// # Examples
/// ```
/// use watermill::ptp::RollingPeakToPeak;
/// use watermill::stats::Univariate;
/// let mut rolling_peak_to_peak: RollingPeakToPeak<f64> = RollingPeakToPeak::new(3);
/// for i in 1..10{
///     rolling_peak_to_peak.update(i as f64);
/// }
/// assert_eq!(rolling_peak_to_peak.get(), 2.0);
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct RollingPeakToPeak<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub min: RollingMin<F>,
    pub max: RollingMax<F>,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollingPeakToPeak<F> {
    pub fn new(window_size: usize) -> Self {
        Self {
            min: RollingMin::new(window_size),
            max: RollingMax::new(window_size),
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for RollingPeakToPeak<F> {
    fn update(&mut self, x: F) {
        self.min.update(x);
        self.max.update(x);
    }
    fn get(&self) -> F {
        self.max.get() - self.min.get()
    }
}
