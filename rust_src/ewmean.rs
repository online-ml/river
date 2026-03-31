use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::stats::Univariate;
use serde::{Deserialize, Serialize};
/// Exponentially weighted mean.
/// # Arguments
/// * `alpha` - The closer `alpha` is to 1 the more the statistic will adapt to recent values. Default value is `0.5`.
/// # Examples
/// ```
/// use watermill::ewmean::EWMean;
/// use watermill::stats::Univariate;
/// let mut running_ewmean: EWMean<f64> = EWMean::default();
/// let data = vec![1., 3., 5., 4., 6., 8., 7., 9., 11.];
/// for i in data.iter(){
///     running_ewmean.update(*i as f64);
/// }
/// assert_eq!(running_ewmean.get(), 9.4296875);
/// ```
/// # References
/// [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
///
/// [^2]: [Exponential Moving Average on Streaming Data](https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EWMean<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub mean: F,
    pub alpha: F,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> EWMean<F> {
    pub fn new(alpha: F) -> Self {
        Self {
            mean: F::from_f64(0.0).unwrap(),
            alpha,
        }
    }
}

impl<F> Default for EWMean<F>
where
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            mean: F::from_f64(0.).unwrap(),
            alpha: F::from_f64(0.5).unwrap(),
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for EWMean<F> {
    fn update(&mut self, x: F) {
        if self.mean == F::from_f64(0.).unwrap() {
            self.mean = x;
        } else {
            self.mean = self.alpha * x + (F::from_f64(1.).unwrap() - self.alpha) * self.mean;
        }
    }
    fn get(&self) -> F {
        self.mean
    }
}
