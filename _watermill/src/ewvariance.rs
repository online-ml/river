use crate::ewmean::EWMean;
use crate::stats::Univariate;
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};

/// Exponentially weighted variance.
/// # Arguments
/// * `alpha` - The closer `alpha` is to 1 the more the statistic will adapt to recent values. Default value is `0.5`.
/// # Examples
/// ```
/// use watermill::ewvariance::EWVariance;
/// use watermill::stats::Univariate;
/// let mut running_ewvariance: EWVariance<f64> = EWVariance::default();
/// let data = vec![1., 3., 5., 4., 6., 8., 7., 9., 11.];
/// for i in data.iter(){
///     running_ewvariance.update(*i as f64);
/// }
/// assert_eq!(running_ewvariance.get(),  3.56536865234375);
/// ```
/// # References
/// [^1]: [Finch, T., 2009. Incremental calculation of weighted Var and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
///
/// [^2]: [Exponential Moving Average on Streaming Data](https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EWVariance<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub mean: EWMean<F>,
    pub sq_mean: EWMean<F>,
    pub alpha: F,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> EWVariance<F> {
    pub fn new(alpha: F) -> Self {
        Self {
            mean: EWMean::new(alpha),
            sq_mean: EWMean::new(alpha),
            alpha,
        }
    }
}

impl<F> Default for EWVariance<F>
where
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn default() -> Self {
        let alpha = F::from_f64(0.5).unwrap();
        Self {
            mean: EWMean::new(alpha),
            sq_mean: EWMean::new(alpha),
            alpha,
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for EWVariance<F> {
    fn update(&mut self, x: F) {
        self.mean.update(x);
        self.sq_mean.update(x.powf(F::from_i8(2).unwrap()))
    }
    fn get(&self) -> F {
        self.sq_mean.get() - self.mean.get().powf(F::from_i8(2).unwrap())
    }
}
