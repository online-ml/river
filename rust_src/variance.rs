use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::mean::Mean;
use crate::stats::{Revertable, RollableUnivariate, Univariate};
use serde::{Deserialize, Serialize};
/// Running variance using Belford Algorithm.
/// # Arguments
/// * `ddof` - Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n` represents the number of seen elements.
/// # Examples
/// ```
/// use watermill::variance::Variance;
/// use watermill::stats::{Univariate, Revertable};
/// let data: Vec<f64> = vec![3., 5., 4., 7., 10., 12.];
/// let data_revert = data.clone();
/// let mut running_variance: Variance<f64> = Variance::default();
/// for x in data.into_iter(){
///     running_variance.update(x);
/// }
/// assert_eq!(running_variance.get(), 12.566666666666668);
/// // You can revert the variance
///
/// for x in data_revert.into_iter().rev(){
///     running_variance.revert(x);
/// }
/// assert_eq!(running_variance.get(), 0.);
/// ```
/// # References
/// [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
///
/// [^2]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Variance<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub mean: Mean<F>,
    pub ddof: u32,
    pub state: F,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Variance<F> {
    pub fn new(ddof: u32) -> Self {
        Self {
            mean: Mean::new(),
            ddof,
            state: F::from_f64(0.).unwrap(),
        }
    }
}

impl<F> Default for Variance<F>
where
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            mean: Mean::new(),
            ddof: 1,
            state: F::from_f64(0.).unwrap(),
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Variance<F> {
    fn update(&mut self, x: F) {
        let mean_old = self.mean.get();
        self.mean.update(x);
        let mean_new = self.mean.get();
        self.state += (x - mean_old) * (x - mean_new);
    }
    fn get(&self) -> F {
        let mean_n = self.mean.n.get();
        if mean_n > F::from_u32(self.ddof).unwrap() {
            return self.state / (mean_n - F::from_u32(self.ddof).unwrap());
        }
        F::from_f64(0.).unwrap()
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Revertable<F> for Variance<F> {
    fn revert(&mut self, x: F) -> Result<(), &'static str> {
        let mean_old = self.mean.get();
        self.mean.revert(x)?;
        let mean_new = self.mean.get();
        self.state -= (x - mean_old) * (x - mean_new);
        Ok(())
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollableUnivariate<F> for Variance<F> {}
