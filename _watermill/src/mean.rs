use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::count::Count;
use crate::stats::{Revertable, RollableUnivariate, Univariate};
use serde::{Deserialize, Serialize};

/// Running mean.
/// # Examples
/// ```
/// use watermill::mean::Mean;
/// use watermill::stats::{Univariate, Revertable};
/// let mut running_mean: Mean<f64> = Mean::new();
/// for i in 0..10{
///     running_mean.update(i as f64);
/// }
/// assert_eq!(running_mean.get(), 4.5);
///
/// // You can revert the mean
/// for i in (0..10).rev(){
///     running_mean.revert(i as f64);
/// }
/// assert_eq!(running_mean.get(), 0.);
/// ```
/// # References
/// [^1]: [West, D. H. D. (1979). Updating mean and variance estimates: An improved method. Communications of the ACM, 22(9), 532-535.](https://dl.acm.org/doi/10.1145/359146.359153)
///
/// [^2]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
///
/// [^3]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Mean<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub mean: F,
    pub n: Count<F>,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Default for Mean<F> {
    fn default() -> Self {
        Self {
            mean: F::from_f64(0.0).unwrap(),
            n: Count::new(),
        }
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Mean<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Mean<F> {
    fn update(&mut self, x: F) {
        self.n.update(x);
        self.mean += (F::from_f64(1.).unwrap() / self.n.get()) * (x - self.mean);
    }
    fn get(&self) -> F {
        self.mean
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Revertable<F> for Mean<F> {
    fn revert(&mut self, x: F) -> Result<(), &'static str> {
        self.n.revert(x)?;

        let count = self.n.get();
        if count == F::from_f64(0.).unwrap() {
            self.mean = F::from_f64(0.0).unwrap();
        } else {
            self.mean -= (F::from_f64(1.0).unwrap() / count) * (x - self.mean);
        }
        Ok(())
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollableUnivariate<F> for Mean<F> {}
