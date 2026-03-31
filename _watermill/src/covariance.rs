use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::mean::Mean;
use crate::stats::{Bivariate, Univariate};
use serde::{Deserialize, Serialize};
/// Running Covariance.
/// # Examples
/// ```
/// use watermill::covariance::Covariance;
/// use watermill::stats::Bivariate;
/// let mut running_cov: Covariance<f64> = Covariance::default();
/// let x: Vec<f64> = vec![-2.1,  -1.,  4.3];
/// let y: Vec<f64> = vec![3., 1.1, 0.12];
/// for (xi, yi) in x.iter().zip(y.iter()){
///     running_cov.update(*xi,*yi);
/// }
/// assert_eq!(running_cov.get(), -4.286);
/// ```
/// # References
/// [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
///
/// [^2]: Schubert, E. and Gertz, M., 2018, July. Numerically stable parallel computation of (co-) variance. In Proceedings of the 30th International Conference on Scientific and Statistical Database Management (pp. 1-12).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Covariance<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub ddof: u32,
    pub mean_x: Mean<F>,
    pub mean_y: Mean<F>,
    c: F,
    pub cov: F,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Covariance<F> {
    pub fn new(ddof: u32) -> Self {
        Self {
            mean_x: Mean::new(),
            mean_y: Mean::new(),
            ddof,
            c: F::from_f64(0.).unwrap(),
            cov: F::from_f64(0.).unwrap(),
        }
    }
}

impl<F> Default for Covariance<F>
where
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            ddof: 1,
            mean_x: Mean::new(),
            mean_y: Mean::new(),
            c: F::from_f64(0.).unwrap(),
            cov: F::from_f64(0.).unwrap(),
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Bivariate<F> for Covariance<F> {
    fn update(&mut self, x: F, y: F) {
        let dx = x - self.mean_x.get();
        self.mean_x.update(x);
        self.mean_y.update(y);
        self.c += dx * (y - self.mean_y.get());
        self.cov = self.c
            / (F::from_f64(1.)
                .unwrap()
                .max(self.mean_x.n.get() - F::from_u32(self.ddof).unwrap()));
    }
    fn get(&self) -> F {
        self.cov
    }
}
