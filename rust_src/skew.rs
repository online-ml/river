use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::moments::CentralMoments;
use crate::stats::Univariate;
use serde::{Deserialize, Serialize};
/// Running Skew.
/// # Arguments
/// * `bias` - If `false`, then the calculations are corrected for statistical bias.
/// # Examples
/// ```
/// use river::skew::Skew;
/// use river::stats::Univariate;
/// let data: Vec<f64> = vec![ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,-0.23413696];
/// let mut running_skew: Skew<f64> = Skew::default();
/// for x in data.iter(){
///     running_skew.update(*x);
///     println!("Skew value: {}", running_skew.get());
/// }
/// assert_eq!(running_skew.get(), 1.056115635439031);
/// ```
/// With bias enabled.
/// ```
/// use river::skew::Skew;
/// use river::stats::Univariate;
/// let data: Vec<f64> = vec![ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,-0.23413696];
/// let mut running_skew: Skew<f64> = Skew::new(true);
/// for x in data.iter(){
///     running_skew.update(*x);
///     println!("Skew with bias value: {}", running_skew.get());
/// }
/// assert_eq!(running_skew.get(), 0.771277809151813);
/// ```
/// # References
/// [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Skew<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub central_moments: CentralMoments<F>,
    pub bias: bool,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Skew<F> {
    pub fn new(bias: bool) -> Self {
        Self {
            central_moments: CentralMoments::new(),
            bias,
        }
    }
}

impl<F> Default for Skew<F>
where
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn default() -> Self {
        Self {
            central_moments: CentralMoments::new(),
            bias: false,
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Skew<F> {
    #[inline(always)]
    fn update(&mut self, x: F) {
        self.central_moments.count.update(x);
        self.central_moments.update_delta(x);
        self.central_moments.update_m1(x);
        self.central_moments.update_sum_delta();
        self.central_moments.update_m3();
        self.central_moments.update_m2();
    }
    #[inline(always)]
    fn get(&self) -> F {
        let n = self.central_moments.count.get();
        let _0 = F::from_f64(0.).unwrap();

        let mut skew: F = _0;
        if self.central_moments.m2 != _0 {
            let m2 = self.central_moments.m2;
            skew = n.sqrt() * self.central_moments.m3
                / (m2 * m2.sqrt());
        }
        let _1 = F::from_f64(1.).unwrap();
        let _2 = F::from_f64(2.).unwrap();
        if (!self.bias) && n > _2 {
            return ((n - _1) * n).sqrt()
                / (n - _2)
                * skew;
        }
        skew
    }
}
