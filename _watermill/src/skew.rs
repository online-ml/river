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
/// use watermill::skew::Skew;
/// use watermill::stats::Univariate;
/// let data: Vec<f64> = vec![ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,-0.23413696];
/// let mut running_skew: Skew<f64> = Skew::default();
/// for x in data.iter(){
///     running_skew.update(*x);
///     println!("Skew value: {}", running_skew.get());
/// }
/// assert_eq!(running_skew.get(), 1.0561156354390309);
/// ```
/// With bias enabled.
/// ```
/// use watermill::skew::Skew;
/// use watermill::stats::Univariate;
/// let data: Vec<f64> = vec![ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,-0.23413696];
/// let mut running_skew: Skew<f64> = Skew::new(true);
/// for x in data.iter(){
///     running_skew.update(*x);
///     println!("Skew with bias value: {}", running_skew.get());
/// }
/// assert_eq!(running_skew.get(),0.7712778091518129);
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
    fn update(&mut self, x: F) {
        self.central_moments.count.update(x);
        self.central_moments.update_delta(x);
        self.central_moments.update_m1(x);
        self.central_moments.update_sum_delta();
        self.central_moments.update_m3();
        self.central_moments.update_m2();
    }
    fn get(&self) -> F {
        let n = self.central_moments.count.get();

        let mut skew: F = F::from_f64(0.).unwrap();
        if self.central_moments.m2 != F::from_f64(0.).unwrap() {
            skew += n.powf(F::from_f64(0.5).unwrap()) * self.central_moments.m3
                / self.central_moments.m2.powf(F::from_f64(1.5).unwrap());
        }
        if (!self.bias) && n > F::from_f64(2.).unwrap() {
            return ((n - F::from_f64(1.).unwrap()) * n).powf(F::from_f64(0.5).unwrap())
                / (n - F::from_f64(2.).unwrap())
                * skew;
        }
        skew
    }
}
