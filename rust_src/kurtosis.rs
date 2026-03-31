use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::moments::CentralMoments;
use crate::stats::Univariate;
use serde::{Deserialize, Serialize};
/// Running Kurtosis.
/// # Arguments
/// * `bias` - If `false`, then the calculations are corrected for statistical bias.
/// # Examples
/// ```
/// use watermill::kurtosis::Kurtosis;
/// use watermill::stats::Univariate;
/// let data: Vec<f64> = vec![ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,-0.23413696];
/// let mut running_kurtosis: Kurtosis<f64> = Kurtosis::default();
/// for x in data.iter(){
///     running_kurtosis.update(*x);
///     println!("Kurtosis value: {}", running_kurtosis.get());
/// }
/// assert_eq!(running_kurtosis.get(), 0.46142635465045007);
/// ```
/// With bias enabled.
/// ```
/// use watermill::kurtosis::Kurtosis;
/// use watermill::stats::Univariate;
/// let data: Vec<f64> = vec![ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,-0.23413696];
/// let mut running_kurtosis: Kurtosis<f64> = Kurtosis::new(true);
/// for x in data.iter(){
///     running_kurtosis.update(*x);
///     println!("Kurtosis with bias value: {}", running_kurtosis.get());
/// }
/// assert_eq!(running_kurtosis.get(), -0.6989395355484169);
/// ```
/// # References
/// [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Kurtosis<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub bias: bool,
    pub central_moments: CentralMoments<F>,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Kurtosis<F> {
    pub fn new(bias: bool) -> Self {
        Self {
            central_moments: CentralMoments::new(),
            bias,
        }
    }
}

impl<F> Default for Kurtosis<F>
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

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Kurtosis<F> {
    fn update(&mut self, x: F) {
        self.central_moments.count.update(x);
        self.central_moments.update_delta(x);
        self.central_moments.update_m1(x);
        self.central_moments.update_sum_delta();
        self.central_moments.update_m4();
        self.central_moments.update_m3();
        self.central_moments.update_m2();
    }
    fn get(&self) -> F {
        let n = self.central_moments.count.get();
        let mut kurtosis: F = F::from_f64(0.).unwrap();
        if self.central_moments.m2 != F::from_f64(0.).unwrap() {
            kurtosis += n * self.central_moments.m4
                / self.central_moments.m2.powf(F::from_f64(2.).unwrap());
        }
        if (!self.bias) && n > F::from_f64(3.).unwrap() {
            return F::from_f64(1.).unwrap()
                / (n - F::from_f64(2.).unwrap())
                / (n - F::from_f64(3.).unwrap())
                * ((n.powf(F::from_f64(2.).unwrap()) - F::from_f64(1.).unwrap()) * kurtosis
                    - F::from_f64(3.).unwrap()
                        * (n - F::from_f64(1.).unwrap()).powf(F::from_f64(2.).unwrap()));
        }
        kurtosis - F::from_f64(3.).unwrap()
    }
}
