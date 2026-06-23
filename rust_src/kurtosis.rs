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
/// use river::kurtosis::Kurtosis;
/// use river::stats::Univariate;
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
/// use river::kurtosis::Kurtosis;
/// use river::stats::Univariate;
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
    #[inline(always)]
    fn update(&mut self, x: F) {
        self.central_moments.count.update(x);
        self.central_moments.update_delta(x);
        self.central_moments.update_m1(x);
        self.central_moments.update_sum_delta();
        self.central_moments.update_m4();
        self.central_moments.update_m3();
        self.central_moments.update_m2();
    }
    #[inline(always)]
    fn get(&self) -> F {
        let n = self.central_moments.count.get();
        let _0 = F::from_f64(0.).unwrap();
        let _1 = F::from_f64(1.).unwrap();
        let _2 = F::from_f64(2.).unwrap();
        let _3 = F::from_f64(3.).unwrap();

        let mut kurtosis: F = _0;
        if self.central_moments.m2 != _0 {
            let m2 = self.central_moments.m2;
            kurtosis = kurtosis + n * self.central_moments.m4
                / (m2 * m2);
        }
        if (!self.bias) && n > _3 {
            let nm1 = n - _1;
            return _1
                / (n - _2)
                / (n - _3)
                * ((n * n - _1) * kurtosis
                    - _3 * nm1 * nm1);
        }
        kurtosis - _3
    }
}
