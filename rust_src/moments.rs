use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::count::Count;
use crate::stats::Univariate;
use serde::{Deserialize, Serialize};
/// Computes central moments using Welford's algorithm.
/// # References
/// [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CentralMoments<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub delta: F,
    pub sum_delta: F,
    /// Mean of sum of differences.
    pub m1: F,
    /// Sums of powers of differences from the mean order 1.
    pub m2: F,
    /// Sums of powers of differences from the mean order 2.
    pub m3: F,
    /// Sums of powers of differences from the mean order 3.
    pub m4: F,
    /// Sums of powers of differences from the mean order 4.
    pub count: Count<F>,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Default for CentralMoments<F> {
    fn default() -> Self {
        let _0 = F::from_f64(0.).unwrap();
        Self {
            delta: _0,
            sum_delta: _0,
            m1: _0,
            m2: _0,
            m3: _0,
            m4: _0,
            count: Count::new(),
        }
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> CentralMoments<F> {
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn update_delta(&mut self, x: F) {
        self.delta = (x - self.sum_delta) / self.count.get()
    }
    #[inline(always)]
    pub fn update_sum_delta(&mut self) {
        self.sum_delta += self.delta
    }
    #[inline(always)]
    pub fn update_m1(&mut self, x: F) {
        self.m1 = (x - self.sum_delta) * self.delta * (self.count.get() - F::from_f64(1.).unwrap())
    }
    #[inline(always)]
    pub fn update_m2(&mut self) {
        self.m2 += self.m1
    }
    #[inline(always)]
    pub fn update_m3(&mut self) {
        self.m3 += self.m1 * self.delta * (self.count.get() - F::from_f64(2.).unwrap())
            - F::from_f64(3.).unwrap() * self.delta * self.m2
    }
    #[inline(always)]
    pub fn update_m4(&mut self) {
        let delta_square = self.delta * self.delta;
        let n = self.count.get();
        self.m4 += self.m1
            * delta_square
            * (n * n
                - F::from_f64(3.).unwrap() * n
                + F::from_f64(3.).unwrap())
            + F::from_f64(6.).unwrap() * delta_square * self.m2
            - F::from_f64(4.).unwrap() * self.delta * self.m3
    }
}
