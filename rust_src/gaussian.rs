use crate::count::Count;
use crate::mean::Mean;
use crate::stats::Univariate;
use crate::variance::Variance;
use libm::erf;
use serde::{Deserialize, Serialize};

const HALF_LOG_TAU: f64 = 0.918_938_533_204_672_7;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gaussian {
    distribution: Variance<f64>,
}

impl Gaussian {
    pub fn new(ddof: u32) -> Self {
        Self {
            distribution: Variance::new(ddof),
        }
    }

    pub fn from_state(n: f64, mean: f64, variance: f64, ddof: u32) -> Self {
        Self {
            distribution: Variance {
                mean: Mean {
                    mean,
                    n: Count { count: n },
                },
                ddof,
                state: (n - ddof as f64).max(0.0) * variance,
            },
        }
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64, weight: f64) {
        self.distribution.update_weighted(value, weight);
    }

    pub fn update_batch(&mut self, n: f64, mean: f64, state: f64) {
        if n == 0.0 {
            return;
        }
        let current_n = self.n_samples();
        if current_n == 0.0 {
            self.distribution.mean.mean = mean;
            self.distribution.mean.n.count = n;
            self.distribution.state = state;
            return;
        }
        let total = current_n + n;
        let delta = mean - self.mean();
        self.distribution.mean.mean += delta * n / total;
        self.distribution.mean.n.count = total;
        self.distribution.state += state + delta * delta * current_n * n / total;
    }

    #[inline(always)]
    pub fn revert(&mut self, value: f64, weight: f64) -> Result<(), &'static str> {
        self.distribution.revert_weighted(value, weight)
    }

    #[inline(always)]
    pub fn n_samples(&self) -> f64 {
        self.distribution.mean.n.get()
    }

    #[inline(always)]
    pub fn mean(&self) -> f64 {
        self.distribution.mean.get()
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        self.distribution.get()
    }

    #[inline(always)]
    pub fn standard_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline(always)]
    pub fn pdf(&self, value: f64) -> f64 {
        self.log_pdf(value).exp()
    }

    #[inline(always)]
    pub fn log_pdf(&self, value: f64) -> f64 {
        let variance = self.variance();
        if variance > 0.0 {
            (value - self.mean()).powi(2) / (-2.0 * variance) - HALF_LOG_TAU - 0.5 * variance.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    #[inline(always)]
    pub fn cdf(&self, value: f64) -> f64 {
        let standard_deviation = self.standard_deviation();
        if standard_deviation > 0.0 {
            0.5 * (1.0 + erf((value - self.mean()) / standard_deviation / std::f64::consts::SQRT_2))
        } else {
            0.0
        }
    }

    pub fn ddof(&self) -> u32 {
        self.distribution.ddof
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_and_revert_weighted_observations() {
        let mut gaussian = Gaussian::default();
        gaussian.update(1.0, 2.0);
        gaussian.update(3.0, 1.0);
        assert_eq!(gaussian.n_samples(), 3.0);
        assert!((gaussian.mean() - 5.0 / 3.0).abs() < 1e-12);
        assert!((gaussian.variance() - 4.0 / 3.0).abs() < 1e-12);
        gaussian.revert(1.0, 1.0).unwrap();
        assert_eq!(gaussian.n_samples(), 2.0);
        assert!((gaussian.mean() - 2.0).abs() < 1e-12);
        assert!((gaussian.variance() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn state_round_trip() {
        let gaussian = Gaussian::from_state(10.0, 2.0, 4.0, 1);
        assert_eq!(gaussian.n_samples(), 10.0);
        assert_eq!(gaussian.mean(), 2.0);
        assert_eq!(gaussian.variance(), 4.0);
        assert_eq!(gaussian.standard_deviation(), 2.0);
    }

    #[test]
    fn batch_update_matches_scalar_updates() {
        let mut batch = Gaussian::default();
        batch.update(1.0, 1.0);
        batch.update_batch(3.0, 3.0, 2.0);
        let mut scalar = Gaussian::default();
        for value in [1.0, 2.0, 3.0, 4.0] {
            scalar.update(value, 1.0);
        }
        assert!((batch.mean() - scalar.mean()).abs() < 1e-12);
        assert!((batch.variance() - scalar.variance()).abs() < 1e-12);
    }

    #[test]
    fn degenerate_distribution_has_zero_density() {
        let mut gaussian = Gaussian::default();
        gaussian.update(1.0, 1.0);
        assert_eq!(gaussian.pdf(1.0), 0.0);
        assert_eq!(gaussian.log_pdf(1.0), f64::NEG_INFINITY);
        assert_eq!(gaussian.cdf(1.0), 0.0);
    }
}
