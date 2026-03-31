use crate::sorted_window::SortedWindow;
use crate::stats::Univariate;
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};
/// Running min.
/// # Examples
/// ```
/// use watermill::minimum::Min;
/// use watermill::stats::Univariate;
/// let mut running_min: Min<f64> = Min::new();
/// for i in 1..10{
///     running_min.update(i as f64);
/// }
/// assert_eq!(running_min.get(), 1.0);
/// ```
///
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Min<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub min: F,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Default for Min<F> {
    fn default() -> Self {
        Self {
            min: F::max_value(),
        }
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Min<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Min<F> {
    fn update(&mut self, x: F) {
        if self.min > x {
            self.min = x;
        }
    }
    fn get(&self) -> F {
        self.min
    }
}

/// Rolling min.
/// # Arguments
/// * `window_size` - Size of the rolling window.
/// # Examples
/// ```
/// use watermill::minimum::RollingMin;
/// use watermill::stats::Univariate;
/// let mut rolling_min: RollingMin<f64> = RollingMin::new(3);
/// for i in 1..10{
///     rolling_min.update(i as f64);
/// }
/// assert_eq!(rolling_min.get(), 7.0);
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct RollingMin<F: Float + FromPrimitive + AddAssign + SubAssign> {
    sorted_window: SortedWindow<F>,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollingMin<F> {
    pub fn new(window_size: usize) -> Self {
        Self {
            sorted_window: SortedWindow::new(window_size),
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for RollingMin<F> {
    fn update(&mut self, x: F) {
        self.sorted_window.push_back(x);
    }
    fn get(&self) -> F {
        self.sorted_window.front()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn min_default() {
        let mut min: Min<f64> = Min::default();
        min.update(1.0);
        assert_eq!(min.get(), 1.0);
    }
}
