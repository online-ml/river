use crate::sorted_window::SortedWindow;
use crate::stats::Univariate;
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};
/// Running max.
/// # Examples
/// ```
/// use watermill::maximum::Max;
/// use watermill::stats::Univariate;
/// let mut running_max: Max<f64> = Max::new();
/// for i in 1..10{
///     running_max.update(i as f64);
/// }
/// assert_eq!(running_max.get(), 9.0);
/// ```
///
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Max<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub max: F,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Default for Max<F> {
    fn default() -> Self {
        Self {
            max: F::min_value(),
        }
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Max<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Max<F> {
    fn update(&mut self, x: F) {
        if self.max < x {
            self.max = x;
        }
    }
    fn get(&self) -> F {
        self.max
    }
}

/// Running absolute max.
/// # Examples
/// ```
/// use watermill::maximum::AbsMax;
/// use watermill::stats::Univariate;
/// let mut running_abs_max: AbsMax<f64> = AbsMax::new();
/// for i in -17..10{
///     running_abs_max.update(i as f64);
/// }
/// assert_eq!(running_abs_max.get(), 17.0);
/// ```
///
#[derive(Debug)]
pub struct AbsMax<F: Float + FromPrimitive + AddAssign + SubAssign> {
    abs_max: F,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Default for AbsMax<F> {
    fn default() -> Self {
        Self {
            abs_max: F::from_f64(0.0).unwrap(),
        }
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> AbsMax<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for AbsMax<F> {
    fn update(&mut self, x: F) {
        if self.abs_max < x.abs() {
            self.abs_max = x.abs();
        }
    }
    fn get(&self) -> F {
        self.abs_max
    }
}

/// Rolling max.
/// # Arguments
/// * `window_size` - Size of the rolling window.
/// # Examples
/// ```
/// use watermill::maximum::RollingMax;
/// use watermill::stats::Univariate;
/// let mut rolling_max: RollingMax<f64> = RollingMax::new(3);
/// for i in 1..10{
///     rolling_max.update(i as f64);
/// }
/// assert_eq!(rolling_max.get(), 9.0);
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct RollingMax<F: Float + FromPrimitive + AddAssign + SubAssign> {
    sorted_window: SortedWindow<F>,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollingMax<F> {
    pub fn new(window_size: usize) -> Self {
        Self {
            sorted_window: SortedWindow::new(window_size),
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for RollingMax<F> {
    fn update(&mut self, x: F) {
        self.sorted_window.push_back(x);
    }
    fn get(&self) -> F {
        self.sorted_window.back()
    }
}
