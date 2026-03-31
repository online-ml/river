use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::stats::{Revertable, RollableUnivariate, Univariate};
use serde::{Deserialize, Serialize};
/// Running count.
/// # Examples
/// ```
/// use watermill::stats::{Univariate, Revertable};
/// use watermill::count::Count;
/// let mut running_count: Count<f64> = Count::new();
/// for i in 1..10{
///     running_count.update(i as f64);
/// }
/// assert_eq!(running_count.get(), 9.0);
///
/// // You can revert the count
///
/// for i in (1..10).rev(){
///     running_count.revert(i as f64);
/// }
/// assert_eq!(running_count.get(), 0.);
///```
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Count<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub count: F,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Default for Count<F> {
    fn default() -> Self {
        Self {
            count: F::from_f64(0.0).unwrap(),
        }
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Count<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Count<F> {
    #[warn(unused_variables)]
    fn update(&mut self, _x: F) {
        self.count += F::from_f64(1.).unwrap();
    }
    fn get(&self) -> F {
        self.count
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Revertable<F> for Count<F> {
    fn revert(&mut self, _x: F) -> std::result::Result<(), &'static str> {
        if self.count == F::from_f64(0.).unwrap() {
            return Err("Count cannot go below 0");
        }
        self.count -= F::from_f64(1.).unwrap();
        Ok(())
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollableUnivariate<F> for Count<F> {}
