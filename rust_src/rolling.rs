use crate::stats::{RollableUnivariate, Univariate};
use num::{Float, FromPrimitive};
use std::{
    collections::VecDeque,
    ops::{AddAssign, SubAssign},
};

/// Generic wrapper for performing rolling computations.
/// This can be wrapped around any struct which implements a `Univariate` and a `Revertable` and `RollableUnivariate`
/// traits.
/// Inputs to `update` are stored in a `VecDeque`. Elements of the queue are popped when the window is
//  full.
/// # Arguments
/// * `to_roll` - A running statistics which implements `Univariate` and `Revertable` and `RollableUnivariate` trait.
/// * `window_size` - Size of sliding window.
/// # Examples
/// ```
///
/// use watermill::stats::{RollableUnivariate, Univariate};
/// use watermill::sum::Sum;
/// use watermill::rolling::Rolling;
/// let data = vec![9.,7.,3.,2.,6.,1., 8., 5., 4.];
/// let mut running_sum: Sum<f64> = Sum::new();
/// // We wrap `running_sum` inside the `Rolling` struct.
/// let mut rolling_sum: Rolling<_, f64> = Rolling::new(&mut running_sum, 2).unwrap();
/// for x in data.iter(){
///     rolling_sum.update(*x as f64);
/// }
/// assert_eq!(rolling_sum.get(), 9.0);
/// ```
pub struct Rolling<'a, U, F>
where
    U: RollableUnivariate<F>,  // Optimization: Generic over U (the concrete type) instead of dyn for static dispatch
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    to_roll: &'a mut U,  // Optimization: &mut U instead of &mut dyn
    window_size: usize,
    window: VecDeque<F>,
}

impl<'a, U, F> Rolling<'a, U, F>
where
    U: RollableUnivariate<F>,
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    pub fn new(to_roll: &'a mut U, window_size: usize) -> Result<Self, &'static str> {  // Optimization: &'static str for error (clearer, no lifetime tie)
        if window_size == 0 {
            return Err("Window size should not equal to 0");
        }
        Ok(Self {
            to_roll,
            window_size,
            window: VecDeque::with_capacity(window_size),  // Optimization: Preallocate to avoid reallocs during growth
        })
    }
}

impl<'a, U, F> Univariate<F> for Rolling<'a, U, F>
where
    U: RollableUnivariate<F>,
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn update(&mut self, x: F) {
        if self.window.len() == self.window_size {
            // To handle the error, the program panics because returning the error type would change
            // the interface of the get method. This problem is unlikely to happen because we
            // control the size of the sliding window in the constructor.
            let oldest = self.window.front().copied().expect("Window should not be empty");  // Optimization: copied() for clarity/safety (F is Copy-like for floats); expect for debug assert
            match self.to_roll.revert(oldest) {
                Ok(()) => (),  // Assume revert returns Result<(), _>; adjust if different
                Err(err) => panic!("{}", err),
            };
            self.window.pop_front();
            self.window.push_back(x);
        } else {
            self.window.push_back(x);
        }
        self.to_roll.update(x);
    }

    fn get(&self) -> F {
        self.to_roll.get()
    }
}

mod tests {
    #[test]
    fn it_works() {
        use crate::rolling::Rolling;
        use crate::stats::Univariate;
        use crate::variance::Variance;
        let data = vec![9., 7., 3., 2., 6., 1., 8., 5., 4.];
        let mut running_var: Variance<f64> = Variance::default();
        // We wrap `running_var` inside the `Rolling` struct.
        let mut rolling_var: Rolling<_, f64> = Rolling::new(&mut running_var, 2).unwrap();  // Note: _ for type inference
        for x in data.iter() {
            rolling_var.update(*x as f64);
        }
        assert_eq!(rolling_var.get(), 0.5);
    }
}