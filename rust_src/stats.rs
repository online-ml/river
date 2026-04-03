use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};
pub trait Univariate<F: Float + FromPrimitive + AddAssign + SubAssign> {
    fn update(&mut self, x: F);
    fn get(&self) -> F;
}

pub trait Bivariate<F: Float + FromPrimitive + AddAssign + SubAssign> {
    fn update(&mut self, x: F, y: F);
    fn get(&self) -> F;
}

pub trait Revertable<F: Float + FromPrimitive + AddAssign + SubAssign> {
    fn revert(&mut self, x: F) -> Result<(), &'static str>;
}

pub trait RollableUnivariate<F: Float + FromPrimitive + AddAssign + SubAssign>:
    Revertable<F> + Univariate<F>
{
}
