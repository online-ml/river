use num::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::count::Count;
use crate::ewmean::EWMean;
use crate::ewvariance::EWVariance;
use crate::iqr::IQR;
use crate::kurtosis::Kurtosis;
use crate::maximum::{AbsMax, Max};
use crate::mean::Mean;
use crate::minimum::Min;
use crate::ptp::PeakToPeak;
use crate::quantile::Quantile;
use crate::skew::Skew;
use crate::stats::Univariate;
use crate::sum::Sum;
use crate::variance::Variance;

#[doc(hidden)]
pub struct IterStat<U, I>
where
    U: Univariate<I::Item>,
    I: Iterator,
    I::Item: Float + FromPrimitive + AddAssign + SubAssign + 'static,
{
    stat: U,
    underlying: I,
}

impl<U, I> Iterator for IterStat<U, I>
where
    U: Univariate<I::Item>,
    I: Iterator,
    I::Item: Float + FromPrimitive + AddAssign + SubAssign,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.underlying.next() {
            self.stat.update(x);
            return Some(self.stat.get());
        }
        None
    }
}

pub trait IterStatisticsExtend: Iterator {
    /// Running sum.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3.];
    /// let vec_true: Vec<f64> = vec![1., 3., 6.];
    /// for (d, t) in data.into_iter().online_sum().zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_sum(self) -> IterStat<Sum<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Sum::new(),
            underlying: self,
        }
    }
    /// Running mean.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3.];
    /// let vec_true: Vec<f64> = vec![1., 1.5, 2.];
    /// for (d, t) in data.into_iter().online_mean().zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_mean(self) -> IterStat<Mean<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Mean::new(),
            underlying: self,
        }
    }
    /// Running count.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3.];
    /// let vec_true: Vec<f64> = vec![1., 2., 3.];
    /// for (d, t) in data.into_iter().online_count().zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_count(self) -> IterStat<Count<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Count::new(),
            underlying: self,
        }
    }

    /// Running exponentially weighted mean.
    /// # Arguments
    /// * `alpha` - The closer `alpha` is to 1 the more the statistic will adapt to recent values. Default value is `0.5`.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3.];
    /// let vec_true: Vec<f64> = vec![1., 1.9, 2.89];
    /// for (d, t) in data.into_iter().online_ewmean(0.9_f64).zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_ewmean(self, alpha: Self::Item) -> IterStat<EWMean<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: EWMean::new(alpha),
            underlying: self,
        }
    }
    /// Running exponentially weighted variance.
    /// # Arguments
    /// * `alpha` - The closer `alpha` is to 1 the more the statistic will adapt to recent values. Default value is `0.5`.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3.];
    /// let vec_true: Vec<f64> = vec![0., 0.0900000000000003, 0.11789999999999878];
    /// for (d, t) in data.into_iter().online_ewvar(0.9_f64).zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_ewvar(self, alpha: Self::Item) -> IterStat<EWVariance<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: EWVariance::new(alpha),
            underlying: self,
        }
    }
    /// Running inter quantile range.
    /// # Arguments
    /// * `q_inf` - Desired inferior quantile, must be between 0 and 1. Defaults to `0.25`.
    /// * `q_sup` -  Desired superior quantile, must be between 0 and 1. Defaults to `0.75`.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., 4.];
    /// let vec_true: Vec<f64> = vec![0., 1., 2., 2.];
    /// for (d, t) in data.into_iter().online_iqr(0.25_f64, 0.75_f64).zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_iqr(self, q_inf: Self::Item, q_sup: Self::Item) -> IterStat<IQR<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: IQR::new(q_inf, q_sup).expect("q_inf must be strictly less than q_sup"),
            underlying: self,
        }
    }
    /// Running Kurtosis.
    /// # Arguments
    /// * `bias` - If `false`, then the calculations are corrected for statistical bias.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., 4.];
    /// let vec_true: Vec<f64> = vec![-3., -2., -1.5, -1.200000000000001];
    /// for (d, t) in data.into_iter().online_kurtosis(false).zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_kurtosis(self, bias: bool) -> IterStat<Kurtosis<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Kurtosis::new(bias),
            underlying: self,
        }
    }
    /// Running max.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., 2.];
    /// let vec_true: Vec<f64> = vec![1., 2., 3., 3.];
    /// for (d, t) in data.into_iter().online_max().zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_max(self) -> IterStat<Max<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Max::new(),
            underlying: self,
        }
    }
    /// Running absolute max.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., -4.];
    /// let vec_true: Vec<f64> = vec![1., 2., 3., 4.];
    /// for (d, t) in data.into_iter().online_abs_max().zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_abs_max(self) -> IterStat<AbsMax<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: AbsMax::new(),
            underlying: self,
        }
    }
    /// Running min.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., -4.];
    /// let vec_true: Vec<f64> = vec![1., 1., 1., -4.];
    /// for (d, t) in data.into_iter().online_min().zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_min(self) -> IterStat<Min<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Min::new(),
            underlying: self,
        }
    }
    /// Running peak to peak.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., -4.];
    /// let vec_true: Vec<f64> = vec![0., 1., 2., 7.];
    /// for (d, t) in data.into_iter().online_ptp().zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_ptp(self) -> IterStat<PeakToPeak<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: PeakToPeak::new(),
            underlying: self,
        }
    }
    /// Running quantile.
    /// # Arguments
    /// * `q` - Desired quantile.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let vec_true: Vec<f64> = vec![9., 9., 7., 7., 6., 6., 6., 6., 5.];
    /// let data: Vec<f64> = vec![9., 7., 3., 2., 6., 1., 8., 5., 4.];
    /// for (d, t) in data
    ///     .into_iter()
    ///     .online_quantile(0.5_f64)
    ///     .zip(vec_true.into_iter())
    /// {
    ///     assert_eq!(d, t);
    /// }
    /// ```
    fn online_quantile(self, q: Self::Item) -> IterStat<Quantile<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Quantile::new(q).expect("q should be between 0 and 1"),
            underlying: self,
        }
    }
    /// Running Skewness.
    /// # Arguments
    /// * `bias` - If `false`, then the calculations are corrected for statistical bias.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., -4.];
    /// let vec_true: Vec<f64> = vec![0., 0., 0., -1.5970779829307837];
    /// for (d, t) in data.into_iter().online_skew(false).zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_skew(self, bias: bool) -> IterStat<Skew<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Skew::new(bias),
            underlying: self,
        }
    }
    /// Running Variance.
    /// # Arguments
    /// * `ddof` - Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n` represents the number of seen elements.
    /// # Examples
    ///
    /// ```
    /// use watermill::iter::IterStatisticsExtend;
    /// let data: Vec<f64> = vec![1., 2., 3., -4.];
    /// let vec_true: Vec<f64> = vec![0., 0.5, 1., 9.666666666666666];
    /// for (d, t) in data.into_iter().online_var(1).zip(vec_true.into_iter()){
    ///     assert_eq!(d, t);
    /// }
    ///
    /// ```
    fn online_var(self, ddof: u32) -> IterStat<Variance<Self::Item>, Self>
    where
        Self::Item: Float + FromPrimitive + AddAssign + SubAssign,
        Self: Sized,
    {
        IterStat {
            stat: Variance::new(ddof),
            underlying: self,
        }
    }
}
impl<I: Iterator> IterStatisticsExtend for I {}