//!# watermill
//!
//! `watermill` is crate for Blazingly fast, generic and serializable online statistics.
//!
//!## Quickstart
//! Let's compute the online median and then serialize it:
//!```
//!use watermill::quantile::Quantile;
//!use watermill::stats::Univariate;
//!let data = vec![9., 7., 3., 2., 6., 1., 8., 5., 4.];
//!let mut running_median: Quantile<f64> = Quantile::new(0.5_f64).unwrap();
//!for x in data.iter() {
//!    running_median.update(*x as f64); // update the current statistics
//!    println!("The actual median value is: {}", running_median.get());
//!}
//!assert_eq!(running_median.get(), 5.0);
//!
//!// Convert the statistic to a JSON string.
//!let serialized = serde_json::to_string(&running_median).unwrap();
//!
//!// Convert the JSON string back to a statistic.
//!let deserialized: Quantile<f64> = serde_json::from_str(&serialized).unwrap();
//!
//!```
//!
//!## Installation
//!Add the following line to your `cargo.toml`:
//!```bash
//![dependencies]
//! watermill = "0.1.0"
//!```
//!## Statistics available
//!| Statistics                      | Revertable ?|
//!|---------------------------------|----------|
//!| Mean                            | ✅        |
//!| Variance                        | ✅        |
//!| Sum                             | ✅        |
//!| Min                             | ✅        |
//!| Max                             | ✅        |
//!| Count                           | ❌        |
//!| Quantile                        | ✅        |
//!| Peak to peak                    | ✅        |
//!| Exponentially weighted mean     | ❌        |
//!| Exponentially weighted variance | ❌        |
//!| Interquartile range             | ✅        |
//!| Kurtosis                        | ❌        |
//!| Skewness                        | ❌        |
//!| Covariance                      | ❌        |
//!## Inspiration
//!The `stats` module of the [`river`](https://github.com/online-ml/river) library in `Python` greatly inspired this crate.

pub mod count;
pub mod covariance;
pub mod ewmean;
pub mod ewvariance;
pub mod iqr;
pub mod iter;
pub mod kurtosis;
pub mod maximum;
pub mod mean;
pub mod minimum;
pub mod moments;
pub mod ptp;
pub mod quantile;
pub mod rolling;
pub mod skew;
pub mod sorted_window;
pub mod stats;
pub mod sum;
pub mod variance;
