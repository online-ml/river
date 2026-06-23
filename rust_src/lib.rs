// River: blazingly fast, generic and serializable online statistics.

pub mod adwin;
pub mod count;
pub mod covariance;
pub mod ewmean;
pub mod ewvariance;
pub mod expected_mutual_info;
pub mod feature_hashing;
pub mod iqr;
pub mod iter;
pub mod kurtosis;
pub mod maximum;
pub mod mean;
pub mod minimum;
pub mod mondrian;
pub mod moments;
pub mod ptp;
pub mod quantile;
pub mod rolling;
pub mod rolling_pr_auc;
pub mod rolling_roc_auc;
pub mod skew;
pub mod sorted_window;
pub mod stats;
pub mod sum;
pub mod variance;
pub mod vectordict;

mod pyo3_bindings;
