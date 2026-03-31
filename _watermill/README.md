# Online statistics in Rust 🦀 

**`watermill` is crate 🦀 for Blazingly fast, generic and serializable online statistics.**

## Quickstart
---------
Let's compute the online median and then serialize it:
```rust
use watermill::quantile::Quantile;
use watermill::stats::Univariate;
let data: Vec<f64> = vec![9., 7., 3., 2., 6., 1., 8., 5., 4.];
let mut running_median: Quantile<f64> = Quantile::new(0.5_f64).unwrap();
for x in data.into_iter() {
    running_median.update(x); // update the current statistics
    println!("The actual median value is: {}", running_median.get());
}
assert_eq!(running_median.get(), 5.0);

// Convert the statistic to a JSON string.
let serialized = serde_json::to_string(&running_median).unwrap();

// Convert the JSON string back to a statistic.
let deserialized: Quantile<f64> = serde_json::from_str(&serialized).unwrap();

```
Now let's compute the online sum using the iterators:
```rust
use watermill::iter::IterStatisticsExtend;
let data: Vec<f64> = vec![1., 2., 3.];
let vec_true: Vec<f64> = vec![1., 3., 6.];
for (d, t) in data.into_iter().online_sum().zip(vec_true.into_iter()) {
    assert_eq!(d, t); //       ^^^^^^^^^^
}
```

You can also compute rolling statistics; in the following example let's compute the rolling sum on 2 previous data: 
```rust

use watermill::rolling::Rolling;
use watermill::stats::Univariate;
use watermill::variance::Variance;
let data: Vec<f64> = vec![9., 7., 3., 2., 6., 1., 8., 5., 4.];
let mut running_var: Variance<f64> = Variance::default();
// We wrap `running_var` inside the `Rolling` struct.
let mut rolling_var: Rolling<f64> = Rolling::new(&mut running_var, 2).unwrap();
for x in data.into_iter() {
    rolling_var.update(x);
}
assert_eq!(rolling_var.get(), 0.5);
```

## Installation
---------
Add the following line to your `cargo.toml`:
```
[dependencies]
watermill = "0.1.0"
```

## Statistics available
| Statistics                      	| Rollable ?|
|---------------------------------	|----------	|
| Mean                            	| ✅        	|
| Variance                        	| ✅        	|
| Sum                             	| ✅        	|
| Min                             	| ✅        	|
| Max                             	| ✅        	|
| Count                           	| ❌        	|
| Quantile                        	| ✅        	|
| Peak to peak                    	| ✅        	|
| Exponentially weighted mean     	| ❌        	|
| Exponentially weighted variance 	| ❌        	|
| Interquartile range             	| ✅        	|
| Kurtosis                        	| ❌        	|
| Skewness                        	| ❌        	|
| Covariance                      	| ❌        	|

## Inspiration
---------
The `stats` module of the [`river`](https://github.com/online-ml/river) library in `Python` greatly inspired this crate. 