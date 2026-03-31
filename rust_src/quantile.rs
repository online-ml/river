use crate::sorted_window::SortedWindow;
use num::{Float, FromPrimitive, ToPrimitive};
use std::ops::{AddAssign, SubAssign};

use crate::stats::Univariate;
use serde::{Deserialize, Serialize};
/// Running quantile estimator using P-square Algorithm.
/// # Arguments
/// * `q` - quantile value. **WARNING** Should between `0` and `1`. Defaults to `0.5`.
/// # Examples
/// ```
/// use watermill::quantile::Quantile;
/// use watermill::stats::Univariate;
/// let data = vec![9.,7.,3.,2.,6.,1., 8., 5., 4.];
/// let mut running_quantile: Quantile<f64> = Quantile::default();
/// for x in data.iter(){
///     running_quantile.update(*x as f64);
///     //println!("{}", running_quantile.get());
///     running_quantile.get();
/// }
/// assert_eq!(running_quantile.get(), 5.0);
/// ```
/// # References
/// [^1]: [The P² Algorithm for Dynamic Calculation of Quantiles and Histograms Without Storing Observations](https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf)
///
/// [^2]: [P² quantile estimator: estimating the median without storing values](https://aakinshin.net/posts/p2-quantile-estimator-intro/)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quantile<F: Float + FromPrimitive + AddAssign + SubAssign> {
    q: F,
    desired_marker_position: Vec<F>,
    marker_position: Vec<F>,
    position: Vec<F>,
    heights: Vec<F>,
    heights_sorted: bool,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Quantile<F> {
    pub fn new(q: F) -> Result<Self, &'static str> {
        if F::from_f64(0.).unwrap() > q && F::from_f64(1.).unwrap() < q {
            return Err("q should be betweek 0 and 1");
        }
        Ok(Self {
            q,
            desired_marker_position: vec![
                F::from_f64(0.).unwrap(),
                q / F::from_f64(2.).unwrap(),
                q,
                (F::from_f64(1.).unwrap() + q) / F::from_f64(2.).unwrap(),
                F::from_f64(1.).unwrap(),
            ],
            marker_position: vec![
                F::from_f64(1.).unwrap(),
                F::from_f64(1.).unwrap() + F::from_f64(2.).unwrap() * q,
                F::from_f64(1.).unwrap() + F::from_f64(4.).unwrap() * q,
                F::from_f64(3.).unwrap() + F::from_f64(2.).unwrap() * q,
                F::from_f64(5.).unwrap(),
            ],
            position: (1..=5).map(|x| F::from_i32(x).unwrap()).collect(),
            heights: Vec::new(),
            heights_sorted: false,
        })
    }
    fn find_k(&mut self, x: F) -> usize {
        let mut k: Option<usize> = None;
        if x < self.heights[0] {
            self.heights[0] = x;
            k = Some(1);
        } else {
            for i in 1..=4 {
                if self.heights[i - 1] <= x && x < self.heights[i] {
                    k = Some(i);
                    break;
                }
            }
            // If k is None it means that the previous loop did not break
            if let (Some(last_height), None) = (self.heights.last_mut(), k) {
                if *last_height < x {
                    *last_height = x;
                }
            }
        }
        k.unwrap_or(4)
    }
    fn compute_p2(qp1: F, q: F, qm1: F, d: F, np1: F, n: F, nm1: F) -> F {
        let outer = d / (np1 - nm1);
        let inner_left = (n - nm1 + d) * (qp1 - q) / (np1 - n);
        let inner_right = (np1 - n - d) * (q - qm1) / (n - nm1);
        q + outer * (inner_left + inner_right)
    }

    fn adjust(&mut self) {
        for i in 1..4 {
            let n = self.position[i];
            let q = self.heights[i];

            let mut d = self.marker_position[i] - n;
            if (d >= F::from_f64(1.0).unwrap()
                && self.position[i + 1] - n > F::from_f64(1.0).unwrap())
                || (d <= F::from_f64(-1.).unwrap()
                    && self.position[i - 1] - n < F::from_f64(-1.).unwrap())
            {
                d = F::from_f64(1.).unwrap().copysign(d);
                let qp1 = self.heights[i + 1];
                let qm1 = self.heights[i - 1];
                let np1 = self.position[i + 1];
                let nm1 = self.position[i - 1];

                let qn = Quantile::compute_p2(qp1, q, qm1, d, np1, n, nm1);

                if qm1 < qn && qn < qp1 {
                    self.heights[i] = qn;
                } else {
                    // d can be equals to -1 so we complete the operation in isize domain and go back to usize
                    let linear_index = (i.to_isize().unwrap() + d.to_isize().unwrap())
                        .to_usize()
                        .unwrap();
                    self.heights[i] = q + d * (self.heights[linear_index] - q)
                        / (self.position[linear_index] - n);
                }
                self.position[i] = n + d;
            }
        }
    }
}

impl<F> Default for Quantile<F>
where
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn default() -> Self {
        let q = F::from_f64(0.5).unwrap();
        Self {
            q,
            desired_marker_position: vec![
                F::from_f64(0.).unwrap(),
                q / F::from_f64(2.).unwrap(),
                q,
                (F::from_f64(1.).unwrap() + q) / F::from_f64(2.).unwrap(),
                F::from_f64(1.).unwrap(),
            ],
            marker_position: vec![
                F::from_f64(1.).unwrap(),
                F::from_f64(1.).unwrap() + F::from_f64(2.).unwrap() * q,
                F::from_f64(1.).unwrap() + F::from_f64(4.).unwrap() * q,
                F::from_f64(3.).unwrap() + F::from_f64(2.).unwrap() * q,
                F::from_f64(5.).unwrap(),
            ],
            position: (1..6).map(|x| F::from_i32(x).unwrap()).collect(),
            heights: Vec::new(),
            heights_sorted: false,
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for Quantile<F> {
    fn update(&mut self, x: F) {
        // Initialisation
        if self.heights.len() != 5 {
            self.heights.push(x);
        } else {
            if !self.heights_sorted {
                self.heights.sort_by(|x, y| x.partial_cmp(y).unwrap());
                self.heights_sorted = true;
            }
            // Find cell k such that qk < Xj <= qk+i and adjust extreme values (q1 and q) if necessary
            let k = self.find_k(x);

            // Increment all positions greater than k
            for (index, value) in self.position.iter_mut().enumerate() {
                if index >= k {
                    *value += F::from_f64(1.0).unwrap();
                }
            }

            for (marker, desired_marker) in self
                .marker_position
                .iter_mut()
                .zip(self.desired_marker_position.iter())
            {
                *marker += *desired_marker;
            }
            self.adjust();
        }
        self.heights.sort_by(|x, y| x.partial_cmp(y).unwrap());
    }
    fn get(&self) -> F {
        if self.heights_sorted {
            self.heights[2]
        } else {
            let length = F::from_usize(self.heights.len()).unwrap();
            let index = (length - F::from_f64(1.).unwrap())
                .max(F::from_f64(0.).unwrap())
                .min(length * self.q)
                .to_usize()
                .unwrap();

            self.heights[index]
        }
    }
}

/// Rolling quantile.
/// # Arguments
/// * `q` - quantile value. **WARNING** Should between `0` and `1`.
/// * `window_size` - Size of the rolling window.
/// # Examples
/// ```
/// use watermill::quantile::RollingQuantile;
/// use watermill::stats::Univariate;
/// let mut rolling_quantile: RollingQuantile<f64> = RollingQuantile::new(0.5_f64, 101).unwrap();
/// for i in 0..=100{
///     rolling_quantile.update(i as f64);
///     //println!("{}", rolling_quantile.get());
///     rolling_quantile.get();
/// }
/// assert_eq!(rolling_quantile.get(), 50.0);
/// ```
///

#[derive(Serialize, Deserialize)]
pub struct RollingQuantile<F: Float + FromPrimitive + AddAssign + SubAssign> {
    sorted_window: SortedWindow<F>,
    q: F,
    window_size: usize,
    lower: usize,
    higher: usize,
    frac: F,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> RollingQuantile<F> {
    pub fn new(q: F, window_size: usize) -> Result<Self, &'static str> {
        if F::from_f64(0.).unwrap() > q && F::from_f64(1.).unwrap() < q {
            return Err("q should be betweek 0 and 1");
        }
        let idx = q * (F::from_usize(window_size).unwrap() - F::from_f64(1.).unwrap());
        let lower = idx.floor().to_usize().unwrap();
        let mut higher = lower + 1;
        if higher > window_size - 1 {
            higher = lower.saturating_sub(1); // Avoid attempt to subtract with overflow
        }

        let frac = idx - F::from_usize(lower).unwrap();
        Ok(Self {
            sorted_window: SortedWindow::new(window_size),
            q,
            window_size,
            lower,
            higher,
            frac,
        })
    }
    fn prepare(&self) -> (usize, usize, F) {
        if self.sorted_window.len() < self.window_size {
            let idx = self.q
                * (F::from_usize(self.sorted_window.len()).unwrap() - F::from_f64(1.).unwrap());
            let lower = idx.floor().to_usize().unwrap();
            let mut higher = lower + 1;
            if higher > self.sorted_window.len() - 1 {
                higher = self.sorted_window.len().saturating_sub(1); // Avoid attempt to subtract with overflow
            }

            let frac = idx - F::from_usize(lower).unwrap();
            return (lower, higher, frac);
        }
        (self.lower, self.higher, self.frac)
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> Univariate<F> for RollingQuantile<F> {
    fn update(&mut self, x: F) {
        self.sorted_window.push_back(x);
    }
    fn get(&self) -> F {
        let (lower, higher, frac) = self.prepare();
        self.sorted_window[lower] + (self.sorted_window[higher] - self.sorted_window[lower]) * frac
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn rolling_quantile_edge_case() {
        use crate::quantile::RollingQuantile;
        use crate::stats::Univariate;
        let mut rolling_quantile: RollingQuantile<f64> = RollingQuantile::new(1.0_f64, 1).unwrap();
        for i in 0..=1000 {
            rolling_quantile.update(i as f64);
            //println!("{}", rolling_quantile.get());
            rolling_quantile.get();
        }
        assert_eq!(rolling_quantile.get(), 1000.0);
    }

    #[test]
    fn quantile_d_negative() {
        use crate::quantile::Quantile;
        use crate::stats::Univariate;
        let data: Vec<f64> = vec![
            10.557707193831535,
            8.100043020890668,
            9.100117273476478,
            8.892842952595291,
            10.94588485665605,
            10.706797949691644,
            11.568718270819382,
            8.347755330517664,
        ];
        let mut quantile = Quantile::new(0.25_f64).unwrap();
        for d in data.into_iter() {
            quantile.update(d);
        }
    }
    #[test]
    fn first_five_value() {
        use crate::quantile::Quantile;
        use crate::stats::Univariate;
        let data: Vec<f64> = vec![5., 0., 0., 0., 0., 0., 0., 0.];
        let good_value_001_quantile = vec![5., 0., 0., 0., 0., 0., 0., 0.];
        let good_value_099_quantile = vec![
            5.,
            5.,
            5.,
            5.,
            5.,
            0.,
            0.27777777777777773,
            0.8275462962962963,
        ];
        let mut quantile = Quantile::new(0.01_f64).unwrap();
        for (d, gt) in data
            .clone()
            .into_iter()
            .zip(good_value_001_quantile.into_iter())
        {
            quantile.update(d);
            assert_eq!(quantile.get(), gt);
        }
        let mut quantile = Quantile::new(0.99_f64).unwrap();
        for (d, gt) in data.into_iter().zip(good_value_099_quantile.into_iter()) {
            quantile.update(d);
            assert_eq!(quantile.get(), gt);
        }
    }
}
