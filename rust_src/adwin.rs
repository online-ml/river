// ADaptive WINdowing (ADWIN) drift detector.
//
// The implementation follows Bifet & Gavaldà (2007) "Learning from time-changing
// data with adaptive windowing" and Babcock et al. (2003) for the streaming
// variance update. The algorithm maintains an exponentially-bucketed window: each
// bucket-list slot `i` holds up to `max_buckets+1` summary entries, each summary
// representing 2^i data points. New points enter slot 0; when a slot overflows
// its first two entries are merged into the next slot.
//
// On every update (modulated by `clock`), all possible cut points of the window
// are checked: if `|mean(W0) - mean(W1)| > epsilon_cut` the oldest bucket is
// dropped and the algorithm reports drift. The exact arithmetic order matters
// for reproducibility, so the operation order is preserved precisely.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Bucket {
    max_size: usize,
    current_idx: usize,
    totals: Vec<f64>,
    variances: Vec<f64>,
}

impl Bucket {
    fn new(max_size: usize) -> Self {
        // The original allocates `max_size + 1` slots so the (max_buckets + 1)-th
        // entry fits before merging happens.
        let cap = max_size + 1;
        Self {
            max_size,
            current_idx: 0,
            totals: vec![0.0; cap],
            variances: vec![0.0; cap],
        }
    }

    #[inline]
    fn insert_data(&mut self, value: f64, variance: f64) {
        self.totals[self.current_idx] = value;
        self.variances[self.current_idx] = variance;
        self.current_idx += 1;
    }

    #[inline]
    fn compress(&mut self, n_elements: usize) {
        let remaining = self.current_idx - n_elements;
        if remaining > 0 {
            self.totals.copy_within(n_elements..self.current_idx, 0);
            self.variances.copy_within(n_elements..self.current_idx, 0);
        }
        for i in remaining..self.current_idx {
            self.totals[i] = 0.0;
            self.variances[i] = 0.0;
        }
        self.current_idx -= n_elements;
    }

    #[inline]
    fn remove(&mut self) {
        self.compress(1);
    }

    #[inline]
    fn total_at(&self, index: usize) -> f64 {
        self.totals[index]
    }

    #[inline]
    fn variance_at(&self, index: usize) -> f64 {
        self.variances[index]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveWindowing {
    delta: f64,
    clock: i32,
    max_buckets: usize,
    // Stored as f64 (rather than the i32 the Python API exposes) because
    // `detect_change`'s hot loop compares them against `width`/`n0`/`n1`,
    // which are all f64. Casting on each comparison would just add noise.
    min_window_length: f64,
    grace_period: f64,

    total: f64,
    variance: f64,
    width: f64,
    total_width: f64,

    n_buckets: i32,
    max_n_buckets: i32,
    tick: i32,
    n_detections: i32,

    bucket_list: Vec<Bucket>,
}

impl AdaptiveWindowing {
    pub fn new(
        delta: f64,
        clock: i32,
        max_buckets: usize,
        min_window_length: i32,
        grace_period: i32,
    ) -> Self {
        Self {
            delta,
            clock,
            max_buckets,
            min_window_length: min_window_length as f64,
            grace_period: grace_period as f64,

            total: 0.0,
            variance: 0.0,
            width: 0.0,
            total_width: 0.0,

            n_buckets: 0,
            max_n_buckets: 0,
            tick: 0,
            n_detections: 0,

            bucket_list: vec![Bucket::new(max_buckets)],
        }
    }

    pub fn update(&mut self, value: f64) -> bool {
        self.insert_element(value, 0.0);
        self.detect_change()
    }

    pub fn n_detections(&self) -> i32 {
        self.n_detections
    }
    pub fn width(&self) -> f64 {
        self.width
    }
    pub fn total(&self) -> f64 {
        self.total
    }
    pub fn variance(&self) -> f64 {
        self.variance
    }
    pub fn variance_in_window(&self) -> f64 {
        self.variance / self.width
    }

    fn insert_element(&mut self, value: f64, variance: f64) {
        // The first bucket always exists; insertion can never panic.
        self.bucket_list[0].insert_data(value, variance);
        self.n_buckets += 1;
        if self.n_buckets > self.max_n_buckets {
            self.max_n_buckets = self.n_buckets;
        }

        // Welford-style streaming variance update. Order of operations is fixed
        // so f64 rounding is deterministic across runs.
        self.width += 1.0;
        let mut incremental_variance = 0.0;
        if self.width > 1.0 {
            let delta_v = value - self.total / (self.width - 1.0);
            incremental_variance = (self.width - 1.0) * delta_v * delta_v / self.width;
        }
        self.variance += incremental_variance;
        self.total += value;

        self.compress_buckets();
    }

    fn delete_element(&mut self) -> f64 {
        let last = self.bucket_list.len() - 1;
        let n = (1u64 << last) as f64;
        let (u, v) = {
            let bucket = &self.bucket_list[last];
            (bucket.total_at(0), bucket.variance_at(0))
        };
        let mu = u / n;

        self.width -= n;
        self.total -= u;
        let mu_window = self.total / self.width;
        let inc = v + n * self.width * (mu - mu_window) * (mu - mu_window) / (n + self.width);
        self.variance -= inc;

        self.bucket_list[last].remove();
        self.n_buckets -= 1;

        // Pop the empty trailing bucket so subsequent iterations don't see it.
        if self.bucket_list[last].current_idx == 0 {
            self.bucket_list.pop();
        }

        n
    }

    fn compress_buckets(&mut self) {
        let mut idx = 0usize;
        loop {
            let k = self.bucket_list[idx].current_idx;
            // loop-exit condition: stop when this bucket is not yet full
            if k != self.max_buckets + 1 {
                break;
            }

            let n1 = (1u64 << idx) as f64;
            let (t0, t1) = (
                self.bucket_list[idx].total_at(0),
                self.bucket_list[idx].total_at(1),
            );
            let (v0, v1) = (
                self.bucket_list[idx].variance_at(0),
                self.bucket_list[idx].variance_at(1),
            );
            let mu1 = t0 / n1;
            let mu2 = t1 / n1;
            let total12 = t0 + t1;
            let temp = n1 * 0.5 * (mu1 - mu2) * (mu1 - mu2);
            let v12 = v0 + v1 + temp;

            // Allocate a new next bucket if needed, then insert and compress.
            if idx + 1 >= self.bucket_list.len() {
                self.bucket_list.push(Bucket::new(self.max_buckets));
            }
            self.bucket_list[idx + 1].insert_data(total12, v12);
            self.n_buckets += 1;
            self.bucket_list[idx].compress(2);

            // The next bucket is not yet full, so we're done.
            if self.bucket_list[idx + 1].current_idx <= self.max_buckets {
                break;
            }

            idx += 1;
        }
    }

    fn detect_change(&mut self) -> bool {
        let mut change_detected = false;
        self.tick += 1;

        if (self.tick % self.clock == 0) && (self.width > self.grace_period) {
            let mut reduce_width = true;
            while reduce_width {
                reduce_width = false;
                let mut exit_flag = false;
                let mut n0 = 0.0_f64;
                let mut n1 = self.width;
                let mut u0 = 0.0_f64;
                let mut u1 = self.total;

                let deque_len = self.bucket_list.len();

                'outer: for idx in (0..deque_len).rev() {
                    if exit_flag {
                        break;
                    }
                    let bucket_size = (1u64 << idx) as f64;
                    let bucket_current_idx = self.bucket_list[idx].current_idx;
                    for k in 0..bucket_current_idx {
                        n0 += bucket_size;
                        n1 -= bucket_size;
                        let total_at = self.bucket_list[idx].total_at(k);
                        u0 += total_at;
                        u1 -= total_at;

                        if idx == 0 && k == bucket_current_idx - 1 {
                            exit_flag = true;
                            break;
                        }

                        let delta_mean = (u0 / n0) - (u1 / n1);
                        if n1 >= self.min_window_length
                            && n0 >= self.min_window_length
                            && self.evaluate_cut(n0, n1, delta_mean, self.delta)
                        {
                            reduce_width = true;
                            change_detected = true;
                            if self.width > 0.0 {
                                self.delete_element();
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        self.total_width += self.width;
        if change_detected {
            self.n_detections += 1;
        }
        change_detected
    }

    fn evaluate_cut(&self, n0: f64, n1: f64, delta_mean: f64, delta: f64) -> bool {
        let delta_prime = (2.0 * self.width.ln() / delta).ln();
        let m_recip = 1.0 / (n0 - self.min_window_length + 1.0)
            + 1.0 / (n1 - self.min_window_length + 1.0);
        let epsilon = (2.0 * m_recip * (self.variance / self.width) * delta_prime).sqrt()
            + 2.0 / 3.0 * delta_prime * m_recip;
        delta_mean.abs() > epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_aw() -> AdaptiveWindowing {
        AdaptiveWindowing::new(0.002, 32, 5, 5, 10)
    }

    #[test]
    fn first_update_no_drift() {
        let mut aw = default_aw();
        assert!(!aw.update(1.0));
        assert_eq!(aw.width(), 1.0);
        assert_eq!(aw.total(), 1.0);
        assert_eq!(aw.variance(), 0.0);
    }

    #[test]
    fn width_grows_with_constant_stream() {
        let mut aw = default_aw();
        for i in 1..=100 {
            aw.update(1.0);
            assert_eq!(aw.width(), i as f64);
        }
    }

    #[test]
    fn estimation_after_short_sequence() {
        let mut aw = default_aw();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            aw.update(v);
        }
        assert_eq!(aw.total(), 15.0);
        assert_eq!(aw.width(), 5.0);
        // Sum of squared deviations from mean 3.0
        let expected = (1.0_f64 - 3.0).powi(2)
            + (2.0_f64 - 3.0).powi(2)
            + (3.0_f64 - 3.0).powi(2)
            + (4.0_f64 - 3.0).powi(2)
            + (5.0_f64 - 3.0).powi(2);
        assert!((aw.variance() - expected).abs() < 1e-9);
    }

    #[test]
    fn no_drift_in_constant_stream() {
        let mut aw = default_aw();
        for _ in 0..5_000 {
            assert!(!aw.update(1.0));
        }
        assert_eq!(aw.n_detections(), 0);
    }

    #[test]
    fn detects_large_shift() {
        let mut aw = AdaptiveWindowing::new(0.01, 32, 5, 5, 10);
        for _ in 0..200 {
            aw.update(0.0);
        }
        let mut detected = false;
        for _ in 0..200 {
            if aw.update(100.0) {
                detected = true;
                break;
            }
        }
        assert!(detected, "large shift must be detected");
    }
}
