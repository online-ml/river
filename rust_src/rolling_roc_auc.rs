// Rolling ROC AUC over a sliding window, ported from the original C++ implementation
// based on Brzezinski and Stefanowski, 2017.
//
// The hot path is `update` — `get` is called comparatively rarely — so we keep
// log-N insert/remove via a `BTreeSet` rather than a flat sorted Vec.
//
// Storage:
//   * `window` — FIFO of entries for O(1) eviction.
//   * `ordered` — BTreeSet of the same entries used for the score-sorted sweep
//     in `get`. Entries pack `(label, seq)` into a single `u64` so each item
//     is 16 bytes (vs. 24 for the natural `(f64, i32, u64)` layout) — fewer
//     bytes per node makes the per-call `get` traversal faster too.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeSet, VecDeque};

const LABEL_BIT: u64 = 1 << 63;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Entry {
    score: f64,
    packed: u64,
}

impl Entry {
    #[inline]
    fn new(score: f64, label: i32, seq: u64) -> Self {
        debug_assert!(seq < LABEL_BIT, "seq overflow into label bit");
        let packed = if label == 1 { LABEL_BIT | seq } else { seq };
        Self { score, packed }
    }
    #[inline]
    fn label(&self) -> i32 {
        ((self.packed >> 63) & 1) as i32
    }
}

impl PartialEq for Entry {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.packed == other.packed
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(self.packed.cmp(&other.packed))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RollingROCAUC {
    positive_label: i32,
    window_size: usize,
    positives: usize,
    seq: u64,
    window: VecDeque<Entry>,
    ordered: BTreeSet<Entry>,
}

impl RollingROCAUC {
    pub fn new(positive_label: i32, window_size: usize) -> Self {
        Self {
            positive_label,
            window_size,
            positives: 0,
            seq: 0,
            window: VecDeque::with_capacity(window_size),
            ordered: BTreeSet::new(),
        }
    }

    pub fn update(&mut self, label: i32, score: f64) {
        if self.window.len() == self.window_size {
            self.remove_last();
        }
        self.insert(label, score);
    }

    pub fn revert(&mut self, label: i32, score: f64) {
        let normalized = if label == self.positive_label { 1 } else { 0 };

        let pos = self
            .window
            .iter()
            .position(|e| e.score == score && e.label() == normalized);
        let Some(pos) = pos else { return };

        let entry = self.window.remove(pos).unwrap();
        if normalized == 1 {
            self.positives -= 1;
        }
        self.ordered.remove(&entry);
    }

    pub fn get(&self) -> f64 {
        let total = self.ordered.len();
        if self.positives == 0 || total == self.positives {
            return 0.0;
        }

        // Integer arithmetic and truncating `(c + prev_c) / 2` mirrors the C++
        // behavior; tied scores across classes get an integer-truncated half-credit
        // instead of the proper Mann-Whitney 0.5.
        let mut auc: i64 = 0;
        let mut last_pos_score = f64::NAN;
        let mut c: i64 = 0;
        let mut prev_c: i64 = 0;

        for e in self.ordered.iter().rev() {
            if e.label() == 1 {
                if e.score != last_pos_score {
                    prev_c = c;
                    last_pos_score = e.score;
                }
                c += 1;
            } else if e.score == last_pos_score {
                auc += (c + prev_c) / 2;
            } else {
                auc += c;
            }
        }

        let negatives = total - self.positives;
        auc as f64 / (self.positives as f64 * negatives as f64)
    }

    pub fn true_labels(&self) -> Vec<i32> {
        self.window.iter().map(|e| e.label()).collect()
    }

    pub fn scores(&self) -> Vec<f64> {
        self.window.iter().map(|e| e.score).collect()
    }

    fn insert(&mut self, label: i32, score: f64) {
        let normalized = if label == self.positive_label {
            self.positives += 1;
            1
        } else {
            0
        };
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        let entry = Entry::new(score, normalized, seq);
        self.window.push_back(entry);
        self.ordered.insert(entry);
    }

    fn remove_last(&mut self) {
        let entry = self.window.pop_front().expect("window is non-empty");
        if entry.label() == 1 {
            self.positives -= 1;
        }
        self.ordered.remove(&entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn empty_window_returns_zero() {
        let m = RollingROCAUC::new(1, 10);
        assert_eq!(m.get(), 0.0);
    }

    #[test]
    fn single_class_returns_zero() {
        let mut m = RollingROCAUC::new(1, 10);
        m.update(1, 0.4);
        m.update(1, 0.7);
        assert_eq!(m.get(), 0.0);
        let mut m = RollingROCAUC::new(1, 10);
        m.update(0, 0.4);
        m.update(0, 0.7);
        assert_eq!(m.get(), 0.0);
    }

    #[test]
    fn doctest_parity() {
        let y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1];
        let y_pred = [0.3, 0.5, 0.5, 0.7, 0.1, 0.3, 0.1, 0.4, 0.35, 0.8];
        let mut m = RollingROCAUC::new(1, 4);
        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            m.update(*yt, *yp);
        }
        assert!(close(m.get(), 0.75), "got {}", m.get());
    }

    #[test]
    fn full_window_evicts_oldest() {
        let mut m = RollingROCAUC::new(1, 3);
        m.update(0, 0.1);
        m.update(1, 0.9);
        m.update(0, 0.2);
        m.update(1, 0.5);
        assert!(close(m.get(), 1.0));
    }

    #[test]
    fn revert_with_default_pos_label() {
        let mut m = RollingROCAUC::new(1, 10);
        m.update(0, 0.1);
        m.update(1, 0.9);
        m.update(0, 0.2);
        m.update(1, 0.5);
        let before = m.get();
        m.revert(1, 0.5);
        let after = m.get();
        assert!(close(after, 1.0));
        assert!(close(before, 1.0));
    }

    #[test]
    fn revert_with_custom_pos_label_does_not_corrupt_ordered_set() {
        let mut m = RollingROCAUC::new(2, 10);
        m.update(2, 0.9);
        m.update(5, 0.1);
        m.update(2, 0.8);
        m.update(5, 0.2);
        m.revert(2, 0.9);
        assert!(close(m.get(), 1.0));
    }

    #[test]
    fn revert_no_match_is_noop() {
        let mut m = RollingROCAUC::new(1, 10);
        m.update(0, 0.1);
        m.update(1, 0.9);
        let before = m.get();
        m.revert(1, 0.42);
        assert!(close(m.get(), before));
        assert_eq!(m.scores().len(), 2);
    }
}
