// Rolling Precision-Recall AUC over a sliding window, ported from the original
// C++ implementation based on Gomes, Grégio, Alves, and Almeida, 2023.
//
// See `rolling_roc_auc.rs` for notes on the storage layout (FIFO window plus a
// BTreeSet of 16-byte packed entries used by the score-sorted sweep in `get`).

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
pub struct RollingPRAUC {
    positive_label: i32,
    window_size: usize,
    positives: usize,
    seq: u64,
    window: VecDeque<Entry>,
    ordered: BTreeSet<Entry>,
}

impl RollingPRAUC {
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
        let total = self.window.len();
        if self.positives == 0 || total == self.positives {
            return 0.0;
        }

        let mut fp = total - self.positives;
        let mut tp = self.positives;
        let mut tp_prev = tp;

        let mut auc = 0.0_f64;
        let mut score_prev = f64::INFINITY;

        let mut prec = tp as f64 / (tp + fp) as f64;
        let mut prec_prev = prec;

        for e in self.ordered.iter() {
            if e.score != score_prev {
                prec = tp as f64 / (tp + fp) as f64;
                if prec_prev > prec {
                    prec = prec_prev;
                }
                auc += trapz_area(tp as f64, tp_prev as f64, prec, prec_prev);

                score_prev = e.score;
                tp_prev = tp;
                prec_prev = prec;
            }

            if e.label() == 1 {
                tp -= 1;
            } else {
                fp -= 1;
            }
        }

        auc += trapz_area(tp as f64, tp_prev as f64, 1.0, prec_prev);
        auc / self.positives as f64
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

fn trapz_area(x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    (x1 - x2).abs() * (y1 + y2) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn empty_window_returns_zero() {
        let m = RollingPRAUC::new(1, 10);
        assert_eq!(m.get(), 0.0);
    }

    #[test]
    fn single_class_returns_zero() {
        let mut m = RollingPRAUC::new(1, 10);
        m.update(1, 0.5);
        m.update(1, 0.7);
        assert_eq!(m.get(), 0.0);
    }

    #[test]
    fn doctest_parity() {
        let y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1];
        let y_pred = [0.3, 0.5, 0.5, 0.7, 0.1, 0.3, 0.1, 0.4, 0.35, 0.8];
        let mut m = RollingPRAUC::new(1, 4);
        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            m.update(*yt, *yp);
        }
        let v = m.get();
        assert!((v - 0.8333333333333333).abs() < 1e-9, "got {}", v);
    }

    #[test]
    fn full_window_evicts_oldest() {
        let mut m = RollingPRAUC::new(1, 3);
        m.update(0, 0.1);
        m.update(1, 0.9);
        m.update(0, 0.2);
        m.update(1, 0.5);
        assert!(close(m.get(), 1.0));
    }

    #[test]
    fn revert_with_custom_pos_label_does_not_corrupt_ordered_set() {
        let mut m = RollingPRAUC::new(2, 10);
        m.update(2, 0.9);
        m.update(5, 0.1);
        m.update(2, 0.8);
        m.update(5, 0.2);
        m.revert(2, 0.9);
        assert!(close(m.get(), 1.0));
    }

    #[test]
    fn revert_no_match_is_noop() {
        let mut m = RollingPRAUC::new(1, 10);
        m.update(0, 0.1);
        m.update(1, 0.9);
        let before = m.get();
        m.revert(1, 0.42);
        assert!(close(m.get(), before));
        assert_eq!(m.scores().len(), 2);
    }
}
