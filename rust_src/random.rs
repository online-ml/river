use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct Random {
    state: [u64; 4],
}

impl Random {
    pub(crate) fn new(seed: u64) -> Self {
        let mut seed = seed;
        Self {
            state: std::array::from_fn(|_| split_mix_64(&mut seed)),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let shift = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= shift;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    pub(crate) fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    pub(crate) fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }

    pub(crate) fn weighted_index(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let target = self.random() * total;
        let mut cumulative = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if target < cumulative {
                return index;
            }
        }
        weights.len() - 1
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        let upper = upper as u64;
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % upper) as usize;
            }
        }
    }

    pub(crate) fn sample_values(&mut self, population: &[usize], size: usize) -> Vec<usize> {
        let mut set_size = 21_usize;
        if size > 5 {
            let target = size.saturating_mul(3);
            let mut power = 1_usize;
            while power < target {
                power = power.saturating_mul(4);
            }
            set_size = set_size.saturating_add(power);
        }
        if population.len() <= set_size {
            let mut values = population.to_vec();
            let mut result = Vec::with_capacity(size);
            for index in 0..size {
                let selected = self.randbelow(population.len() - index);
                result.push(values[selected]);
                values[selected] = values[population.len() - index - 1];
            }
            return result;
        }
        let mut selected = HashSet::with_capacity(size);
        let mut values = Vec::with_capacity(size);
        while values.len() < size {
            let index = self.randbelow(population.len());
            if selected.insert(index) {
                values.push(population[index]);
            }
        }
        values
    }
}

fn split_mix_64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_stream_is_reproducible() {
        let mut left = Random::new(42);
        let mut right = Random::new(42);
        for _ in 0..1_000 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }

    #[test]
    fn samples_are_unique_and_in_range() {
        let mut rng = Random::new(42);
        let sample = rng.sample_values(&(0..100).collect::<Vec<_>>(), 20);
        assert_eq!(sample.len(), 20);
        assert_eq!(sample.iter().collect::<HashSet<_>>().len(), 20);
        assert!(sample.iter().all(|value| *value < 100));
    }

    #[test]
    fn floats_are_in_unit_interval() {
        let mut rng = Random::new(42);
        for _ in 0..10_000 {
            assert!((0.0..1.0).contains(&rng.random()));
        }
    }
}
