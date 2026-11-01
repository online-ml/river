use crate::gaussian::Gaussian;
use crate::tree_arena::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SplitCriterion {
    InfoGain,
    Gini,
    Hellinger,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LeafPrediction {
    MajorityClass,
    NaiveBayes,
    NaiveBayesAdaptive,
}

pub type NodeSnapshot = (isize, f64, usize, usize, Vec<f64>, usize, bool);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Rng {
    state: Vec<u32>,
    index: usize,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let key = if seed > u32::MAX as u64 {
            vec![seed as u32, (seed >> 32) as u32]
        } else {
            vec![seed as u32]
        };
        let mut state = vec![0; 624];
        state[0] = 19_650_218;
        for index in 1..624 {
            state[index] = 1_812_433_253_u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        let mut i = 1;
        let mut j = 0;
        for _ in 0..624.max(key.len()) {
            state[i] = (state[i] ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            state[i] = (state[i]
                ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        state[0] = 0x8000_0000;
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn getrandbits(&mut self, mut bits: u32) -> u64 {
        if bits <= 32 {
            return (self.next_u32() >> (32 - bits)) as u64;
        }
        let mut value = 0_u64;
        let mut shift = 0;
        while bits > 0 {
            let take = bits.min(32);
            let word = if take == 32 {
                self.next_u32()
            } else {
                self.next_u32() >> (32 - take)
            };
            value |= (word as u64) << shift;
            shift += 32;
            bits -= take;
        }
        value
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = self.getrandbits(bits) as usize;
            if value < upper {
                return value;
            }
        }
    }

    fn sample_values(&mut self, population: &[usize], size: usize) -> Vec<usize> {
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct GaussianObserver {
    distribution: Gaussian,
    min: f64,
    max: f64,
}

impl GaussianObserver {
    fn update(&mut self, value: f64, weight: f64) {
        if self.weight() == 0.0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.distribution.update(value, weight);
    }

    fn weight(&self) -> f64 {
        self.distribution.n_samples()
    }

    fn log_pdf(&self, value: f64) -> f64 {
        self.distribution.log_pdf(value)
    }

    fn cdf(&self, value: f64) -> f64 {
        self.distribution.cdf(value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Leaf {
    counts: Vec<f64>,
    observers: Vec<GaussianObserver>,
    features: Vec<usize>,
    depth: usize,
    last_split_attempt_at: f64,
    majority_correct: f64,
    naive_bayes_correct: f64,
    active: bool,
}

impl Leaf {
    fn new(depth: usize, counts: Vec<f64>) -> Self {
        let last_split_attempt_at = counts.iter().sum();
        Self {
            counts,
            observers: Vec::new(),
            features: Vec::new(),
            depth,
            last_split_attempt_at,
            majority_correct: 0.0,
            naive_bayes_correct: 0.0,
            active: true,
        }
    }

    fn total_weight(&self) -> f64 {
        self.counts.iter().sum()
    }

    fn majority_proba(&self) -> Vec<f64> {
        let total = self.total_weight();
        if total == 0.0 {
            vec![0.0; self.counts.len()]
        } else {
            self.counts.iter().map(|count| count / total).collect()
        }
    }

    fn naive_bayes_proba(&self, x: &[f64]) -> Vec<f64> {
        let total = self.total_weight();
        if total == 0.0 {
            return vec![0.0; self.counts.len()];
        }
        let mut votes = vec![f64::NEG_INFINITY; self.counts.len()];
        for (class_index, class_weight) in self.counts.iter().copied().enumerate() {
            if class_weight <= 0.0 {
                continue;
            }
            let mut vote = (class_weight / total).ln();
            for (feature_index, feature) in self.features.iter().enumerate() {
                if x[*feature].is_nan() {
                    continue;
                }
                let observer = &self.observers[feature_index * self.counts.len() + class_index];
                let log_probability = observer.log_pdf(x[*feature]);
                if log_probability.is_finite() {
                    vote += log_probability;
                }
            }
            votes[class_index] = vote;
        }
        let maximum = votes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return vec![0.0; self.counts.len()];
        }
        let normalizer: f64 = votes
            .iter()
            .map(|value| {
                if value.is_finite() {
                    (value - maximum).exp()
                } else {
                    0.0
                }
            })
            .sum();
        votes
            .iter()
            .map(|value| {
                if value.is_finite() {
                    (value - maximum).exp() / normalizer
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn naive_bayes_prediction(&self, x: &[f64]) -> Option<usize> {
        self.naive_bayes_prediction_ordered(x, 0..self.counts.len())
    }

    fn naive_bayes_prediction_ordered(
        &self,
        x: &[f64],
        order: impl IntoIterator<Item = usize>,
    ) -> Option<usize> {
        let total = self.total_weight();
        if total == 0.0 {
            return None;
        }
        let mut best = None;
        let mut best_vote = f64::NEG_INFINITY;
        for class_index in order {
            let class_weight = self.counts[class_index];
            if class_weight <= 0.0 {
                continue;
            }
            let mut vote = (class_weight / total).ln();
            for (feature_index, feature) in self.features.iter().enumerate() {
                if x[*feature].is_nan() {
                    continue;
                }
                let observer = &self.observers[feature_index * self.counts.len() + class_index];
                let log_probability = observer.log_pdf(x[*feature]);
                if log_probability.is_finite() {
                    vote += log_probability;
                }
            }
            if vote > best_vote {
                best = Some(class_index);
                best_vote = vote;
            }
        }
        best
    }

    fn predict(&self, x: &[f64], prediction: LeafPrediction, nb_threshold: f64) -> usize {
        if !self.active {
            return argmax(&self.counts);
        }
        match prediction {
            LeafPrediction::MajorityClass => argmax(&self.counts),
            LeafPrediction::NaiveBayes if self.total_weight() >= nb_threshold => {
                self.naive_bayes_prediction(x).unwrap_or(0)
            }
            LeafPrediction::NaiveBayes => argmax(&self.counts),
            LeafPrediction::NaiveBayesAdaptive
                if self.naive_bayes_correct >= self.majority_correct =>
            {
                self.naive_bayes_prediction(x).unwrap_or(0)
            }
            LeafPrediction::NaiveBayesAdaptive => argmax(&self.counts),
        }
    }

    fn predict_ordered(
        &self,
        x: &[f64],
        prediction: LeafPrediction,
        nb_threshold: f64,
        order: &[usize],
    ) -> usize {
        if !self.active {
            return argmax_ordered(&self.counts, order);
        }
        match prediction {
            LeafPrediction::MajorityClass => argmax_ordered(&self.counts, order),
            LeafPrediction::NaiveBayes if self.total_weight() >= nb_threshold => self
                .naive_bayes_prediction_ordered(x, order.iter().copied())
                .unwrap_or(0),
            LeafPrediction::NaiveBayes => argmax_ordered(&self.counts, order),
            LeafPrediction::NaiveBayesAdaptive
                if self.naive_bayes_correct >= self.majority_correct =>
            {
                self.naive_bayes_prediction_ordered(x, order.iter().copied())
                    .unwrap_or(0)
            }
            LeafPrediction::NaiveBayesAdaptive => argmax_ordered(&self.counts, order),
        }
    }

    fn predict_proba(&self, x: &[f64], prediction: LeafPrediction, nb_threshold: f64) -> Vec<f64> {
        if !self.active {
            return self.majority_proba();
        }
        match prediction {
            LeafPrediction::MajorityClass => self.majority_proba(),
            LeafPrediction::NaiveBayes if self.total_weight() >= nb_threshold => {
                self.naive_bayes_proba(x)
            }
            LeafPrediction::NaiveBayes => self.majority_proba(),
            LeafPrediction::NaiveBayesAdaptive
                if self.naive_bayes_correct >= self.majority_correct =>
            {
                self.naive_bayes_proba(x)
            }
            LeafPrediction::NaiveBayesAdaptive => self.majority_proba(),
        }
    }

    fn ensure_features(
        &mut self,
        x: &[f64],
        n_features: usize,
        max_features: usize,
        all_features: bool,
        selected_features: Option<&[usize]>,
        rng: &mut Rng,
    ) {
        if self.features.is_empty() {
            self.features = if all_features || max_features >= n_features {
                (0..n_features).collect()
            } else if let Some(selected_features) = selected_features {
                selected_features.to_vec()
            } else {
                let available = x
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| (!value.is_nan()).then_some(index))
                    .collect::<Vec<_>>();
                if available.len() >= max_features {
                    rng.sample_values(&available, max_features)
                } else {
                    available
                }
            };
            self.observers =
                vec![GaussianObserver::default(); self.features.len() * self.counts.len()];
        } else if all_features && self.features.len() < n_features {
            for feature in self.features.len()..n_features {
                self.features.push(feature);
                self.observers
                    .extend((0..self.counts.len()).map(|_| GaussianObserver::default()));
            }
        }
    }

    fn insert_class(&mut self, index: usize, grow: bool) {
        let old_n_classes = self.counts.len();
        if grow {
            self.counts.insert(index, 0.0);
        } else if index == 0 {
            self.counts.swap(0, 1);
        }
        let mut observers = Vec::with_capacity(self.features.len() * self.counts.len());
        for feature_index in 0..self.features.len() {
            let start = feature_index * old_n_classes;
            let mut feature_observers = self.observers[start..start + old_n_classes].to_vec();
            if grow {
                feature_observers.insert(index, GaussianObserver::default());
            } else if index == 0 {
                feature_observers.swap(0, 1);
            }
            observers.extend(feature_observers);
        }
        self.observers = observers;
    }

    fn learn(
        &mut self,
        x: &[f64],
        y: usize,
        weight: f64,
        n_features: usize,
        max_features: usize,
        all_features: bool,
        prediction: LeafPrediction,
        selected_features: Option<&[usize]>,
        rng: &mut Rng,
    ) {
        if self.active && matches!(prediction, LeafPrediction::NaiveBayesAdaptive) {
            let majority = argmax(&self.counts);
            if self.total_weight() == 0.0 || majority == y {
                self.majority_correct += weight;
            }
            if self
                .naive_bayes_prediction(x)
                .is_some_and(|prediction| prediction == y)
            {
                self.naive_bayes_correct += weight;
            }
        }
        self.counts[y] += weight;
        if self.active {
            self.ensure_features(
                x,
                n_features,
                max_features,
                all_features,
                selected_features,
                rng,
            );
            for (feature_index, feature) in self.features.iter().enumerate() {
                if !x[*feature].is_nan() {
                    self.observers[feature_index * self.counts.len() + y]
                        .update(x[*feature], weight);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Branch {
    feature: usize,
    threshold: f64,
    left: NodeId,
    right: NodeId,
    left_weight: f64,
    right_weight: f64,
}

#[derive(Clone, Debug)]
struct SplitCandidate {
    merit: f64,
    feature: usize,
    threshold: f64,
    left: Vec<f64>,
    right: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoeffdingTreeClassifier {
    root: NodeId,
    branches: Vec<Branch>,
    leaves: Vec<Leaf>,
    n_features: usize,
    n_classes: usize,
    max_features: usize,
    all_features: bool,
    grace_period: f64,
    delta: f64,
    tau: f64,
    n_splits: usize,
    max_depth: usize,
    split_criterion: SplitCriterion,
    leaf_prediction: LeafPrediction,
    nb_threshold: f64,
    min_branch_fraction: f64,
    max_share_to_split: f64,
    merit_preprune: bool,
    total_weight_seen: f64,
    rng: Rng,
}

impl HoeffdingTreeClassifier {
    pub fn new(
        n_features: usize,
        n_classes: usize,
        max_features: usize,
        all_features: bool,
        grace_period: f64,
        delta: f64,
        tau: f64,
        n_splits: usize,
        max_depth: usize,
        seed: u64,
        split_criterion: SplitCriterion,
        leaf_prediction: LeafPrediction,
        nb_threshold: f64,
        min_branch_fraction: f64,
        max_share_to_split: f64,
        merit_preprune: bool,
    ) -> Self {
        Self {
            root: NodeId::leaf(0),
            branches: Vec::new(),
            leaves: vec![Leaf::new(0, vec![0.0; n_classes])],
            n_features,
            n_classes,
            max_features: max_features.clamp(1, n_features),
            all_features,
            grace_period,
            delta,
            tau,
            n_splits,
            max_depth,
            split_criterion,
            leaf_prediction,
            nb_threshold,
            min_branch_fraction,
            max_share_to_split,
            merit_preprune,
            total_weight_seen: 0.0,
            rng: Rng::new(seed),
        }
    }

    fn leaf_index(&self, x: &[f64]) -> usize {
        let mut node = self.root;
        loop {
            if node.is_leaf() {
                return node.index();
            }
            let branch = &self.branches[node.index()];
            node = if x[branch.feature].is_nan() {
                if branch.left_weight >= branch.right_weight {
                    branch.left
                } else {
                    branch.right
                }
            } else if x[branch.feature] <= branch.threshold {
                branch.left
            } else {
                branch.right
            };
        }
    }

    fn learning_leaf_index(&mut self, x: &[f64], weight: f64) -> usize {
        let mut node = self.root;
        loop {
            if node.is_leaf() {
                return node.index();
            }
            let branch = &mut self.branches[node.index()];
            let go_right = if x[branch.feature].is_nan() {
                if branch.left_weight >= branch.right_weight {
                    branch.left_weight += weight;
                    false
                } else {
                    branch.right_weight += weight;
                    true
                }
            } else if x[branch.feature] <= branch.threshold {
                branch.left_weight += weight;
                false
            } else {
                branch.right_weight += weight;
                true
            };
            node = if go_right { branch.right } else { branch.left };
        }
    }

    fn replace_leaf(&mut self, x: &[f64], leaf: usize, replacement: NodeId) {
        let mut node = self.root;
        let mut parent = None;
        while !node.is_leaf() {
            let branch_index = node.index();
            let branch = &self.branches[branch_index];
            let go_right = if x[branch.feature].is_nan() {
                branch.left_weight < branch.right_weight
            } else {
                x[branch.feature] > branch.threshold
            };
            node = if go_right { branch.right } else { branch.left };
            parent = Some((branch_index, go_right));
        }
        assert_eq!(node.index(), leaf);
        if let Some((branch_index, right)) = parent {
            if right {
                self.branches[branch_index].right = replacement;
            } else {
                self.branches[branch_index].left = replacement;
            }
        } else {
            self.root = replacement;
        }
    }

    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        self.leaves[self.leaf_index(x)].predict_proba(x, self.leaf_prediction, self.nb_threshold)
    }

    pub fn predict(&self, x: &[f64]) -> usize {
        self.leaves[self.leaf_index(x)].predict(x, self.leaf_prediction, self.nb_threshold)
    }

    pub fn predict_ordered(&self, x: &[f64], order: &[usize]) -> usize {
        self.leaves[self.leaf_index(x)].predict_ordered(
            x,
            self.leaf_prediction,
            self.nb_threshold,
            order,
        )
    }

    pub fn learn(&mut self, x: &[f64], y: usize, weight: f64) {
        self.learn_with_selected_features(x, y, weight, None);
    }

    pub fn feature_sample_request(&self, x: &[f64]) -> Option<(Vec<usize>, usize)> {
        let leaf = &self.leaves[self.leaf_index(x)];
        if self.all_features || !leaf.features.is_empty() {
            return None;
        }
        let available = x
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (!value.is_nan()).then_some(index))
            .collect::<Vec<_>>();
        (available.len() >= self.max_features).then_some((available, self.max_features))
    }

    pub fn learn_with_selected_features(
        &mut self,
        x: &[f64],
        y: usize,
        weight: f64,
        selected_features: Option<&[usize]>,
    ) {
        self.total_weight_seen += weight;
        let leaf_index = self.learning_leaf_index(x, weight);
        let leaf = &mut self.leaves[leaf_index];
        leaf.learn(
            x,
            y,
            weight,
            self.n_features,
            self.max_features,
            self.all_features,
            self.leaf_prediction,
            selected_features,
            &mut self.rng,
        );
        let should_attempt = leaf.active
            && leaf.depth < self.max_depth
            && leaf.total_weight() - leaf.last_split_attempt_at >= self.grace_period;
        if should_attempt && !self.attempt_split(leaf_index, x) {
            let leaf = &mut self.leaves[leaf_index];
            leaf.last_split_attempt_at = leaf.total_weight();
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.branches.len() + self.leaves.len()
    }

    pub fn n_branches(&self) -> usize {
        self.branches.len()
    }

    pub fn n_leaves(&self) -> usize {
        self.leaves.len()
    }

    pub fn n_active_leaves(&self) -> usize {
        self.leaves.iter().filter(|leaf| leaf.active).count()
    }

    pub fn height(&self) -> usize {
        self.leaves.iter().map(|leaf| leaf.depth).max().unwrap_or(0)
    }

    pub fn total_weight_seen(&self) -> f64 {
        self.total_weight_seen
    }

    pub fn nodes(&self) -> Vec<NodeSnapshot> {
        let mut ordered = Vec::with_capacity(self.n_nodes());
        let mut branch_positions = vec![0; self.branches.len()];
        let mut leaf_positions = vec![0; self.leaves.len()];
        let mut stack = vec![(self.root, 0)];
        while let Some((node, depth)) = stack.pop() {
            let position = ordered.len();
            ordered.push((node, depth));
            if node.is_leaf() {
                leaf_positions[node.index()] = position;
            } else {
                branch_positions[node.index()] = position;
                let branch = &self.branches[node.index()];
                stack.push((branch.right, depth + 1));
                stack.push((branch.left, depth + 1));
            }
        }

        let position = |node: NodeId| {
            if node.is_leaf() {
                leaf_positions[node.index()]
            } else {
                branch_positions[node.index()]
            }
        };
        let mut counts = vec![vec![0.0; self.n_classes]; ordered.len()];
        for (node_position, (node, _)) in ordered.iter().enumerate().rev() {
            if node.is_leaf() {
                counts[node_position].clone_from(&self.leaves[node.index()].counts);
            } else {
                let branch = &self.branches[node.index()];
                let left = position(branch.left);
                let right = position(branch.right);
                counts[node_position] = counts[left]
                    .iter()
                    .zip(&counts[right])
                    .map(|(left, right)| left + right)
                    .collect();
            }
        }

        ordered
            .iter()
            .enumerate()
            .map(|(node_position, (node, depth))| {
                if node.is_leaf() {
                    let leaf = &self.leaves[node.index()];
                    (
                        -1,
                        f64::NAN,
                        0,
                        0,
                        counts[node_position].clone(),
                        *depth,
                        leaf.active,
                    )
                } else {
                    let branch = &self.branches[node.index()];
                    (
                        branch.feature as isize,
                        branch.threshold,
                        position(branch.left),
                        position(branch.right),
                        counts[node_position].clone(),
                        *depth,
                        false,
                    )
                }
            })
            .collect()
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub fn add_features(&mut self, count: usize) {
        self.n_features += count;
    }

    pub fn add_class(&mut self) {
        self.insert_class(self.n_classes, true);
    }

    pub fn insert_class(&mut self, index: usize, grow: bool) {
        if grow {
            self.n_classes += 1;
        }
        for leaf in &mut self.leaves {
            leaf.insert_class(index, grow);
        }
    }

    pub fn set_mutable_parameters(&mut self, grace_period: f64, delta: f64, tau: f64) {
        self.grace_period = grace_period;
        self.delta = delta;
        self.tau = tau;
    }

    fn attempt_split(&mut self, leaf_index: usize, x: &[f64]) -> bool {
        let leaf = &self.leaves[leaf_index];
        let total = leaf.total_weight();
        let majority = leaf.counts.iter().copied().fold(0.0, f64::max);
        let observed_classes = leaf.counts.iter().filter(|count| **count > 0.0).count();
        let dominated = majority / total > self.max_share_to_split;
        if observed_classes < 2 {
            return false;
        }
        if dominated {
            self.leaves[leaf_index].active = false;
            return false;
        }
        let split = {
            let leaf = &self.leaves[leaf_index];
            let mut suggestions =
                Vec::with_capacity(leaf.features.len() + usize::from(self.merit_preprune));
            if self.merit_preprune {
                suggestions.push(None);
            }
            for (feature_index, feature) in leaf.features.iter().enumerate() {
                let observer_start = feature_index * leaf.counts.len();
                let observer = &leaf.observers[observer_start..observer_start + leaf.counts.len()];
                if observer.iter().any(|stats| stats.weight() > 0.0) {
                    suggestions.push(best_split_candidate(
                        observer,
                        &leaf.counts,
                        *feature,
                        self.n_splits,
                        self.min_branch_fraction,
                        self.split_criterion,
                    ));
                }
            }
            suggestions.sort_by(|left, right| {
                left.as_ref()
                    .map_or(f64::NEG_INFINITY, |candidate| candidate.merit)
                    .total_cmp(
                        &right
                            .as_ref()
                            .map_or(f64::NEG_INFINITY, |candidate| candidate.merit),
                    )
            });
            let should_split = if suggestions.len() < 2 {
                !suggestions.is_empty()
            } else {
                let best_merit = suggestions
                    .last()
                    .and_then(Option::as_ref)
                    .map_or(f64::NEG_INFINITY, |candidate| candidate.merit);
                let second_merit = suggestions[suggestions.len() - 2]
                    .as_ref()
                    .map_or(f64::NEG_INFINITY, |candidate| candidate.merit);
                let merit_range = match self.split_criterion {
                    SplitCriterion::Gini => 1.0,
                    SplitCriterion::InfoGain | SplitCriterion::Hellinger => {
                        (observed_classes.max(2) as f64).log2()
                    }
                };
                let total = leaf.total_weight();
                let bound =
                    ((merit_range * merit_range * (1.0 / self.delta).ln()) / (2.0 * total)).sqrt();
                best_merit - second_merit > bound || bound < self.tau
            };
            if !should_split {
                return false;
            }
            let best = match suggestions.pop().flatten() {
                Some(candidate) => candidate,
                None => {
                    self.leaves[leaf_index].active = false;
                    return false;
                }
            };
            (leaf.depth, best)
        };
        let (depth, best) = split;
        let left_weight = best.left.iter().sum();
        let right_weight = best.right.iter().sum();
        let left = NodeId::leaf(leaf_index);
        self.leaves[leaf_index] = Leaf::new(depth + 1, best.left);
        let right = NodeId::leaf(self.leaves.len());
        self.leaves.push(Leaf::new(depth + 1, best.right));
        let branch = NodeId::branch(self.branches.len());
        self.branches.push(Branch {
            feature: best.feature,
            threshold: best.threshold,
            left,
            right,
            left_weight,
            right_weight,
        });
        self.replace_leaf(x, leaf_index, branch);
        true
    }
}

fn best_split_candidate(
    observer: &[GaussianObserver],
    counts: &[f64],
    feature: usize,
    n_splits: usize,
    min_branch_fraction: f64,
    criterion: SplitCriterion,
) -> Option<SplitCandidate> {
    let minimum = observer
        .iter()
        .filter(|stats| stats.weight() > 0.0)
        .map(|stats| stats.min)
        .fold(f64::INFINITY, f64::min);
    let maximum = observer
        .iter()
        .filter(|stats| stats.weight() > 0.0)
        .map(|stats| stats.max)
        .fold(f64::NEG_INFINITY, f64::max);
    if !minimum.is_finite() || maximum <= minimum {
        return None;
    }
    let mut best: Option<SplitCandidate> = None;
    let span = maximum - minimum;
    let bin_size = span / (n_splits as f64 + 1.0);
    for i in 1..=n_splits {
        let threshold = if bin_size.is_finite() {
            minimum + bin_size * i as f64
        } else {
            let ratio = i as f64 / (n_splits as f64 + 1.0);
            minimum * (1.0 - ratio) + maximum * ratio
        };
        let mut left = vec![0.0; counts.len()];
        let mut right = vec![0.0; counts.len()];
        for class_index in 0..counts.len() {
            let stats = &observer[class_index];
            let weight = stats.weight();
            if weight == 0.0 {
                continue;
            }
            if threshold < stats.min {
                right[class_index] = weight;
            } else if threshold >= stats.max {
                left[class_index] = weight;
            } else {
                left[class_index] = stats.cdf(threshold) * weight;
                right[class_index] = weight - left[class_index];
            }
        }
        let left_weight: f64 = left.iter().sum();
        let right_weight: f64 = right.iter().sum();
        let total = left_weight + right_weight;
        if total == 0.0
            || left_weight / total <= min_branch_fraction
            || right_weight / total <= min_branch_fraction
        {
            continue;
        }
        let merit = split_merit(counts, &left, &right, criterion);
        if best
            .as_ref()
            .is_none_or(|candidate| merit > candidate.merit)
        {
            best = Some(SplitCandidate {
                merit,
                feature,
                threshold,
                left,
                right,
            });
        }
    }
    best
}

fn split_merit(counts: &[f64], left: &[f64], right: &[f64], criterion: SplitCriterion) -> f64 {
    let left_weight: f64 = left.iter().sum();
    let right_weight: f64 = right.iter().sum();
    let total = left_weight + right_weight;
    match criterion {
        SplitCriterion::InfoGain => {
            entropy(counts) - (left_weight * entropy(left) + right_weight * entropy(right)) / total
        }
        SplitCriterion::Gini => {
            1.0 - (left_weight * gini(left) + right_weight * gini(right)) / total
        }
        SplitCriterion::Hellinger => {
            let total_negative = left[0] + right[0];
            let total_positive = left[1] + right[1];
            if total_negative == 0.0 || total_positive == 0.0 {
                return 0.0;
            }
            let left_difference =
                (left[0] / total_negative).sqrt() - (left[1] / total_positive).sqrt();
            let right_difference =
                (right[0] / total_negative).sqrt() - (right[1] / total_positive).sqrt();
            (left_difference * left_difference + right_difference * right_difference).sqrt()
        }
    }
}

fn entropy(counts: &[f64]) -> f64 {
    let mut entropy = 0.0;
    let mut total = 0.0;
    for count in counts.iter().filter(|count| **count > 0.0) {
        entropy -= count * count.log2();
        total += count;
    }
    if total == 0.0 {
        return 0.0;
    }
    (entropy + total * total.log2()) / total
}

fn gini(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 1.0;
    }
    1.0 - counts
        .iter()
        .map(|count| {
            let probability = count / total;
            probability * probability
        })
        .sum::<f64>()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .map_or(0, |(index, _)| index)
}

fn argmax_ordered(values: &[f64], order: &[usize]) -> usize {
    order
        .iter()
        .copied()
        .reduce(|best, candidate| {
            if values[candidate] > values[best] {
                candidate
            } else {
                best
            }
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> HoeffdingTreeClassifier {
        HoeffdingTreeClassifier::new(
            2,
            2,
            1,
            false,
            20.0,
            1e-7,
            0.05,
            10,
            64,
            42,
            SplitCriterion::InfoGain,
            LeafPrediction::NaiveBayesAdaptive,
            0.0,
            0.01,
            0.99,
            true,
        )
    }

    #[test]
    fn tree_learns_linear_boundary() {
        let mut tree = tree();
        let mut correct = 0;
        for i in 0..2000 {
            let x = [
                ((i * 17) % 101) as f64 / 100.0,
                ((i * 31) % 103) as f64 / 102.0,
            ];
            let y = usize::from(x[0] + x[1] > 1.0);
            correct += usize::from(tree.predict(&x) == y);
            tree.learn(&x, y, 1.0);
        }
        assert!(correct > 1500);
        assert!(tree.n_nodes() > 1);
    }

    #[test]
    fn weighted_gaussian_matches_frequency_weights() {
        let mut stats = GaussianObserver::default();
        stats.update(1.0, 2.0);
        stats.update(3.0, 1.0);
        assert!((stats.distribution.mean() - 5.0 / 3.0).abs() < 1e-12);
        assert!((stats.distribution.variance() - 4.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn extreme_values_produce_finite_probabilities() {
        let mut tree = tree();
        tree.learn(&[1e308, -1e308], 0, 1.0);
        tree.learn(&[-1e308, 1e308], 1, 1.0);
        let probabilities = tree.predict_proba(&[1e308, 1e308]);
        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_mean_stays_finite_for_opposite_extremes() {
        let mut stats = GaussianObserver::default();
        stats.update(-1e308, 1.0);
        stats.update(1e308, 1.0);
        assert!(stats.distribution.mean().is_finite());
        assert_eq!(stats.distribution.mean(), 0.0);
    }

    #[test]
    fn invalid_feature_split_does_not_deactivate_leaf() {
        let mut tree = tree();
        tree.grace_period = 1.0;
        tree.learn(&[1.0, 1.0], 0, 1.0);
        tree.learn(&[1.0, 1.0], 1, 1.0);
        assert_eq!(tree.n_active_leaves(), 1);
    }

    #[test]
    fn dominant_leaf_is_deactivated() {
        let mut tree = tree();
        tree.grace_period = 1.0;
        tree.max_share_to_split = 0.8;
        tree.learn(&[0.0, 0.0], 0, 9.0);
        tree.learn(&[1.0, 1.0], 1, 1.0);
        assert_eq!(tree.n_active_leaves(), 0);
    }

    #[test]
    fn equal_merits_preserve_feature_order() {
        let mut tree = tree();
        tree.max_features = 2;
        tree.grace_period = 4.0;
        tree.delta = 0.9;
        tree.tau = 1.0;
        for (value, class_index) in [(-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1)] {
            tree.learn(&[value, value], class_index, 1.0);
        }
        assert_eq!(tree.branches[tree.root.index()].feature, 1);
    }

    #[test]
    fn random_leaf_allocates_only_selected_observers() {
        let mut tree = HoeffdingTreeClassifier::new(
            10_000,
            2,
            3,
            false,
            20.0,
            1e-7,
            0.05,
            10,
            64,
            42,
            SplitCriterion::InfoGain,
            LeafPrediction::NaiveBayesAdaptive,
            0.0,
            0.01,
            0.99,
            true,
        );
        tree.learn(&vec![0.0; 10_000], 0, 1.0);
        let leaf = &tree.leaves[tree.root.index()];
        assert_eq!(leaf.features.len(), 3);
        assert_eq!(
            leaf.features.iter().copied().collect::<HashSet<_>>().len(),
            3
        );
        assert!(leaf.features.iter().all(|feature| *feature < 10_000));
        assert_eq!(leaf.observers.len(), 6);
    }

    #[test]
    fn label_prediction_matches_probability_argmax() {
        let mut tree = tree();
        for i in 0..1_000 {
            let x = [i as f64 / 1_000.0, (i * 17 % 101) as f64 / 100.0];
            tree.learn(&x, usize::from(x[0] + x[1] > 1.0), 1.0);
        }
        for i in 0..100 {
            let x = [i as f64 / 100.0, 1.0 - i as f64 / 100.0];
            assert_eq!(tree.predict(&x), argmax(&tree.predict_proba(&x)));
        }
    }

    #[test]
    fn random_stream_matches_python() {
        let mut rng = Rng::new(42);
        assert_eq!(
            (0..5).map(|_| rng.getrandbits(4)).collect::<Vec<_>>(),
            vec![10, 1, 0, 11, 4]
        );
        assert_eq!(
            rng.sample_values(&(0..10).collect::<Vec<_>>(), 3),
            vec![3, 9, 2]
        );
    }

    #[test]
    fn tree_grows_features_and_classes() {
        let mut tree = tree();
        tree.learn(&[0.0, 1.0], 0, 1.0);
        tree.add_features(1);
        tree.add_class();
        tree.learn(&[f64::NAN, 1.0, -1.0], 2, 1.0);

        assert_eq!(tree.n_features(), 3);
        assert_eq!(tree.n_classes(), 3);
        assert_eq!(tree.predict_proba(&[0.0, f64::NAN, 1.0]).len(), 3);
    }

    #[test]
    fn arena_and_snapshot_stay_dense() {
        let mut tree = tree();
        tree.grace_period = 2.0;
        tree.delta = 0.9;
        tree.tau = 1.0;
        tree.max_depth = 8;
        for i in 0..5_000 {
            let x = [((i * 17) % 101) as f64, ((i * 31) % 103) as f64];
            tree.learn(&x, i % 2, 1.0);
        }

        let nodes = tree.nodes();
        assert_eq!(nodes.len(), tree.n_nodes());
        assert_eq!(tree.n_leaves(), tree.n_branches() + 1);
        for node in &nodes {
            if node.0 < 0 {
                continue;
            }
            let left = &nodes[node.2];
            let right = &nodes[node.3];
            assert_eq!(left.5, node.5 + 1);
            assert_eq!(right.5, node.5 + 1);
            for class_index in 0..tree.n_classes {
                assert!(
                    (node.4[class_index] - left.4[class_index] - right.4[class_index]).abs()
                        < 1e-12
                );
            }
        }
    }
}
