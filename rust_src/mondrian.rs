use crate::mean::Mean;
use crate::random::Random;
use crate::stats::Univariate;
use crate::tree_arena::NodeId;
use serde::{Deserialize, Serialize};

const LOG_HALF: f64 = -std::f64::consts::LN_2;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct FeatureRange {
    min: f64,
    max: f64,
    observed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Node<T> {
    time: f64,
    ranges: Vec<FeatureRange>,
    weight: f64,
    log_weight_tree: f64,
    n_samples: u64,
    target: T,
}

impl<T: Default> Node<T> {
    fn new(time: f64, n_features: usize) -> Self {
        Self {
            time,
            ranges: vec![FeatureRange::default(); n_features],
            weight: 0.0,
            log_weight_tree: 0.0,
            n_samples: 0,
            target: T::default(),
        }
    }
}

impl<T> Node<T> {
    fn update_ranges(&mut self, values: &[f64], present: &[bool]) {
        for (feature, range) in self.ranges.iter_mut().enumerate() {
            if !present[feature] {
                continue;
            }
            let value = values[feature];
            if !range.observed {
                range.min = value;
                range.max = value;
                range.observed = true;
            } else if value < range.min {
                range.min = value;
            } else if value > range.max {
                range.max = value;
            }
        }
    }

    fn extensions(
        &self,
        values: &[f64],
        present: &[bool],
        feature_order: &[usize],
    ) -> Vec<(usize, f64)> {
        feature_order
            .iter()
            .filter_map(|feature| self.extension(*feature, values, present))
            .collect()
    }

    fn extension_sum(&self, values: &[f64], present: &[bool], feature_order: &[usize]) -> f64 {
        feature_order
            .iter()
            .filter_map(|feature| self.extension(*feature, values, present))
            .map(|(_, extension)| extension)
            .sum()
    }

    fn extension(&self, feature: usize, values: &[f64], present: &[bool]) -> Option<(usize, f64)> {
        let range = self.ranges[feature];
        if !present[feature] || !range.observed {
            return None;
        }
        let value = values[feature];
        let extension = if value < range.min {
            range.min - value
        } else if value > range.max {
            value - range.max
        } else {
            0.0
        };
        (extension > 0.0).then_some((feature, extension))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Branch<T> {
    node: Node<T>,
    feature: usize,
    threshold: f64,
    left: NodeId,
    right: NodeId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Arena<T> {
    root: NodeId,
    branches: Vec<Branch<T>>,
    leaves: Vec<Node<T>>,
    n_features: usize,
    rng: Random,
}

impl<T: Clone + Default> Arena<T> {
    fn new(n_features: usize, rng: Random) -> Self {
        Self {
            root: NodeId::leaf(0),
            branches: Vec::new(),
            leaves: vec![Node::new(0.0, n_features)],
            n_features,
            rng,
        }
    }

    fn node(&self, node: NodeId) -> &Node<T> {
        if node.is_leaf() {
            &self.leaves[node.index()]
        } else {
            &self.branches[node.index()].node
        }
    }

    fn node_mut(&mut self, node: NodeId) -> &mut Node<T> {
        if node.is_leaf() {
            &mut self.leaves[node.index()]
        } else {
            &mut self.branches[node.index()].node
        }
    }

    fn n_nodes(&self) -> usize {
        self.branches.len() + self.leaves.len()
    }

    fn add_features(&mut self, count: usize) {
        self.n_features += count;
        for branch in &mut self.branches {
            branch
                .node
                .ranges
                .resize(self.n_features, FeatureRange::default());
        }
        for leaf in &mut self.leaves {
            leaf.ranges.resize(self.n_features, FeatureRange::default());
        }
    }

    fn child(&self, branch: usize, values: &[f64], present: &[bool]) -> NodeId {
        let branch = &self.branches[branch];
        if present[branch.feature] {
            if values[branch.feature] <= branch.threshold {
                branch.left
            } else {
                branch.right
            }
        } else if self.node(branch.left).weight < self.node(branch.right).weight {
            branch.right
        } else {
            branch.left
        }
    }

    fn path(&self, values: &[f64], present: &[bool]) -> Vec<NodeId> {
        let mut path = Vec::with_capacity(32);
        let mut node = self.root;
        loop {
            path.push(node);
            if node.is_leaf() {
                return path;
            }
            node = self.child(node.index(), values, present);
        }
    }

    fn split_candidate(
        &mut self,
        node: NodeId,
        values: &[f64],
        present: &[bool],
        feature_order: &[usize],
        can_split: bool,
    ) -> Option<(f64, Vec<(usize, f64)>)> {
        if !can_split {
            return None;
        }
        let state = self.node(node);
        let node_time = state.time;
        let sum = state.extension_sum(values, present, feature_order);
        if sum <= 0.0 {
            return None;
        }
        let split_time = node_time - (1.0 - self.rng.random()).ln() / sum;
        if split_time <= 0.0
            || (!node.is_leaf() && split_time >= self.node(self.branches[node.index()].left).time)
        {
            return None;
        }
        Some((
            split_time,
            self.node(node).extensions(values, present, feature_order),
        ))
    }

    fn replace(&mut self, old: NodeId, replacement: NodeId, parent: Option<NodeId>) {
        if let Some(parent) = parent {
            let parent = &mut self.branches[parent.index()];
            if parent.left == old {
                parent.left = replacement;
            } else {
                assert_eq!(parent.right, old);
                parent.right = replacement;
            }
        } else {
            assert_eq!(self.root, old);
            self.root = replacement;
        }
    }

    fn split(
        &mut self,
        node: NodeId,
        parent: Option<NodeId>,
        split_time: f64,
        threshold: f64,
        feature: usize,
        is_right_extension: bool,
    ) -> (NodeId, NodeId) {
        if node.is_leaf() {
            let leaf_index = node.index();
            let original = self.leaves[leaf_index].clone();
            let mut inherited = original.clone();
            inherited.time = split_time;
            let new_leaf = Node::new(split_time, self.n_features);
            let appended = NodeId::leaf(self.leaves.len());
            let (left, right, sample_leaf) = if is_right_extension {
                self.leaves[leaf_index] = inherited;
                self.leaves.push(new_leaf);
                (node, appended, appended)
            } else {
                self.leaves[leaf_index] = new_leaf;
                self.leaves.push(inherited);
                (node, appended, node)
            };
            let branch = NodeId::branch(self.branches.len());
            self.branches.push(Branch {
                node: original,
                feature,
                threshold,
                left,
                right,
            });
            self.replace(node, branch, parent);
            (branch, sample_leaf)
        } else {
            let branch_index = node.index();
            let original = self.branches[branch_index].clone();
            let mut inherited_node = original.node.clone();
            inherited_node.time = split_time;
            let inherited = NodeId::branch(self.branches.len());
            self.branches.push(Branch {
                node: inherited_node,
                feature: original.feature,
                threshold: original.threshold,
                left: original.left,
                right: original.right,
            });
            let sample_leaf = NodeId::leaf(self.leaves.len());
            self.leaves.push(Node::new(split_time, self.n_features));
            let branch = &mut self.branches[branch_index];
            branch.feature = feature;
            branch.threshold = threshold;
            if is_right_extension {
                branch.left = inherited;
                branch.right = sample_leaf;
            } else {
                branch.left = sample_leaf;
                branch.right = inherited;
            }
            (node, sample_leaf)
        }
    }

    fn update_upwards(&mut self, path: &[NodeId]) {
        for node in path.iter().rev() {
            let log_weight_tree = if node.is_leaf() {
                self.node(*node).weight
            } else {
                let branch = &self.branches[node.index()];
                log_sum_2_exp(
                    branch.node.weight,
                    self.node(branch.left).log_weight_tree
                        + self.node(branch.right).log_weight_tree,
                )
            };
            self.node_mut(*node).log_weight_tree = log_weight_tree;
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ClassificationTarget {
    counts: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MondrianTreeClassifier {
    arena: Arena<ClassificationTarget>,
    n_classes: usize,
    path: Vec<NodeId>,
}

impl MondrianTreeClassifier {
    pub fn seeded(n_features: usize, n_classes: usize, seed: u64) -> Self {
        Self::from_rng(n_features, n_classes, Random::new(seed))
    }

    fn from_rng(n_features: usize, n_classes: usize, rng: Random) -> Self {
        Self {
            arena: Arena::new(n_features, rng),
            n_classes,
            path: Vec::new(),
        }
    }

    pub fn add_features(&mut self, count: usize) {
        self.arena.add_features(count);
    }

    pub fn add_class(&mut self) {
        self.n_classes += 1;
    }

    pub fn n_features(&self) -> usize {
        self.arena.n_features
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub fn n_nodes(&self) -> usize {
        self.arena.n_nodes()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn learn(
        &mut self,
        values: &[f64],
        present: &[bool],
        feature_order: &[usize],
        y: usize,
        iteration: usize,
        step: f64,
        use_aggregation: bool,
        dirichlet: f64,
        split_pure: bool,
        max_nodes: Option<usize>,
    ) -> usize {
        if iteration == 0 {
            self.update_node(
                self.arena.root,
                values,
                present,
                y,
                step,
                use_aggregation,
                dirichlet,
                false,
            );
            return 0;
        }

        let mut path = std::mem::take(&mut self.path);
        path.clear();
        let mut current = self.arena.root;
        let mut nodes_added = 0;
        loop {
            path.push(current);
            let node = self.arena.node(current);
            let count = node.target.counts.get(y).copied().unwrap_or(0);
            let can_split = (split_pure || node.n_samples != count)
                && max_nodes.is_none_or(|limit| self.arena.n_nodes().saturating_add(2) <= limit);
            if let Some((split_time, extensions)) =
                self.arena
                    .split_candidate(current, values, present, feature_order, can_split)
            {
                let weights = extensions
                    .iter()
                    .map(|(_, extension)| *extension)
                    .collect::<Vec<_>>();
                let feature = extensions[self.arena.rng.weighted_index(&weights)].0;
                let range = self.arena.node(current).ranges[feature];
                let value = values[feature];
                let is_right_extension = value > range.max;
                let threshold = if is_right_extension {
                    self.arena.rng.uniform(range.max, value)
                } else {
                    self.arena.rng.uniform(value, range.min)
                };
                let parent = (path.len() > 1).then(|| path[path.len() - 2]);
                let (branch, sample_leaf) = self.arena.split(
                    current,
                    parent,
                    split_time,
                    threshold,
                    feature,
                    is_right_extension,
                );
                *path.last_mut().unwrap() = branch;
                self.update_node(
                    branch,
                    values,
                    present,
                    y,
                    step,
                    use_aggregation,
                    dirichlet,
                    true,
                );
                self.update_node(
                    sample_leaf,
                    values,
                    present,
                    y,
                    step,
                    use_aggregation,
                    dirichlet,
                    false,
                );
                path.push(sample_leaf);
                nodes_added = 2;
                break;
            }

            self.update_node(
                current,
                values,
                present,
                y,
                step,
                use_aggregation,
                dirichlet,
                true,
            );
            if current.is_leaf() {
                break;
            }
            current = self.arena.child(current.index(), values, present);
        }
        if use_aggregation {
            self.arena.update_upwards(&path);
        }
        path.clear();
        self.path = path;
        nodes_added
    }

    pub fn predict_proba(
        &self,
        values: &[f64],
        present: &[bool],
        dirichlet: f64,
        use_aggregation: bool,
    ) -> Vec<f64> {
        let path = self.arena.path(values, present);
        let mut scores = self.scores(*path.last().unwrap(), dirichlet);
        if use_aggregation {
            for node in path[..path.len() - 1].iter().rev() {
                let state = self.arena.node(*node);
                let half_weight = 0.5 * (state.weight - state.log_weight_tree).exp();
                let node_scores = self.scores(*node, dirichlet);
                for (score, node_score) in scores.iter_mut().zip(node_scores) {
                    *score = half_weight * node_score + (1.0 - half_weight) * *score;
                }
            }
            let total: f64 = scores.iter().sum();
            if total > 0.0 && !total.is_nan() {
                for score in &mut scores {
                    *score /= total;
                }
            }
        }
        scores
    }

    #[allow(clippy::too_many_arguments)]
    fn update_node(
        &mut self,
        node: NodeId,
        values: &[f64],
        present: &[bool],
        y: usize,
        step: f64,
        use_aggregation: bool,
        dirichlet: f64,
        update_weight: bool,
    ) {
        let state = self.arena.node_mut(node);
        state.update_ranges(values, present);
        state.n_samples += 1;
        let count = state.target.counts.get(y).copied().unwrap_or(0);
        if update_weight && use_aggregation {
            let score = (count as f64 + dirichlet)
                / (state.n_samples as f64 + dirichlet * self.n_classes as f64);
            state.weight += step * score.ln();
        }
        if y >= state.target.counts.len() {
            state.target.counts.resize(y + 1, 0);
        }
        state.target.counts[y] += 1;
    }

    fn scores(&self, node: NodeId, dirichlet: f64) -> Vec<f64> {
        let state = self.arena.node(node);
        let denominator = state.n_samples as f64 + dirichlet * self.n_classes as f64;
        (0..self.n_classes)
            .map(|class_index| {
                (state.target.counts.get(class_index).copied().unwrap_or(0) as f64 + dirichlet)
                    / denominator
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct RegressionTarget {
    mean: Mean<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MondrianTreeRegressor {
    arena: Arena<RegressionTarget>,
    path: Vec<NodeId>,
}

impl MondrianTreeRegressor {
    pub fn seeded(n_features: usize, seed: u64) -> Self {
        Self::from_rng(n_features, Random::new(seed))
    }

    fn from_rng(n_features: usize, rng: Random) -> Self {
        Self {
            arena: Arena::new(n_features, rng),
            path: Vec::new(),
        }
    }

    pub fn add_features(&mut self, count: usize) {
        self.arena.add_features(count);
    }

    pub fn n_features(&self) -> usize {
        self.arena.n_features
    }

    pub fn n_nodes(&self) -> usize {
        self.arena.n_nodes()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn learn(
        &mut self,
        values: &[f64],
        present: &[bool],
        feature_order: &[usize],
        y: f64,
        iteration: usize,
        step: f64,
        use_aggregation: bool,
        max_nodes: Option<usize>,
    ) -> usize {
        if iteration == 0 {
            self.update_node(
                self.arena.root,
                values,
                present,
                y,
                step,
                use_aggregation,
                false,
            );
            return 0;
        }

        let mut path = std::mem::take(&mut self.path);
        path.clear();
        let mut current = self.arena.root;
        let mut nodes_added = 0;
        loop {
            path.push(current);
            let can_split =
                max_nodes.is_none_or(|limit| self.arena.n_nodes().saturating_add(2) <= limit);
            if let Some((split_time, extensions)) =
                self.arena
                    .split_candidate(current, values, present, feature_order, can_split)
            {
                let weights = extensions
                    .iter()
                    .map(|(_, extension)| *extension)
                    .collect::<Vec<_>>();
                let feature = extensions[self.arena.rng.weighted_index(&weights)].0;
                let range = self.arena.node(current).ranges[feature];
                let value = values[feature];
                let is_right_extension = value > range.max;
                let threshold = if is_right_extension {
                    self.arena.rng.uniform(range.max, value)
                } else {
                    self.arena.rng.uniform(value, range.min)
                };
                let parent = (path.len() > 1).then(|| path[path.len() - 2]);
                let (branch, sample_leaf) = self.arena.split(
                    current,
                    parent,
                    split_time,
                    threshold,
                    feature,
                    is_right_extension,
                );
                *path.last_mut().unwrap() = branch;
                self.update_node(branch, values, present, y, step, use_aggregation, true);
                self.update_node(
                    sample_leaf,
                    values,
                    present,
                    y,
                    step,
                    use_aggregation,
                    false,
                );
                path.push(sample_leaf);
                nodes_added = 2;
                break;
            }

            self.update_node(current, values, present, y, step, use_aggregation, true);
            if current.is_leaf() {
                break;
            }
            current = self.arena.child(current.index(), values, present);
        }
        if use_aggregation {
            self.arena.update_upwards(&path);
        }
        path.clear();
        self.path = path;
        nodes_added
    }

    pub fn predict(&self, values: &[f64], present: &[bool], use_aggregation: bool) -> f64 {
        let path = self.arena.path(values, present);
        let mut prediction = self.arena.node(*path.last().unwrap()).target.mean.get();
        if use_aggregation {
            for node in path[..path.len() - 1].iter().rev() {
                let state = self.arena.node(*node);
                let half_weight = 0.5 * (state.weight - state.log_weight_tree).exp();
                prediction =
                    half_weight * state.target.mean.get() + (1.0 - half_weight) * prediction;
            }
        }
        prediction
    }

    fn update_node(
        &mut self,
        node: NodeId,
        values: &[f64],
        present: &[bool],
        y: f64,
        step: f64,
        use_aggregation: bool,
        update_weight: bool,
    ) {
        let state = self.arena.node_mut(node);
        state.update_ranges(values, present);
        state.n_samples += 1;
        if update_weight && use_aggregation {
            let residual = state.target.mean.get() - y;
            state.weight -= step * residual * residual / 2.0;
        }
        state.target.mean.update(y);
    }
}

fn log_sum_2_exp(left: f64, right: f64) -> f64 {
    if left > right {
        left + LOG_HALF + (right - left).exp().ln_1p()
    } else {
        right + LOG_HALF + (left - right).exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_keeps_arena_dense() {
        let mut tree = MondrianTreeClassifier::from_rng(2, 2, Random::new(42));
        for i in 0..1_000 {
            let values = [i as f64 % 17.0, i as f64 % 31.0];
            tree.learn(
                &values,
                &[true, true],
                &[0, 1],
                i % 2,
                i,
                1.0,
                true,
                0.5,
                false,
                Some(255),
            );
        }
        assert_eq!(tree.arena.leaves.len(), tree.arena.branches.len() + 1);
        assert!(tree.n_nodes() > 1);
    }

    #[test]
    fn regressor_predicts_finite_values() {
        let mut tree = MondrianTreeRegressor::from_rng(2, Random::new(42));
        for i in 0..100 {
            let values = [i as f64 / 100.0, (i * 17 % 101) as f64 / 100.0];
            tree.learn(
                &values,
                &[true, true],
                &[0, 1],
                values[0] + values[1],
                i,
                1.0,
                true,
                Some(255),
            );
        }
        assert!(tree.predict(&[0.2, 0.4], &[true, true], true).is_finite());
    }
}
