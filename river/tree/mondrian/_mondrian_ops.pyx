# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport exp, log, log1p

cdef double _LOG_HALF = log(0.5)


cpdef double log_sum_2_exp_c(double a, double b):
    """Compute log((exp(a) + exp(b)) / 2) in a numerically stable way."""
    if a > b:
        return a + _LOG_HALF + log1p(exp(b - a))
    else:
        return b + _LOG_HALF + log1p(exp(a - b))


cpdef tuple range_extension_c(dict range_min, dict range_max, dict x):
    """Compute range extension sum and per-feature extension dict.

    Only features present in x are considered. Features not yet known to the
    node (absent from range_min) contribute zero extension.
    """
    cdef double extensions_sum = 0.0
    cdef double x_f, feat_min, feat_max, diff
    cdef dict extensions = {}
    for f in x:
        x_f = <double>x[f]
        if f in range_min:
            feat_min = <double>range_min[f]
            feat_max = <double>range_max[f]
            if x_f < feat_min:
                diff = feat_min - x_f
            elif x_f > feat_max:
                diff = x_f - feat_max
            else:
                continue
            extensions[f] = diff
            extensions_sum += diff
    return extensions_sum, extensions


cpdef void update_ranges_c(dict range_min, dict range_max, dict x):
    """Update node ranges with new sample (in-place).

    Only features present in x are updated.
    """
    cdef double x_f, cur_min, cur_max
    for f in x:
        x_f = <double>x[f]
        if f in range_min:
            cur_min = <double>range_min[f]
            cur_max = <double>range_max[f]
            if x_f < cur_min:
                range_min[f] = x_f
            if x_f > cur_max:
                range_max[f] = x_f
        else:
            range_min[f] = x_f
            range_max[f] = x_f


cpdef list predict_scores_c(list counts, int n_counts, int n_classes, double dirichlet, int n_samples):
    """Compute Dirichlet-smoothed class probabilities."""
    cdef double denom = n_samples + dirichlet * n_classes
    cdef int i
    cdef double c
    cdef list scores = [0.0] * n_classes
    for i in range(n_classes):
        c = <double>counts[i] if i < n_counts else 0.0
        scores[i] = (c + dirichlet) / denom
    return scores


cpdef list aggregate_predictions_c(list scores, list pred_new, int n_classes, double half_w):
    """Weighted blend: half_w * pred_new + (1 - half_w) * scores. Returns new list."""
    cdef double complement = 1.0 - half_w
    cdef int i
    cdef list result = [0.0] * n_classes
    for i in range(n_classes):
        result[i] = half_w * <double>pred_new[i] + complement * <double>scores[i]
    return result


cpdef void update_downwards_classifier_c(
    object node,
    dict x,
    int y_idx,
    double dirichlet,
    bint use_aggregation,
    double step,
    bint do_update_weight,
    int n_classes,
):
    """Perform update_downwards for classifier nodes entirely in Cython."""
    cdef int n_samples = node.n_samples
    cdef list counts
    cdef int count_val
    cdef double sc

    # Update ranges
    if n_samples == 0:
        node.memory_range_min = dict(x)
        node.memory_range_max = dict(x)
    else:
        update_ranges_c(node.memory_range_min, node.memory_range_max, x)

    # Increment sample count
    n_samples += 1
    node.n_samples = n_samples

    # Weight update
    if do_update_weight and use_aggregation:
        counts = node.counts
        count_val = <int>counts[y_idx] if y_idx < len(counts) else 0
        sc = (count_val + dirichlet) / (n_samples + dirichlet * n_classes)
        node.weight = node.weight + step * log(sc)

    # Count update
    counts = node.counts
    if y_idx >= len(counts):
        counts.extend([0] * (y_idx + 1 - len(counts)))
    counts[y_idx] = <int>counts[y_idx] + 1


cpdef void update_downwards_regressor_c(
    object node,
    dict x,
    double sample_value,
    bint use_aggregation,
    double step,
    bint do_update_weight,
):
    """Perform update_downwards for regressor nodes entirely in Cython."""
    cdef int n_samples = node.n_samples
    cdef double prediction, r, loss_t

    # Update ranges
    if n_samples == 0:
        node.memory_range_min = dict(x)
        node.memory_range_max = dict(x)
    else:
        update_ranges_c(node.memory_range_min, node.memory_range_max, x)

    # Increment sample count
    n_samples += 1
    node.n_samples = n_samples

    # Weight update (inline loss + update_weight)
    if do_update_weight and use_aggregation:
        prediction = <double>node._mean.get()
        r = prediction - sample_value
        loss_t = r * r / 2.0
        node.weight = node.weight - step * loss_t

    # Update mean using stats.Mean
    node._mean.update(sample_value)


cpdef object go_downwards_classifier_c(
    object root,
    dict x,
    int y_idx,
    int n_classes,
    double dirichlet,
    bint use_aggregation,
    double step,
    bint split_pure,
    int iteration,
    int max_nodes,
    int n_nodes,
    object rng_random,
    object rng_choices,
    object rng_uniform,
    object split_fn,
):
    """Full _go_downwards loop for classifier in Cython.

    Returns (leaf_node, new_root_or_None, n_nodes_added).
    """
    cdef object current_node = root
    cdef dict extensions
    cdef list counts, ext_features, ext_weights
    cdef double extensions_sum, split_time, split_time_candidate, T
    cdef double x_f, range_min_f, range_max_f, threshold, child_time
    cdef int count_val, branch_no, nodes_added
    cdef bint do_split_check, is_right_extension, was_leaf, is_leaf
    cdef object left, right, parent, feature
    cdef object new_root = None
    nodes_added = 0

    if iteration == 0:
        update_downwards_classifier_c(
            current_node, x, y_idx, dirichlet, use_aggregation, step,
            False, n_classes,
        )
        return current_node, new_root, nodes_added

    branch_no = -1
    while True:
        # Compute range extension
        extensions_sum, extensions = range_extension_c(
            current_node.memory_range_min, current_node.memory_range_max, x
        )

        # Compute split time
        split_time = 0.0
        if max_nodes >= 0 and (n_nodes + nodes_added) >= max_nodes:
            pass  # max_nodes reached, no split
        elif extensions_sum > 0:
            do_split_check = split_pure
            if not do_split_check:
                counts = current_node.counts
                count_val = <int>counts[y_idx] if y_idx < len(counts) else 0
                if current_node.n_samples != count_val:
                    do_split_check = True
            if do_split_check:
                T = -log(1.0 - <double>rng_random()) / extensions_sum
                split_time_candidate = <double>current_node.time + T
                is_leaf = current_node.is_leaf
                if is_leaf:
                    split_time = split_time_candidate
                else:
                    child_time = <double>current_node.children[0].time
                    if split_time_candidate < child_time:
                        split_time = split_time_candidate

        if split_time > 0:
            # Select split feature weighted by extensions (sorted for determinism)
            ext_features = sorted(extensions.keys())
            ext_weights = [extensions[f] for f in ext_features]
            feature = rng_choices(ext_features, ext_weights, k=1)[0]

            x_f = <double>x[feature]
            range_min_f = <double>current_node.memory_range_min[feature]
            range_max_f = <double>current_node.memory_range_max[feature]
            is_right_extension = x_f > range_max_f
            if is_right_extension:
                threshold = <double>rng_uniform(range_max_f, x_f)
            else:
                threshold = <double>rng_uniform(x_f, range_min_f)

            was_leaf = current_node.is_leaf
            current_node = split_fn(
                current_node, split_time, threshold,
                feature, is_right_extension,
            )
            nodes_added += 2

            if current_node.parent is None:
                new_root = current_node
            elif was_leaf:
                parent = current_node.parent
                if branch_no == 0:
                    parent.children = (current_node, parent.children[1])
                else:
                    parent.children = (parent.children[0], current_node)

            update_downwards_classifier_c(
                current_node, x, y_idx, dirichlet, use_aggregation, step,
                True, n_classes,
            )

            left, right = current_node.children
            if is_right_extension:
                current_node = right
            else:
                current_node = left

            update_downwards_classifier_c(
                current_node, x, y_idx, dirichlet, use_aggregation, step,
                False, n_classes,
            )
            return current_node, new_root, nodes_added
        else:
            update_downwards_classifier_c(
                current_node, x, y_idx, dirichlet, use_aggregation, step,
                True, n_classes,
            )
            if current_node.is_leaf:
                return current_node, new_root, nodes_added
            else:
                feature = current_node.feature
                if feature in x:
                    if <double>x[feature] <= <double>current_node.threshold:
                        branch_no = 0
                        current_node = current_node.children[0]
                    else:
                        branch_no = 1
                        current_node = current_node.children[1]
                else:
                    branch_no, current_node = current_node.most_common_path()


cpdef object go_downwards_regressor_c(
    object root,
    dict x,
    double sample_value,
    bint use_aggregation,
    double step,
    int iteration,
    int max_nodes,
    int n_nodes,
    object rng_random,
    object rng_choices,
    object rng_uniform,
    object split_fn,
):
    """Full _go_downwards loop for regressor in Cython.

    Returns (leaf_node, new_root_or_None, n_nodes_added).
    """
    cdef object current_node = root
    cdef dict extensions
    cdef list ext_features, ext_weights
    cdef double extensions_sum, split_time, split_time_candidate, T
    cdef double x_f, range_min_f, range_max_f, threshold, child_time
    cdef int branch_no, nodes_added
    cdef bint is_right_extension, was_leaf, is_leaf
    cdef object left, right, parent, feature
    cdef object new_root = None
    nodes_added = 0

    if iteration == 0:
        update_downwards_regressor_c(
            current_node, x, sample_value, use_aggregation, step,
            False,
        )
        return current_node, new_root, nodes_added

    branch_no = -1
    while True:
        extensions_sum, extensions = range_extension_c(
            current_node.memory_range_min, current_node.memory_range_max, x
        )

        split_time = 0.0
        if max_nodes >= 0 and (n_nodes + nodes_added) >= max_nodes:
            pass  # max_nodes reached, no split
        elif extensions_sum > 0:
            T = -log(1.0 - <double>rng_random()) / extensions_sum
            split_time_candidate = <double>current_node.time + T
            is_leaf = current_node.is_leaf
            if is_leaf:
                split_time = split_time_candidate
            else:
                child_time = <double>current_node.children[0].time
                if split_time_candidate < child_time:
                    split_time = split_time_candidate

        if split_time > 0:
            # Select split feature weighted by extensions (sorted for determinism)
            ext_features = sorted(extensions.keys())
            ext_weights = [extensions[f] for f in ext_features]
            feature = rng_choices(ext_features, ext_weights, k=1)[0]

            x_f = <double>x[feature]
            range_min_f = <double>current_node.memory_range_min[feature]
            range_max_f = <double>current_node.memory_range_max[feature]
            is_right_extension = x_f > range_max_f
            if is_right_extension:
                threshold = <double>rng_uniform(range_max_f, x_f)
            else:
                threshold = <double>rng_uniform(x_f, range_min_f)

            was_leaf = current_node.is_leaf
            current_node = split_fn(
                current_node, split_time, threshold,
                feature, is_right_extension,
            )
            nodes_added += 2

            if current_node.parent is None:
                new_root = current_node
            elif was_leaf:
                parent = current_node.parent
                if branch_no == 0:
                    parent.children = (current_node, parent.children[1])
                else:
                    parent.children = (parent.children[0], current_node)

            update_downwards_regressor_c(
                current_node, x, sample_value, use_aggregation, step,
                True,
            )

            left, right = current_node.children
            if is_right_extension:
                current_node = right
            else:
                current_node = left

            update_downwards_regressor_c(
                current_node, x, sample_value, use_aggregation, step,
                False,
            )
            return current_node, new_root, nodes_added
        else:
            update_downwards_regressor_c(
                current_node, x, sample_value, use_aggregation, step,
                True,
            )
            if current_node.is_leaf:
                return current_node, new_root, nodes_added
            else:
                feature = current_node.feature
                if feature in x:
                    if <double>x[feature] <= <double>current_node.threshold:
                        branch_no = 0
                        current_node = current_node.children[0]
                    else:
                        branch_no = 1
                        current_node = current_node.children[1]
                else:
                    branch_no, current_node = current_node.most_common_path()


cpdef list predict_proba_upward_c(object leaf, int n_classes, double dirichlet):
    """Walk from leaf to root aggregating predictions. Returns normalized scores."""
    cdef list scores, counts, pred_new
    cdef int n_counts, i
    cdef double weight, log_weight_tree, w, half_w, complement
    cdef double total, inv_total, denom, c
    cdef object current

    # Leaf prediction (inline predict_scores_c to avoid function call)
    current = leaf
    counts = current.counts
    n_counts = len(counts)
    denom = current.n_samples + dirichlet * n_classes
    scores = [0.0] * n_classes
    for i in range(n_classes):
        c = <double>counts[i] if i < n_counts else 0.0
        scores[i] = (c + dirichlet) / denom

    # Walk up to root
    current = current.parent
    while current is not None:
        weight = current.weight
        log_weight_tree = current.log_weight_tree
        w = exp(weight - log_weight_tree)
        half_w = 0.5 * w
        complement = 1.0 - half_w

        # Inline prediction for this node
        counts = current.counts
        n_counts = len(counts)
        denom = current.n_samples + dirichlet * n_classes
        pred_new = [0.0] * n_classes
        for i in range(n_classes):
            c = <double>counts[i] if i < n_counts else 0.0
            pred_new[i] = (c + dirichlet) / denom

        # Aggregate in-place
        for i in range(n_classes):
            scores[i] = half_w * <double>pred_new[i] + complement * <double>scores[i]

        current = current.parent

    # Normalize
    total = 0.0
    for i in range(n_classes):
        total += <double>scores[i]
    if total > 0.0 and total == total:  # not NaN
        inv_total = 1.0 / total
        for i in range(n_classes):
            scores[i] = <double>scores[i] * inv_total

    return scores
