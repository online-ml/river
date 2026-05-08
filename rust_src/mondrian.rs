// Mondrian tree numerical helpers — port of river/tree/mondrian/_mondrian_ops.pyx.
//
// All exported functions are PyO3 wrappers that operate on plain Python objects
// (dicts, lists, MondrianNode instances) so they're a drop-in replacement for
// the Cython entry points the Python code already imports.
//
// The hot inner loops touch Python objects through the standard PyO3 attribute
// and dict APIs. Cython compiles `node.attr` on a `cdef object` to the same
// `PyObject_GetAttr` call, so per-attribute cost lines up — the win comes from
// avoiding the Cython glue plus structural simplifications:
//   - the leaf-to-root `_go_upwards` walk is now a single Rust call instead of
//     N Python frame setups;
//   - prediction descends + aggregates entirely in Rust (no Python loop);
//   - in `go_downwards_*`, range bounds are fetched once and reused, and the
//     `extensions` dict is only allocated when we actually split (most loop
//     iterations descend without splitting).

use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyFloat, PyInt, PyList, PyListMethods};

const LOG_HALF: f64 = -std::f64::consts::LN_2;

#[inline]
fn log_sum_2_exp_inner(a: f64, b: f64) -> f64 {
    if a > b {
        a + LOG_HALF + (b - a).exp().ln_1p()
    } else {
        b + LOG_HALF + (a - b).exp().ln_1p()
    }
}

#[pyfunction]
pub fn log_sum_2_exp(a: f64, b: f64) -> f64 {
    log_sum_2_exp_inner(a, b)
}

/// Read a Python float-or-int into f64 using dedicated fast-paths. The generic
/// `extract::<f64>()` route falls back through `__float__` and adds a type
/// check; this saves a handful of ns per value inside the hot loops.
#[inline]
fn as_f64(v: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(f) = v.downcast::<PyFloat>() {
        return Ok(f.value());
    }
    if let Ok(i) = v.downcast::<PyInt>() {
        return i.extract::<f64>();
    }
    v.extract::<f64>()
}

/// In-place: extend `(range_min, range_max)` so they cover x. Mirrors
/// `update_ranges_c` — only features present in `x` are considered.
fn update_ranges_inner<'py>(
    range_min: &Bound<'py, PyDict>,
    range_max: &Bound<'py, PyDict>,
    x: &Bound<'py, PyDict>,
) -> PyResult<()> {
    for (k, v) in x.iter() {
        let xf = as_f64(&v)?;
        if let Some(cur_min) = range_min.get_item(&k)? {
            let mn = as_f64(&cur_min)?;
            if xf < mn {
                range_min.set_item(&k, xf)?;
            } else {
                let cur_max = range_max
                    .get_item(&k)?
                    .ok_or_else(|| PyValueError::new_err("range_max missing key"))?;
                let mx = as_f64(&cur_max)?;
                if xf > mx {
                    range_max.set_item(&k, xf)?;
                }
            }
        } else {
            range_min.set_item(&k, xf)?;
            range_max.set_item(&k, xf)?;
        }
    }
    Ok(())
}

#[pyfunction]
pub fn update_ranges<'py>(
    range_min: &Bound<'py, PyDict>,
    range_max: &Bound<'py, PyDict>,
    x: &Bound<'py, PyDict>,
) -> PyResult<()> {
    update_ranges_inner(range_min, range_max, x)
}

/// Mirrors `range_extension_c`. Returns `(extensions_sum, extensions_dict)`.
/// Features unknown to the node (absent from `range_min`) contribute 0.
fn range_extension_inner<'py>(
    py: Python<'py>,
    range_min: &Bound<'py, PyDict>,
    range_max: &Bound<'py, PyDict>,
    x: &Bound<'py, PyDict>,
) -> PyResult<(f64, Bound<'py, PyDict>)> {
    let extensions = PyDict::new(py);
    let mut sum = 0.0f64;
    for (k, v) in x.iter() {
        let xf = as_f64(&v)?;
        let Some(min_v) = range_min.get_item(&k)? else {
            continue;
        };
        let mn = as_f64(&min_v)?;
        let max_v = range_max
            .get_item(&k)?
            .ok_or_else(|| PyValueError::new_err("range_max missing key"))?;
        let mx = as_f64(&max_v)?;
        let diff = if xf < mn {
            mn - xf
        } else if xf > mx {
            xf - mx
        } else {
            continue;
        };
        extensions.set_item(&k, diff)?;
        sum += diff;
    }
    Ok((sum, extensions))
}

/// Same arithmetic as `range_extension_inner` but only returns the sum — the
/// extensions dict is only needed in the (rare) splitting path, so most
/// `go_downwards_*` iterations skip the dict allocation entirely.
fn range_extension_sum<'py>(
    range_min: &Bound<'py, PyDict>,
    range_max: &Bound<'py, PyDict>,
    x: &Bound<'py, PyDict>,
) -> PyResult<f64> {
    let mut sum = 0.0f64;
    for (k, v) in x.iter() {
        let xf = as_f64(&v)?;
        let Some(min_v) = range_min.get_item(&k)? else {
            continue;
        };
        let mn = as_f64(&min_v)?;
        if xf < mn {
            sum += mn - xf;
            continue;
        }
        let max_v = range_max
            .get_item(&k)?
            .ok_or_else(|| PyValueError::new_err("range_max missing key"))?;
        let mx = as_f64(&max_v)?;
        if xf > mx {
            sum += xf - mx;
        }
    }
    Ok(sum)
}

#[pyfunction]
pub fn range_extension<'py>(
    py: Python<'py>,
    range_min: &Bound<'py, PyDict>,
    range_max: &Bound<'py, PyDict>,
    x: &Bound<'py, PyDict>,
) -> PyResult<(f64, Bound<'py, PyDict>)> {
    range_extension_inner(py, range_min, range_max, x)
}

/// Dirichlet-smoothed class probabilities. `counts` is the node's per-class
/// count list (may be shorter than `n_classes` if some classes have not been
/// observed at this node yet).
#[pyfunction]
pub fn predict_scores<'py>(
    py: Python<'py>,
    counts: &Bound<'py, PyList>,
    n_counts: usize,
    n_classes: usize,
    dirichlet: f64,
    n_samples: i64,
) -> PyResult<Bound<'py, PyList>> {
    let denom = n_samples as f64 + dirichlet * n_classes as f64;
    let scores = PyList::new(py, std::iter::repeat_n(0.0f64, n_classes))?;
    for i in 0..n_classes {
        let c = if i < n_counts {
            as_f64(&counts.get_item(i)?)?
        } else {
            0.0
        };
        scores.set_item(i, (c + dirichlet) / denom)?;
    }
    Ok(scores)
}

// --- Internal helpers backing update_downwards ---------------------------------

/// Initialize node ranges from x (used when n_samples == 0). Equivalent to
/// `node.memory_range_min = dict(x); node.memory_range_max = dict(x)`.
fn set_ranges_from_x<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    x: &Bound<'py, PyDict>,
) -> PyResult<()> {
    node.setattr(intern!(py, "memory_range_min"), x.copy()?)?;
    node.setattr(intern!(py, "memory_range_max"), x.copy()?)?;
    Ok(())
}

/// `update_downwards` for a classifier node. `ranges` is the pre-fetched
/// `(range_min, range_max)` pair — if `None`, they're fetched here. Skipping
/// the re-fetch saves two `getattr` calls per node visit in `go_downwards`.
#[allow(clippy::too_many_arguments)]
fn update_downwards_classifier_inner<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    x: &Bound<'py, PyDict>,
    y_idx: i64,
    dirichlet: f64,
    use_aggregation: bool,
    step: f64,
    do_update_weight: bool,
    n_classes: i64,
    ranges: Option<(&Bound<'py, PyDict>, &Bound<'py, PyDict>)>,
) -> PyResult<()> {
    let n_samples_attr = intern!(py, "n_samples");
    let counts_attr = intern!(py, "counts");
    let weight_attr = intern!(py, "weight");

    let mut n_samples: i64 = node.getattr(n_samples_attr)?.extract()?;

    if n_samples == 0 {
        set_ranges_from_x(py, node, x)?;
    } else if let Some((range_min, range_max)) = ranges {
        update_ranges_inner(range_min, range_max, x)?;
    } else {
        let range_min = node
            .getattr(intern!(py, "memory_range_min"))?
            .downcast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let range_max = node
            .getattr(intern!(py, "memory_range_max"))?
            .downcast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        update_ranges_inner(&range_min, &range_max, x)?;
    }

    n_samples += 1;
    node.setattr(n_samples_attr, n_samples)?;

    let counts = node
        .getattr(counts_attr)?
        .downcast_into::<PyList>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let y_idx_usize = y_idx as usize;
    let len = counts.len();

    if do_update_weight && use_aggregation {
        let count_val = if y_idx_usize < len {
            as_f64(&counts.get_item(y_idx_usize)?)?
        } else {
            0.0
        };
        let sc =
            (count_val + dirichlet) / (n_samples as f64 + dirichlet * n_classes as f64);
        let cur_weight: f64 = node.getattr(weight_attr)?.extract()?;
        node.setattr(weight_attr, cur_weight + step * sc.ln())?;
    }

    if y_idx_usize >= len {
        for _ in len..y_idx_usize {
            counts.append(0i64)?;
        }
        counts.append(1i64)?;
    } else {
        let cur: i64 = counts.get_item(y_idx_usize)?.extract()?;
        counts.set_item(y_idx_usize, cur + 1)?;
    }
    Ok(())
}

/// `update_downwards` for a regressor node. `ranges` is the pre-fetched
/// `(range_min, range_max)` pair — if `None`, they're fetched here.
fn update_downwards_regressor_inner<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    x: &Bound<'py, PyDict>,
    sample_value: f64,
    use_aggregation: bool,
    step: f64,
    do_update_weight: bool,
    ranges: Option<(&Bound<'py, PyDict>, &Bound<'py, PyDict>)>,
) -> PyResult<()> {
    let n_samples_attr = intern!(py, "n_samples");
    let weight_attr = intern!(py, "weight");
    let mean_attr = intern!(py, "_mean");
    let get_attr = intern!(py, "get");
    let update_attr = intern!(py, "update");

    let mut n_samples: i64 = node.getattr(n_samples_attr)?.extract()?;

    if n_samples == 0 {
        set_ranges_from_x(py, node, x)?;
    } else if let Some((range_min, range_max)) = ranges {
        update_ranges_inner(range_min, range_max, x)?;
    } else {
        let range_min = node
            .getattr(intern!(py, "memory_range_min"))?
            .downcast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let range_max = node
            .getattr(intern!(py, "memory_range_max"))?
            .downcast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        update_ranges_inner(&range_min, &range_max, x)?;
    }

    n_samples += 1;
    node.setattr(n_samples_attr, n_samples)?;

    let mean_obj = node.getattr(mean_attr)?;
    if do_update_weight && use_aggregation {
        let prediction: f64 = mean_obj.call_method0(get_attr)?.extract()?;
        let r = prediction - sample_value;
        let loss_t = r * r / 2.0;
        let cur_weight: f64 = node.getattr(weight_attr)?.extract()?;
        node.setattr(weight_attr, cur_weight - step * loss_t)?;
    }
    mean_obj.call_method1(update_attr, (sample_value,))?;
    Ok(())
}

/// Sample a feature from `extensions` weighted by extension size. Mirrors
/// the deterministic Cython path: keys are sorted, then `rng.choices` is
/// invoked with the corresponding weights.
fn weighted_choice<'py>(
    py: Python<'py>,
    extensions: &Bound<'py, PyDict>,
    rng_choices: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let keys = extensions.keys();
    let builtins = py.import(intern!(py, "builtins"))?;
    let sorted = builtins.getattr(intern!(py, "sorted"))?;
    let sorted_keys_list = sorted.call1((keys,))?;
    let sorted_keys_list = sorted_keys_list
        .downcast_into::<PyList>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let weights = PyList::empty(py);
    for k in sorted_keys_list.iter() {
        let v = extensions
            .get_item(&k)?
            .ok_or_else(|| PyValueError::new_err("extension key missing"))?;
        weights.append(v)?;
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "k"), 1)?;
    let chosen = rng_choices.call((sorted_keys_list, weights), Some(&kwargs))?;
    let item = chosen.get_item(0)?;
    Ok(item)
}

// --- Tree-walking entry points -------------------------------------------------

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn go_downwards_classifier<'py>(
    py: Python<'py>,
    root: Bound<'py, PyAny>,
    x: Bound<'py, PyDict>,
    y_idx: i64,
    n_classes: i64,
    dirichlet: f64,
    use_aggregation: bool,
    step: f64,
    split_pure: bool,
    iteration: i64,
    max_nodes: i64,
    n_nodes: i64,
    rng_random: Bound<'py, PyAny>,
    rng_choices: Bound<'py, PyAny>,
    rng_uniform: Bound<'py, PyAny>,
    split_fn: Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>, i64)> {
    let counts_attr = intern!(py, "counts");
    let n_samples_attr = intern!(py, "n_samples");
    let time_attr = intern!(py, "time");
    let is_leaf_attr = intern!(py, "is_leaf");
    let children_attr = intern!(py, "children");
    let parent_attr = intern!(py, "parent");
    let feature_attr = intern!(py, "feature");
    let threshold_attr = intern!(py, "threshold");
    let memory_range_min_attr = intern!(py, "memory_range_min");
    let memory_range_max_attr = intern!(py, "memory_range_max");
    let most_common_path_attr = intern!(py, "most_common_path");

    let mut current = root;
    let mut new_root: Option<Py<PyAny>> = None;
    let mut nodes_added: i64 = 0;

    if iteration == 0 {
        update_downwards_classifier_inner(
            py,
            &current,
            &x,
            y_idx,
            dirichlet,
            use_aggregation,
            step,
            false,
            n_classes,
            None,
        )?;
        return Ok((current.unbind(), new_root, nodes_added));
    }

    let mut branch_no: i32 = -1;
    loop {
        let mut split_time = 0.0f64;
        // `range_min` / `range_max` are reused across the candidate-split
        // check, the actual split block, and the descend update — fetched
        // once when needed to avoid 2-3 redundant `getattr`s per iteration.
        let mut range_bounds: Option<(Bound<'py, PyDict>, Bound<'py, PyDict>)> = None;

        // Skip the (expensive) range_extension call for pure non-split nodes
        // (matches the Cython optimisation introduced in #1841).
        let mut do_split_check = split_pure;
        if !do_split_check {
            let counts = current
                .getattr(counts_attr)?
                .downcast_into::<PyList>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let n_samples_here: i64 = current.getattr(n_samples_attr)?.extract()?;
            let count_val: i64 = if (y_idx as usize) < counts.len() {
                counts.get_item(y_idx as usize)?.extract()?
            } else {
                0
            };
            if n_samples_here != count_val {
                do_split_check = true;
            }
        }

        if do_split_check && !(max_nodes >= 0 && (n_nodes + nodes_added) >= max_nodes) {
            let range_min = current
                .getattr(memory_range_min_attr)?
                .downcast_into::<PyDict>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let range_max = current
                .getattr(memory_range_max_attr)?
                .downcast_into::<PyDict>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            // Cheap sum-only pass; the extensions dict is built lazily only
            // if we end up splitting (the common case is to descend).
            let sum = range_extension_sum(&range_min, &range_max, &x)?;
            if sum > 0.0 {
                let r: f64 = rng_random.call0()?.extract()?;
                let t = -((1.0 - r).ln()) / sum;
                let node_time: f64 = current.getattr(time_attr)?.extract()?;
                let candidate = node_time + t;
                let is_leaf: bool = current.getattr(is_leaf_attr)?.extract()?;
                if is_leaf {
                    split_time = candidate;
                } else {
                    let children = current.getattr(children_attr)?;
                    let left_child = children.get_item(0)?;
                    let child_time: f64 = left_child.getattr(time_attr)?.extract()?;
                    if candidate < child_time {
                        split_time = candidate;
                    }
                }
            }
            range_bounds = Some((range_min, range_max));
        }

        if split_time > 0.0 {
            let (range_min, range_max) =
                range_bounds.expect("range bounds set when split_time > 0");
            let (_, extensions) = range_extension_inner(py, &range_min, &range_max, &x)?;
            let feature = weighted_choice(py, &extensions, &rng_choices)?;

            let xf = as_f64(
                &x.get_item(&feature)?
                    .ok_or_else(|| PyValueError::new_err("feature missing in x"))?,
            )?;
            let range_min_f = as_f64(
                &range_min
                    .get_item(&feature)?
                    .ok_or_else(|| PyValueError::new_err("range_min feature missing"))?,
            )?;
            let range_max_f = as_f64(
                &range_max
                    .get_item(&feature)?
                    .ok_or_else(|| PyValueError::new_err("range_max feature missing"))?,
            )?;

            let is_right_extension = xf > range_max_f;
            let threshold: f64 = if is_right_extension {
                rng_uniform.call1((range_max_f, xf))?.extract()?
            } else {
                rng_uniform.call1((xf, range_min_f))?.extract()?
            };

            let was_leaf: bool = current.getattr(is_leaf_attr)?.extract()?;
            let new_node = split_fn.call1((
                current.clone(),
                split_time,
                threshold,
                feature,
                is_right_extension,
            ))?;
            current = new_node;
            nodes_added += 2;

            let parent = current.getattr(parent_attr)?;
            if parent.is_none() {
                new_root = Some(current.clone().unbind());
            } else if was_leaf {
                let parent_children = parent.getattr(children_attr)?;
                let left = parent_children.get_item(0)?;
                let right = parent_children.get_item(1)?;
                let new_children = if branch_no == 0 {
                    (current.clone(), right)
                } else {
                    (left, current.clone())
                };
                parent.setattr(children_attr, new_children)?;
            }

            update_downwards_classifier_inner(
                py,
                &current,
                &x,
                y_idx,
                dirichlet,
                use_aggregation,
                step,
                true,
                n_classes,
                None,
            )?;

            let children = current.getattr(children_attr)?;
            let leaf_child = if is_right_extension {
                children.get_item(1)?
            } else {
                children.get_item(0)?
            };
            update_downwards_classifier_inner(
                py,
                &leaf_child,
                &x,
                y_idx,
                dirichlet,
                use_aggregation,
                step,
                false,
                n_classes,
                None,
            )?;
            return Ok((leaf_child.unbind(), new_root, nodes_added));
        } else {
            let ranges = range_bounds.as_ref().map(|(rmin, rmax)| (rmin, rmax));
            update_downwards_classifier_inner(
                py,
                &current,
                &x,
                y_idx,
                dirichlet,
                use_aggregation,
                step,
                true,
                n_classes,
                ranges,
            )?;
            let is_leaf: bool = current.getattr(is_leaf_attr)?.extract()?;
            if is_leaf {
                return Ok((current.unbind(), new_root, nodes_added));
            }
            let feature = current.getattr(feature_attr)?;
            unsafe {
                let res = ffi::PyDict_Contains(x.as_ptr(), feature.as_ptr());
                if res < 0 {
                    return Err(PyErr::fetch(py));
                }
                if res == 1 {
                    let xf = as_f64(
                        &x.get_item(&feature)?
                            .ok_or_else(|| PyValueError::new_err("feature missing in x"))?,
                    )?;
                    let threshold: f64 = current.getattr(threshold_attr)?.extract()?;
                    let children = current.getattr(children_attr)?;
                    if xf <= threshold {
                        branch_no = 0;
                        current = children.get_item(0)?;
                    } else {
                        branch_no = 1;
                        current = children.get_item(1)?;
                    }
                } else {
                    let result = current.call_method0(most_common_path_attr)?;
                    branch_no = result.get_item(0)?.extract()?;
                    current = result.get_item(1)?;
                }
            }
        }
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn go_downwards_regressor<'py>(
    py: Python<'py>,
    root: Bound<'py, PyAny>,
    x: Bound<'py, PyDict>,
    sample_value: f64,
    use_aggregation: bool,
    step: f64,
    iteration: i64,
    max_nodes: i64,
    n_nodes: i64,
    rng_random: Bound<'py, PyAny>,
    rng_choices: Bound<'py, PyAny>,
    rng_uniform: Bound<'py, PyAny>,
    split_fn: Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>, i64)> {
    let time_attr = intern!(py, "time");
    let is_leaf_attr = intern!(py, "is_leaf");
    let children_attr = intern!(py, "children");
    let parent_attr = intern!(py, "parent");
    let feature_attr = intern!(py, "feature");
    let threshold_attr = intern!(py, "threshold");
    let memory_range_min_attr = intern!(py, "memory_range_min");
    let memory_range_max_attr = intern!(py, "memory_range_max");
    let most_common_path_attr = intern!(py, "most_common_path");

    let mut current = root;
    let mut new_root: Option<Py<PyAny>> = None;
    let mut nodes_added: i64 = 0;

    if iteration == 0 {
        update_downwards_regressor_inner(
            py,
            &current,
            &x,
            sample_value,
            use_aggregation,
            step,
            false,
            None,
        )?;
        return Ok((current.unbind(), new_root, nodes_added));
    }

    let mut branch_no: i32 = -1;
    loop {
        let range_min = current
            .getattr(memory_range_min_attr)?
            .downcast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let range_max = current
            .getattr(memory_range_max_attr)?
            .downcast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut split_time = 0.0f64;
        let capacity_reached = max_nodes >= 0 && (n_nodes + nodes_added) >= max_nodes;
        // Compute only the sum here; the extensions dict is built lazily if
        // we actually split (most iterations descend without splitting).
        let sum = if capacity_reached {
            0.0
        } else {
            range_extension_sum(&range_min, &range_max, &x)?
        };
        if !capacity_reached && sum > 0.0 {
            let r: f64 = rng_random.call0()?.extract()?;
            let t = -((1.0 - r).ln()) / sum;
            let node_time: f64 = current.getattr(time_attr)?.extract()?;
            let candidate = node_time + t;
            let is_leaf: bool = current.getattr(is_leaf_attr)?.extract()?;
            if is_leaf {
                split_time = candidate;
            } else {
                let children = current.getattr(children_attr)?;
                let left_child = children.get_item(0)?;
                let child_time: f64 = left_child.getattr(time_attr)?.extract()?;
                if candidate < child_time {
                    split_time = candidate;
                }
            }
        }

        if split_time > 0.0 {
            let (_, extensions) = range_extension_inner(py, &range_min, &range_max, &x)?;
            let feature = weighted_choice(py, &extensions, &rng_choices)?;

            let xf = as_f64(
                &x.get_item(&feature)?
                    .ok_or_else(|| PyValueError::new_err("feature missing in x"))?,
            )?;
            let range_min_f = as_f64(
                &range_min
                    .get_item(&feature)?
                    .ok_or_else(|| PyValueError::new_err("range_min feature missing"))?,
            )?;
            let range_max_f = as_f64(
                &range_max
                    .get_item(&feature)?
                    .ok_or_else(|| PyValueError::new_err("range_max feature missing"))?,
            )?;

            let is_right_extension = xf > range_max_f;
            let threshold: f64 = if is_right_extension {
                rng_uniform.call1((range_max_f, xf))?.extract()?
            } else {
                rng_uniform.call1((xf, range_min_f))?.extract()?
            };

            let was_leaf: bool = current.getattr(is_leaf_attr)?.extract()?;
            let new_node = split_fn.call1((
                current.clone(),
                split_time,
                threshold,
                feature,
                is_right_extension,
            ))?;
            current = new_node;
            nodes_added += 2;

            let parent = current.getattr(parent_attr)?;
            if parent.is_none() {
                new_root = Some(current.clone().unbind());
            } else if was_leaf {
                let parent_children = parent.getattr(children_attr)?;
                let left = parent_children.get_item(0)?;
                let right = parent_children.get_item(1)?;
                let new_children = if branch_no == 0 {
                    (current.clone(), right)
                } else {
                    (left, current.clone())
                };
                parent.setattr(children_attr, new_children)?;
            }

            update_downwards_regressor_inner(
                py,
                &current,
                &x,
                sample_value,
                use_aggregation,
                step,
                true,
                None,
            )?;

            let children = current.getattr(children_attr)?;
            let leaf_child = if is_right_extension {
                children.get_item(1)?
            } else {
                children.get_item(0)?
            };
            update_downwards_regressor_inner(
                py,
                &leaf_child,
                &x,
                sample_value,
                use_aggregation,
                step,
                false,
                None,
            )?;
            return Ok((leaf_child.unbind(), new_root, nodes_added));
        } else {
            update_downwards_regressor_inner(
                py,
                &current,
                &x,
                sample_value,
                use_aggregation,
                step,
                true,
                Some((&range_min, &range_max)),
            )?;
            let is_leaf: bool = current.getattr(is_leaf_attr)?.extract()?;
            if is_leaf {
                return Ok((current.unbind(), new_root, nodes_added));
            }
            let feature = current.getattr(feature_attr)?;
            unsafe {
                let res = ffi::PyDict_Contains(x.as_ptr(), feature.as_ptr());
                if res < 0 {
                    return Err(PyErr::fetch(py));
                }
                if res == 1 {
                    let xf = as_f64(
                        &x.get_item(&feature)?
                            .ok_or_else(|| PyValueError::new_err("feature missing in x"))?,
                    )?;
                    let threshold: f64 = current.getattr(threshold_attr)?.extract()?;
                    let children = current.getattr(children_attr)?;
                    if xf <= threshold {
                        branch_no = 0;
                        current = children.get_item(0)?;
                    } else {
                        branch_no = 1;
                        current = children.get_item(1)?;
                    }
                } else {
                    let result = current.call_method0(most_common_path_attr)?;
                    branch_no = result.get_item(0)?.extract()?;
                    current = result.get_item(1)?;
                }
            }
        }
    }
}

/// Walk from `leaf` to root combining Dirichlet-smoothed predictions weighted
/// by node weights. Returns a normalised list of length `n_classes`.
#[pyfunction]
pub fn predict_proba_upward<'py>(
    py: Python<'py>,
    leaf: Bound<'py, PyAny>,
    n_classes: usize,
    dirichlet: f64,
) -> PyResult<Bound<'py, PyList>> {
    let counts_attr = intern!(py, "counts");
    let n_samples_attr = intern!(py, "n_samples");
    let parent_attr = intern!(py, "parent");
    let weight_attr = intern!(py, "weight");
    let log_weight_tree_attr = intern!(py, "log_weight_tree");

    let mut current = leaf;
    let counts = current
        .getattr(counts_attr)?
        .downcast_into::<PyList>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n_samples: i64 = current.getattr(n_samples_attr)?.extract()?;
    let denom = n_samples as f64 + dirichlet * n_classes as f64;
    let n_counts = counts.len();

    let mut scores: Vec<f64> = (0..n_classes)
        .map(|i| {
            let c = if i < n_counts {
                counts
                    .get_item(i)
                    .and_then(|v| v.extract::<f64>())
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            (c + dirichlet) / denom
        })
        .collect();

    let mut next = current.getattr(parent_attr)?;
    while !next.is_none() {
        current = next;
        let weight: f64 = current.getattr(weight_attr)?.extract()?;
        let log_weight_tree: f64 = current.getattr(log_weight_tree_attr)?.extract()?;
        let w = (weight - log_weight_tree).exp();
        let half_w = 0.5 * w;
        let complement = 1.0 - half_w;

        let counts = current
            .getattr(counts_attr)?
            .downcast_into::<PyList>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let n_samples: i64 = current.getattr(n_samples_attr)?.extract()?;
        let denom = n_samples as f64 + dirichlet * n_classes as f64;
        let n_counts = counts.len();

        for i in 0..n_classes {
            let c = if i < n_counts {
                counts.get_item(i)?.extract::<f64>()?
            } else {
                0.0
            };
            let pred_new = (c + dirichlet) / denom;
            scores[i] = half_w * pred_new + complement * scores[i];
        }
        next = current.getattr(parent_attr)?;
    }

    let total: f64 = scores.iter().sum();
    if total > 0.0 && !total.is_nan() {
        let inv = 1.0 / total;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
    PyList::new(py, scores)
}

/// Walk from `leaf` up to the root recomputing `log_weight_tree`. Mirrors the
/// Python `_go_upwards` loop plus `MondrianNode.update_weight_tree`. Skipped
/// when `iteration < 1` to match the Python guard.
#[pyfunction]
pub fn go_upwards<'py>(py: Python<'py>, leaf: Bound<'py, PyAny>, iteration: i64) -> PyResult<()> {
    if iteration < 1 {
        return Ok(());
    }
    let is_leaf_attr = intern!(py, "is_leaf");
    let weight_attr = intern!(py, "weight");
    let log_weight_tree_attr = intern!(py, "log_weight_tree");
    let parent_attr = intern!(py, "parent");
    let children_attr = intern!(py, "children");

    let mut current = leaf;
    loop {
        let weight: f64 = current.getattr(weight_attr)?.extract()?;
        let new_log_weight_tree = if current.getattr(is_leaf_attr)?.extract::<bool>()? {
            weight
        } else {
            let children = current.getattr(children_attr)?;
            let left_lwt: f64 = children
                .get_item(0)?
                .getattr(log_weight_tree_attr)?
                .extract()?;
            let right_lwt: f64 = children
                .get_item(1)?
                .getattr(log_weight_tree_attr)?
                .extract()?;
            log_sum_2_exp_inner(weight, left_lwt + right_lwt)
        };
        current.setattr(log_weight_tree_attr, new_log_weight_tree)?;

        let parent = current.getattr(parent_attr)?;
        if parent.is_none() {
            return Ok(());
        }
        current = parent;
    }
}

/// Descend from `root` to a leaf using the standard Mondrian split rules. If
/// the split feature is missing from `x`, follow `most_common_path`. Returns
/// the leaf node.
fn find_leaf<'py>(
    py: Python<'py>,
    root: Bound<'py, PyAny>,
    x: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let is_leaf_attr = intern!(py, "is_leaf");
    let feature_attr = intern!(py, "feature");
    let threshold_attr = intern!(py, "threshold");
    let children_attr = intern!(py, "children");
    let most_common_path_attr = intern!(py, "most_common_path");

    let mut current = root;
    loop {
        if current.getattr(is_leaf_attr)?.extract::<bool>()? {
            return Ok(current);
        }
        let feature = current.getattr(feature_attr)?;
        unsafe {
            let res = ffi::PyDict_Contains(x.as_ptr(), feature.as_ptr());
            if res < 0 {
                return Err(PyErr::fetch(py));
            }
            if res == 1 {
                let xf = as_f64(
                    &x.get_item(&feature)?
                        .ok_or_else(|| PyValueError::new_err("feature missing in x"))?,
                )?;
                let threshold: f64 = current.getattr(threshold_attr)?.extract()?;
                let children = current.getattr(children_attr)?;
                current = if xf <= threshold {
                    children.get_item(0)?
                } else {
                    children.get_item(1)?
                };
            } else {
                let result = current.call_method0(most_common_path_attr)?;
                current = result.get_item(1)?;
            }
        }
    }
}

/// Full classifier prediction in Rust: walk root → leaf, then apply the
/// upward-aggregated Dirichlet score (or just the leaf score when
/// `use_aggregation=False`). Returns the normalised score list.
#[pyfunction]
pub fn predict_proba_classifier<'py>(
    py: Python<'py>,
    root: Bound<'py, PyAny>,
    x: Bound<'py, PyDict>,
    n_classes: usize,
    dirichlet: f64,
    use_aggregation: bool,
) -> PyResult<Bound<'py, PyList>> {
    let leaf = find_leaf(py, root, &x)?;
    if !use_aggregation {
        let counts_attr = intern!(py, "counts");
        let n_samples_attr = intern!(py, "n_samples");
        let counts = leaf
            .getattr(counts_attr)?
            .downcast_into::<PyList>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let n_samples: i64 = leaf.getattr(n_samples_attr)?.extract()?;
        return predict_scores(py, &counts, counts.len(), n_classes, dirichlet, n_samples);
    }
    predict_proba_upward(py, leaf, n_classes, dirichlet)
}

/// Full regressor prediction in Rust: walk root → leaf, take the leaf's mean,
/// then apply the upward-weighted aggregation when `use_aggregation=True`.
#[pyfunction]
pub fn predict_one_regressor<'py>(
    py: Python<'py>,
    root: Bound<'py, PyAny>,
    x: Bound<'py, PyDict>,
    use_aggregation: bool,
) -> PyResult<f64> {
    let mean_attr = intern!(py, "_mean");
    let parent_attr = intern!(py, "parent");
    let weight_attr = intern!(py, "weight");
    let log_weight_tree_attr = intern!(py, "log_weight_tree");
    let get_attr = intern!(py, "get");

    let mut current = find_leaf(py, root, &x)?;
    let mut prediction: f64 = current
        .getattr(mean_attr)?
        .call_method0(get_attr)?
        .extract()?;
    if !use_aggregation {
        return Ok(prediction);
    }

    let mut next = current.getattr(parent_attr)?;
    while !next.is_none() {
        current = next;
        let weight: f64 = current.getattr(weight_attr)?.extract()?;
        let log_weight_tree: f64 = current.getattr(log_weight_tree_attr)?.extract()?;
        let w = (weight - log_weight_tree).exp();
        let half_w = 0.5 * w;
        let pred_new: f64 = current
            .getattr(mean_attr)?
            .call_method0(get_attr)?
            .extract()?;
        prediction = half_w * pred_new + (1.0 - half_w) * prediction;
        next = current.getattr(parent_attr)?;
    }
    Ok(prediction)
}
