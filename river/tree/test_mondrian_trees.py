import pytest

from river.datasets import synth
from river import forest

from river.tree.mondrian.mondrian_tree_nodes import MondrianBranch, MondrianLeaf


def contained(c, o_min, o_max) -> bool:
    return o_min <= c <= o_max


def check_siblings_overlap(nodes):
    for n in nodes:
        if not isinstance(n, MondrianBranch):
            continue

        left, right = n.children

        for feat in left.memory_range_min:
            assert (
                contained(
                    left.memory_range_min[feat],
                    right.memory_range_min[feat],
                    right.memory_range_max[feat],
                )
                or contained(
                    left.memory_range_max[feat],
                    right.memory_range_min[feat],
                    right.memory_range_max[feat],
                )
                or contained(
                    right.memory_range_min[feat],
                    left.memory_range_min[feat],
                    left.memory_range_max[feat],
                )
                or contained(
                    right.memory_range_max[feat],
                    left.memory_range_min[feat],
                    left.memory_range_max[feat],
                )
            )


def check_child_inside_parent(nodes):
    for n in nodes:
        if n.parent is None or n.n_samples == 0:
            continue

        p = n.parent

        for feat in n.memory_range_min:
            assert contained(
                n.memory_range_min[feat],
                p.memory_range_min[feat],
                p.memory_range_max[feat],
            ) and contained(
                n.memory_range_max[feat],
                p.memory_range_min[feat],
                p.memory_range_max[feat],
            )


def check_parent_has_children_sum():
    pass


@pytest.mark.parametrize(
    "dataset, model",
    [
        pytest.param(
            synth.Hyperplane(seed=42).take(300),
            forest.AMFClassifier(n_estimators=2, seed=8),
            id=f"AMFClassifier_Hiperplane"
        ),
        pytest.param(
            synth.Friedman(seed=42).take(300),
            forest.AMFRegressor(n_estimators=2, seed=8),
            id=f"AMFRegressor_Friedman"
        ),
    ],
)
def test_amf_trees(dataset, model):
    for x, y in dataset:
        model.learn_one(x, y)

    for tree in model:
        nodes = list(tree._root.iter_dfs())

        # check_siblings_overlap(nodes)
        check_child_inside_parent(nodes)
