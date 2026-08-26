from __future__ import annotations

import random

import pytest

from river import metrics


def f1atk_with_subclasses(y_true, y_pred, k, relevance_threshold=None):
    prec = metrics.PrecisionAtK(k, relevance_threshold=relevance_threshold)._eval(y_true, y_pred)
    rec = metrics.RecallAtK(k, relevance_threshold=relevance_threshold)._eval(y_true, y_pred)
    try:
        return (2 * prec * rec) / (prec + rec)
    except ZeroDivisionError:
        return 0


def test_ranking_utils():
    with pytest.raises(ValueError):
        metrics.PrecisionAtK()._relevance(dict())
    assert isinstance(metrics.RecallAtK()._relevance(set()), set)
    assert isinstance(metrics.PrecisionAtK(relevance_threshold=1)._relevance(dict()), list)
    assert metrics.PrecisionAtK(k=None)._resolve_k([1,2,3]) == 3
    assert metrics.RecallAtK(k=2)._resolve_k([1,2,3]) == 2
    assert metrics.RecallAtK(k=4)._resolve_k([1,2,3]) == 4
    assert metrics.PrecisionAtK(k=2)._resolve_k([]) == 2

    
def test_precisionatk():
    y_pred = ['a','b','c']   
    y_true = ['c','d']
    assert metrics.PrecisionAtK(3)._eval(y_true, y_pred) == 1/3
    assert metrics.PrecisionAtK(3)._eval(y_true[-1:], y_pred) == 0
    assert metrics.PrecisionAtK(1)._eval(y_true, y_pred) == 0
    assert metrics.PrecisionAtK(2)._eval(y_true[-1:], y_pred) == 0
    assert metrics.PrecisionAtK(4)._eval(y_true, y_pred) == .25
    assert metrics.PrecisionAtK(None)._eval(y_true, []) == 0


def test_recallatk():
    y_pred = ['a','b','c']   
    y_true = ['c','d']
    assert metrics.RecallAtK(3)._eval(y_true, y_pred) == 1/2
    assert metrics.RecallAtK(3)._eval(y_true[-1:], y_pred) == 0
    assert metrics.RecallAtK(1)._eval(y_true, y_pred) == 0
    assert metrics.RecallAtK(2)._eval(y_true[-1:], y_pred) == 0
    assert metrics.RecallAtK(4)._eval(y_true, y_pred) == .5
    assert metrics.RecallAtK(None)._eval(y_true, []) == 0


def test_f1atk_binary():
    items_str = 'abcdefghijklmno'
    len_items_str = len(items_str)
    for _ in range(2000):
        k = random.randint(1,len_items_str)
        pred_len = random.randint(1,len_items_str)
        true_len = random.randint(0,len_items_str)
        y_true = random.sample(items_str, true_len)
        y_pred = random.sample(items_str, pred_len)
        assert metrics.F1AtK(k=k)._eval(y_true, y_pred) == \
            pytest.approx(f1atk_with_subclasses(y_true, y_pred,k))
    assert metrics.F1AtK(None)._eval([], []) == 0

            
def test_f1atk_relevance():
    items_str = 'abcdefghijklmno'
    len_items_str = len(items_str)
    for _ in range(2000):
        k = random.randint(1,len_items_str)
        pred_len = random.randint(1,len_items_str)
        true_len = random.randint(0,len_items_str)
        relevance_threshold = random.randint(1,5)
        y_true = random.sample(items_str, true_len)
        y_true_dict = {}
        for y_true_item in y_true:
            y_true_dict[y_true_item] = random.randint(1,5)
        y_pred = random.sample(items_str, pred_len)
        assert metrics.F1AtK(k=k, relevance_threshold=relevance_threshold)._eval(y_true_dict, y_pred) == \
            pytest.approx(f1atk_with_subclasses(y_true_dict, y_pred,k,relevance_threshold=relevance_threshold))
        assert metrics.F1AtK(None, relevance_threshold=1)._eval({}, []) == 0