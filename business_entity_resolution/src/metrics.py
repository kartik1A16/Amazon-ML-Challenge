"""Macro-averaged F0.5 exactly as defined in the problem statement (singletons count)."""
import numpy as np


def f05_single(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(s1_ids, pred, truth):
    """s1_ids: iterable of Source-1 ids to score; pred/truth: dict id -> set of ids."""
    s1_ids = list(s1_ids)
    if not s1_ids:
        return float("nan")
    return float(np.mean([f05_single(pred.get(s, set()), truth.get(s, set())) for s in s1_ids]))
