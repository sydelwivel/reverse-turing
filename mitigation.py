# mitigation.py
import numpy as np
from sklearn.isotonic import IsotonicRegression

def reweight_by_group(resumes, labels, protected_fn):
    """
    Simple reweighting: sample weights inversely proportional to group's positive rate.
    returns: numpy array of normalized sample weights
    """
    groups = {}
    for r, lab in zip(resumes, labels):
        g = protected_fn(r)
        groups.setdefault(g, []).append(lab)
    weights = []
    for r, lab in zip(resumes, labels):
        g = protected_fn(r)
        grp = groups.get(g, [0])
        sel_rate = np.mean(grp) if len(grp) > 0 else 0.5
        w = 1.0 / max(0.01, sel_rate)
        weights.append(w)
    w = np.array(weights)
    return w / np.mean(w)

def isotonic_postprocess(scores, labels):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, labels)
    def transform(new_scores):
        return ir.transform(np.array(new_scores).reshape(-1))
    return transform
