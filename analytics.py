# analytics.py
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def binomial_test(k, n, p0=0.5):
    """Performs a binomial test for accuracy > chance level."""
    result = stats.binomtest(k, n, p=p0)   # ✅ new API
    return result.pvalue

def kl_divergence(p, q):
    """Kullback-Leibler divergence D_KL(P || Q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return stats.entropy(p, q)

def js_divergence(p, q):
    """Jensen–Shannon divergence (symmetric & bounded)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return jensenshannon(p, q) ** 2   # return divergence, not distance

def earth_movers_distance(p, q):
    """Earth Mover’s Distance (a.k.a Wasserstein distance)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Use Wasserstein distance for 1D histograms
    return stats.wasserstein_distance(p, q)

def train_meta_classifier(X, y):
    """
    Train a simple meta-classifier to distinguish AI vs persona outputs.
    Returns the fitted model and AUC score.
    """
    model = LogisticRegression()
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    return model, auc
