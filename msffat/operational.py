"""Validation-only Max-F1 threshold selection with an exact tie rule."""
import numpy as np


def max_f1_threshold(binary_labels,scores):
    labels,scores = np.asarray(binary_labels),np.asarray(scores,dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels) or not np.isin(labels,[0,1]).all():
        raise ValueError("Expected aligned nonempty binary validation labels and scores")
    if not np.isfinite(scores).all() or scores.min()<0 or scores.max()>1:
        raise ValueError("Expected finite detection scores in [0,1]")
    order = np.argsort(-scores,kind="stable")
    ranked,truth = scores[order],labels[order].astype(int)
    end = np.r_[np.flatnonzero(ranked[:-1]!=ranked[1:]),len(ranked)-1]
    tp = np.cumsum(truth)[end]
    predicted = end+1
    positives = truth.sum()
    f1 = np.divide(2*tp,predicted+positives,out=np.zeros_like(tp,dtype=float),where=(predicted+positives)!=0)
    # Add the no-positive prediction. Descending thresholds make argmax choose
    # the largest threshold among exactly equal computed F1 values.
    values = np.r_[0.,f1]
    thresholds = np.r_[np.nextafter(1.,2.),ranked[end]]
    best = int(np.argmax(values))
    true_positive = 0 if best==0 else int(tp[best-1])
    count = 0 if best==0 else int(predicted[best-1])
    return {"threshold":float(thresholds[best]),"f1":float(values[best]),
            "precision":true_positive/count if count else 0.,
            "recall":true_positive/positives if positives else 0.}
