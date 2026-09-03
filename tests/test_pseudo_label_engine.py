
from __future__ import annotations
import numpy as np
from src.ssl_engine import PseudoLabelEngine, SelectionConfig

def test_class_balanced_caps_and_zero_accept():
    proba=np.full((100,3), 1/3.0)
    eng=PseudoLabelEngine(SelectionConfig(confidence_threshold=0.99, random_state=0))
    sel=eng.select(proba, n_labeled=30, budget=50, labeled_classes=np.array([0,1,2]))
    assert len(sel.indices)==0

def test_deterministic_ties():
    proba=np.array([[0.9,0.05,0.05],[0.9,0.05,0.05],[0.2,0.7,0.1],[0.1,0.2,0.7]],float)
    eng=PseudoLabelEngine(SelectionConfig(confidence_threshold=0.5, per_round_cap=10, use_class_balanced=True, random_state=0, multiplier_of_n_labeled=None))
    a=eng.select(proba, n_labeled=2, budget=50, labeled_classes=np.array([0,1,2]))
    b=eng.select(proba, n_labeled=2, budget=50, labeled_classes=np.array([0,1,2]))
    assert np.array_equal(a.indices,b.indices)

def test_agreement_required():
    p1=np.array([[0.95,0.05],[0.1,0.9],[0.8,0.2]],float)
    p2=np.array([[0.9,0.1],[0.85,0.15],[0.75,0.25]],float)
    eng=PseudoLabelEngine(SelectionConfig(confidence_threshold=0.7, agreement_required=True, random_state=0, multiplier_of_n_labeled=None, stricter_at_budget_50=False))
    sel=eng.select(p1, teacher2_proba=p2, n_labeled=2, budget=100, labeled_classes=np.array([0,1]))
    assert 1 not in set(sel.indices.tolist())
