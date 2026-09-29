"""Importable response adapter so candidate calibration artifacts are portable."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin


class ResponseAdapter(ClassifierMixin,BaseEstimator):
    """Frozen identity over actual class responses, never learns document features."""
    def __init__(self,classes): self.classes=classes
    def fit(self,X,y=None):
        self.classes_=np.asarray(self.classes);self.n_features_in_=X.shape[1];return self
    def decision_function(self,X): return np.asarray(X)
    def predict(self,X): return self.classes_[np.asarray(X).argmax(axis=1)]
