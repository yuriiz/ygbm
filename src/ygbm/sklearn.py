#!/usr/bin/env python3

import numpy as np
import torch
import torch.jit
from pandas import Series
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from . import fit, predict, CCELoss, BCELoss, MSELoss


class YGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        n_jobs=None,
        verbose=0,
        loss=None,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.loss = loss

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        *,
        X_val=None,
        y_val=None,
        sample_weight_val=None,
    ):
        self._categories = list(set(y))
        if self._categories == [0, 1]:
            if self.loss is None:
                self.loss = BCELoss()
            if isinstance(y, Series):
                target = torch.tensor(y.values)[:, None].float()
            else:
                target = torch.tensor(y[:, None]).float()
        else:
            if self.loss is None:
                self.loss = CCELoss()
            target = torch.zeros((len(y), len(self._categories)))
            for i, c in enumerate(self._categories):
                target[y == c, i] = 1
        self._baseline, self._trees = fit(
            self.loss,
            X,
            target,
            sample_weight=sample_weight,
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=sample_weight_val,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            verbose=self.verbose,
        )
        return self

    def predict(self, X, parallel=False):
        for p in predict(X, self._baseline, self._trees, n_jobs=self.n_jobs):
            pass
        return [self._categories[i] for i in np.argmax(p, 1)]

    def predict_proba(self, X, parallel=False):
        for p in predict(X, self._baseline, self._trees, n_jobs=self.n_jobs):
            pass
        return self.loss.predictions(p)


class YGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        loss=None,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        verbose=False,
        n_jobs=None,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.n_jobs = n_jobs
        if loss is None:
            self.loss = MSELoss()
        else:
            self.loss = loss

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        *,
        X_val=None,
        y_val=None,
        sample_weight_val=None,
    ):
        self._baseline, self._trees = fit(
            self.loss,
            X,
            torch.tensor(y).float()[:, None],
            sample_weight=sample_weight,
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=sample_weight_val,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        for p in predict(X, self._baseline, self._trees, n_jobs=self.n_jobs):
            last = p
        return last

    def to_python(self) -> str:
        return "+".join(
            [
                str(self._baseline.item()),
                *[tree.to_python() for (tree,) in self._trees],
            ]
        )
