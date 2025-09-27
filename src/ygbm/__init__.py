from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from time import time

import numpy as np
import torch
import torch.jit
from joblib import Parallel, delayed
from pandas import CategoricalDtype, DataFrame, Series
from scipy.special import expit


class Node(ABC):
    @abstractmethod
    def replace(self, node: "Node", to: "Node") -> "Node":
        pass

    @abstractmethod
    def predict(self, x) -> float:
        pass

    @abstractmethod
    def to_python(self) -> str:
        pass


@dataclass(frozen=True, slots=True, order=True)
class ConstLeafNode(Node):
    value: float

    def replace(self, node: Node, to: Node) -> Node:
        if self is node:
            return to
        return self

    def predict(self, x: list[float]) -> float:
        return self.value

    def to_python(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True, order=True)
class LinearLeafNode(Node):
    value: float
    mean: np.ndarray
    std: np.ndarray
    features: list[int]
    weights: np.ndarray

    def replace(self, node: Node, to: Node) -> Node:
        if self is node:
            return to
        return self

    def predict(self, x: Sequence[float]) -> float:
        return self.value + sum(
            [
                (
                    0
                    if np.isnan(x[feature])
                    else self.weights[i] * (x[feature] - self.mean[i]) / self.std[i]
                )
                for i, feature in enumerate(self.features)
            ]
        )

    def to_python(self) -> str:
        return "({} + sum({} * (x[{}] - {}) / {}))".format(
            self.value,
            self.weights.tolist(),
            self.features,
            self.mean.tolist(),
            self.std.tolist(),
        )


@dataclass(frozen=True, slots=True, order=True)
class NumSplitNode(Node):
    feature: int
    value: float
    missing_left: bool
    missing_right: bool
    left: Node
    right: Node

    def replace(self, node: Node, to: Node) -> Node:
        if self is node:
            return to
        return NumSplitNode(
            self.feature,
            self.value,
            self.missing_left,
            self.missing_right,
            self.left.replace(node, to),
            self.right.replace(node, to),
        )

    def predict(self, x) -> float:
        return (
            (self.left.predict(x) if self.missing_left else self.right.predict(x))
            if np.isnan(x[self.feature])
            else (
                self.left.predict(x)
                if x[self.feature] <= self.value
                else self.right.predict(x)
            )
        )

    def to_python(self) -> str:
        return "({} if x[{}] <= {} else {})".format(
            self.left.to_python(), self.feature, self.value, self.right.to_python()
        )


@dataclass(frozen=True, slots=True, order=True)
class CatSplitNode(Node):
    feature: int
    left_categories: list
    left: Node
    right: Node

    def replace(self, node: Node, to: Node) -> Node:
        if self is node:
            return to
        return CatSplitNode(
            self.feature,
            self.left_categories,
            self.left.replace(node, to),
            self.right.replace(node, to),
        )

    def predict(self, x) -> float:
        return (
            self.left.predict(x)
            if x[self.feature] in self.left_categories
            else self.right.predict(x)
        )

    def to_python(self) -> str:
        return "({} if x[{}] in {} else {})".format(
            self.left.to_python(),
            self.feature,
            self.left_categories,
            self.right.to_python(),
        )


class Loss(ABC):
    @abstractmethod
    def baseline(self, y):
        pass

    @abstractmethod
    def predictions(self, y):
        pass

    @abstractmethod
    def gradients(self, y, target):
        pass

    @abstractmethod
    def hessians(self, y, target):
        pass

    def derivatives(self, y, target):
        p = self.predictions(y)
        return self.gradients(p, target), self.hessians(p, target)


class MSELoss(Loss):
    @staticmethod
    def baseline(y):
        return y.mean(0)

    @staticmethod
    def predictions(y):
        return y

    @staticmethod
    def gradients(y, target):
        return y - target

    @staticmethod
    def hessians(y, target):
        return torch.ones_like(target)


class BCELoss(Loss):
    @staticmethod
    def baseline(y):
        m = y.mean(0)
        return torch.logit(m)

    @staticmethod
    def predictions(y):
        if isinstance(y, torch.Tensor):
            return y.sigmoid()
        else:
            return expit(y)

    @staticmethod
    def gradients(p, target):
        return p - target

    @staticmethod
    def hessians(p, target):
        return p * (1 - p)


class CCELoss(Loss):
    @staticmethod
    def baseline(y):
        m = y.mean(0)
        return m.log() - m.log().mean()

    @staticmethod
    def predictions(y):
        if isinstance(y, torch.Tensor):
            mx, _ = y.max(1)
            e = torch.exp((y.T - mx).T)
        else:
            mx = y.max(1)
            e = np.exp((y.T - mx).T)
        return e / e.sum(1)[:, None]

    @staticmethod
    def gradients(p, target):
        return p - target

    @staticmethod
    def hessians(p, target):
        return p * (1 - p)


@dataclass(frozen=True, slots=True, order=True)
class PinballLoss(Loss):
    quantile: float

    def baseline(self, y):
        return torch.quantile(y, self.quantile)[None,]

    @staticmethod
    def predictions(y):
        return y

    def gradients(self, raw_prediction, y_true):
        return torch.where(
            y_true >= raw_prediction,
            -self.quantile,
            1.0 - self.quantile,
        )

    @staticmethod
    def hessians(p, target):
        return torch.ones_like(target)


N_BINS = 0x100
MISSING_BIN = 0xFF


@torch.jit.script
def _bin_derivatives(X_binned, n_bins: int, gradients, hessians):
    n_features: int = len(X_binned)
    stats = torch.zeros(
        (n_features, 3, n_bins), device=X_binned.device, dtype=gradients.dtype
    ).detach()
    xi = X_binned.int()
    stats[:, 0].scatter_add_(1, xi, gradients.repeat(n_features, 1))
    stats[:, 1].scatter_add_(1, xi, hessians.repeat(n_features, 1))
    stats[:, 2].scatter_add_(
        1,
        xi,
        torch.ones(
            n_features,
            len(gradients),
            device=X_binned.device,
            dtype=gradients.dtype,
        ),
    )
    return stats


@torch.jit.script
def _find_best_split(
    leaf_value: float,
    idx: torch.Tensor,
    gh: torch.Tensor,
    histograms: torch.Tensor,
    sums: torch.Tensor,
    num_features: list[int],
    missing_features: list[int],
    categorical_features: list[int],
    min_samples_leaf: int,
    min_hessian_to_split: float = 1e-3,
    MISSING_BIN: int = MISSING_BIN,
    N_BINS: int = N_BINS,
) -> tuple:
    split = (
        0,
        idx,
        -1,
        0,
        False,
        (
            False,
            False,
        ),
        0,
        0,
        (sums, sums),
    )
    sum_gradients = sums[0]
    sum_hessians = sums[1]
    n_samples = sums[2]
    if sum_hessians < min_hessian_to_split:
        return split
    g = gh[0]
    h = gh[1]
    if g.min() == g.max() and h.min() == h.max():
        # can't split if all values are the same
        return split
    best_category_split: tuple[
        float,
        int,
        list[int],
        float,
        float,
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ] = (0.0, 0, [0], 0.0, 0.0, (torch.empty(0), torch.empty(0)))
    if categorical_features:
        MIN_CAT_SUPPORT = 10
        support_factor = n_samples / sum_hessians
        for f in categorical_features:
            s = histograms[f]
            gradients = s[0]
            hessians = s[1]
            counts = s[2]
            low_support = hessians * support_factor < MIN_CAT_SUPPORT
            sums_left = torch.tensor([0.0, 0.0, 0.0]).to(s.device).detach()
            sums_right = sums.clone()
            score = gradients / (hessians + MIN_CAT_SUPPORT)
            left_categories: list[int] = []
            for category in score.argsort():
                if low_support[category]:
                    continue
                if not counts[category]:
                    assert 0
                    continue
                sums_left += s[:, category]
                sums_right -= s[:, category]
                left_categories.append(category.item())
                if len(left_categories) == (~low_support).sum().item() // 2 + 1:
                    # if we've reached middle, move all low support categories
                    # to the left so they are always in the biggest leaf
                    for c in (low_support & (counts > 0)).nonzero():
                        cint: int = int(c.item())
                        left_categories.append(cint)
                        sums_left += s[:, cint]
                        sums_right -= s[:, cint]
                sum_gradient_left = sums_left[0]
                sum_hessian_left = sums_left[1]
                n_samples_left = sums_left[2]
                if (n_samples_left < min_samples_leaf) or (
                    sum_hessian_left < min_hessian_to_split
                ):
                    continue
                sum_gradient_right = sums_right[0]
                sum_hessian_right = sums_right[1]
                n_samples_right = sums_right[2]
                if (n_samples_right < min_samples_leaf) or (
                    sum_hessian_right < min_hessian_to_split
                ):
                    break
                value_left = -sum_gradient_left / sum_hessian_left
                value_right = -sum_gradient_right / sum_hessian_right
                gain = sum_gradients * leaf_value
                gain -= sum_gradient_left * value_left
                gain -= sum_gradient_right * value_right
                if best_category_split[0] < gain:
                    best_category_split = (
                        gain.item(),
                        f,
                        left_categories[:],
                        value_left.item(),
                        value_right.item(),
                        (sums_left.clone(), sums_right.clone()),
                    )
    if missing_features:
        # add histograms in case NaN go to left leaf
        stats2 = histograms[missing_features]
        stats2[:, :, 0] += stats2[:, :, MISSING_BIN]
        stats2[:, :, MISSING_BIN] = 0
        stats = torch.cat([histograms[num_features], stats2])
    else:
        stats = histograms[num_features]
    stats = stats[:, :, :N_BINS]
    sums_left = stats.cumsum(-1)
    sums_right = sums_left[:, :, -1][:, :, None] - sums_left
    values_left = -sums_left[:, 0] / sums_left[:, 1]
    values_right = -sums_right[:, 0] / sums_right[:, 1]
    gains = sum_gradients * leaf_value
    gains = gains - sums_left[:, 0] * values_left
    gains = gains - sums_right[:, 0] * values_right
    gains[sums_left[:, 2] < min_samples_leaf] = 0
    gains[sums_right[:, 2] < min_samples_leaf] = 0
    gains[sums_left[:, 1] < min_hessian_to_split] = 0
    gains[sums_right[:, 1] < min_hessian_to_split] = 0
    position = int(gains.argmax().item())
    feature, position = position // N_BINS, position % N_BINS
    max_gain = gains[feature][position]
    if max_gain < best_category_split[0]:
        # split on categorical feature
        (
            gain,
            feature,
            left_categories,
            value_left,
            value_right,
            sums_left_right,
        ) = best_category_split
        return (
            -gain,
            idx,
            feature,
            left_categories,
            True,
            (),
            value_left,
            value_right,
            sums_left_right,
        )
    if not max_gain:
        return split
    value_left = values_left[feature][position].item()
    value_right = values_right[feature][position].item()
    sums = sums_left[feature, :, position], sums_right[feature, :, position]
    if feature < len(num_features):
        missing_left = False
        missing_right = stats[feature, 2, MISSING_BIN].item() > 0
        feature = num_features[feature]
    else:
        missing_left = True
        missing_right = False
        feature = missing_features[feature - len(num_features)]
    return (
        -max_gain.item(),
        idx,
        feature,
        position,
        False,
        (
            missing_left,
            missing_right,
        ),
        value_left,
        value_right,
        sums,
    )


@torch.jit.script
def _find_best_splits_jit(
    leaf_infos: list[
        tuple[
            float,
            list[int],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    num_features: list[int],
    missing_features: list[int],
    categorical_features: list[int],
    min_samples_leaf: int,
) -> list[tuple]:
    return [
        torch.jit.wait(fut)
        for fut in [
            torch.jit.fork(
                _find_best_split,
                leaf_value,
                idx,
                gh,
                histograms,
                sums,
                num_features,
                missing_features,
                categorical_features,
                min_samples_leaf,
            )
            for (
                leaf_value,
                parent_features,
                x,
                idx,
                gh,
                histograms,
                sums,
            ) in leaf_infos
        ]
    ]


def _find_best_splits(
    leaf_infos: list[
        tuple[
            int,
            ConstLeafNode,
            list[int],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    num_features: list[int],
    missing_features: list[int],
    categorical_features: list[int],
    bin_midpoints: list[torch.Tensor],
    min_samples_leaf: int,
):
    return [
        (
            gain,
            idx[0].item(),
            position,
            (
                CatSplitNode(
                    feature,
                    bin_midpoints[feature][position],
                    *args,
                    left=ConstLeafNode(left),
                    right=ConstLeafNode(right),
                )
                if is_categorical
                else NumSplitNode(
                    feature,
                    bin_midpoints[feature][position].item(),
                    *args,
                    left=ConstLeafNode(left),
                    right=ConstLeafNode(right),
                )
            ),
            sums,
        )
        for (
            gain,
            idx,
            feature,
            position,
            is_categorical,
            args,
            left,
            right,
            sums,
        ) in _find_best_splits_jit(
            [(leaf.value, *a) for (_depth, leaf, *a) in leaf_infos],
            num_features,
            missing_features,
            categorical_features,
            min_samples_leaf,
        )
    ]


def _split(root, split, categorical_features, n_bins, verbose):
    # split
    (
        gain,
        _id,
        position,
        split,
        s,
        depth,
        leaf,
        parent_features,
        x,
        i,
        gh,
        stats,
        sums,
    ) = split
    if not gain:
        if verbose > 1:
            print("No possible leaves to split.")
        return root, []
    (sl, sr) = s
    if verbose > 1:
        print(
            "Splitting at",
            split.feature,
            "at bin",
            position,
            "value: ",
            leaf.value,
            "->",
            split.left.value,
            "+",
            split.right.value,
            "gain:",
            gain,
            split.feature not in categorical_features and split.missing_left,
        )
    assert gain
    root = root.replace(leaf, split)
    if split.feature in categorical_features:
        idx = torch.isin(x[split.feature], torch.tensor(position).to(x.device))
    else:
        idx = x[split.feature] <= position
        if split.missing_left:
            idx |= x[split.feature] == MISSING_BIN
    nidx = ~idx
    xl = x[:, idx]
    xr = x[:, nidx]
    il = i[idx]
    ir = i[nidx]
    ghl = gh[:, idx]
    ghr = gh[:, nidx]
    if len(il) < len(ir):
        gl, hl = ghl
        left = _bin_derivatives(xl, n_bins, gl, hl)
        right = stats - left
    else:
        gr, hr = ghr
        right = _bin_derivatives(xr, n_bins, gr, hr)
        left = stats - right
    depth += 1
    return root, [
        (
            depth,
            split.left,
            parent_features | {split.feature},
            xl,
            il,
            ghl,
            left,
            sl,
        ),
        (
            depth,
            split.right,
            parent_features | {split.feature},
            xr,
            ir,
            ghr,
            right,
            sr,
        ),
    ]


def fit(
    loss,
    X,
    y,
    sample_weight=None,
    *,
    X_val=None,
    y_val=None,
    sample_weight_val=None,
    # optional params
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    verbose=0,
    device=torch.device("cuda") if torch.cuda.is_available() else None,
) -> tuple[float, list[list[Node]]]:
    start = time()
    n_samples, n_features = X.shape
    n_samples, n_targets = y.shape
    X_dev = []
    bins = []
    bin_midpoints = []
    num_features = []
    missing_features = []
    categorical_features = []
    for f in range(n_features):
        if isinstance(X, DataFrame):
            x = X.iloc[:, f]
        else:
            x = X[:, f]
        if isinstance(x.dtype, CategoricalDtype):
            categorical_features.append(f)
            bins.append(x.cat.codes)
            midpoints = x.cat.categories
            X_dev.append(torch.tensor(x.cat.codes.values).to(device))
        else:
            num_features.append(f)
            distinct_values = np.unique(x)
            if len(distinct_values) <= N_BINS:
                assert np.nan not in distinct_values
                midpoints = distinct_values[:-1] + distinct_values[1:]
                midpoints = 0.5 * midpoints
                d = np.digitize(x, midpoints, right=True)
            else:
                percentiles = np.linspace(0, 100, num=N_BINS)
                percentiles = percentiles[1:-1]
                midpoints = np.percentile(
                    x[~np.isnan(x)], percentiles, method="midpoint"
                )
                d = np.digitize(x, midpoints, right=True)
                assert d.max() < 0xFF
                if np.isnan(x).any():
                    missing_features.append(f)
                    d[np.isnan(x)] = 0xFF
            if isinstance(x, Series):
                X_dev.append(torch.tensor(x.values).to(device))
            else:
                X_dev.append(torch.tensor(x).to(device))
            bins.append(d)
        bin_midpoints.append(midpoints)
    X_dev = torch.stack(X_dev).float().detach()
    X_binned = torch.tensor(np.stack(bins)).to(device).detach()
    n_bins = max(N_BINS, X_binned.max().item() + 1)
    del bins
    y = y.to(device).detach()
    baseline = loss.baseline(y).detach()
    raw_predictions = baseline.repeat(len(y), 1)
    trees: list[list[Node]] = []
    for it in range(max_iter):
        gradients, hessians = loss.derivatives(raw_predictions, y)
        trees.append([])
        for t in range(n_targets):
            if verbose > 1:
                print(f"Target {t} of {n_targets}.")
            g = gradients[:, t]
            h = hessians[:, t]
            # gradients, hessians, counts
            sums = torch.tensor([g.sum(), h.sum(), n_samples]).to(device)
            stats = _bin_derivatives(X_binned, n_bins, g, h)
            root: Node = ConstLeafNode(0)
            new_leaves = [
                (
                    0,
                    root,
                    set(),
                    X_binned,
                    torch.arange(n_samples, device=device).detach(),
                    torch.stack([g, h]).detach(),
                    stats,
                    sums,
                )
            ]
            leaves = []  # leaves sorted by possible split gain
            while len(leaves) + len(new_leaves) < max_leaf_nodes:
                for n, s in zip(
                    new_leaves,
                    _find_best_splits(
                        new_leaves,
                        num_features,
                        missing_features,
                        categorical_features,
                        bin_midpoints,
                        min_samples_leaf,
                    ),
                ):
                    heappush(leaves, (*s, *n))
                root, new_leaves = _split(
                    root,
                    heappop(leaves),
                    categorical_features,
                    n_bins,
                    verbose,
                )
                if not new_leaves:
                    break
            # apply shrinkage and update predictions
            for leaf, parent_features, idx in [
                *[
                    (leaf, parent_features, idx)
                    for (
                        _gain,
                        _id,
                        _position,
                        _split,
                        _s,
                        _depth,
                        leaf,
                        parent_features,
                        _x,
                        idx,
                        _gh,
                        _stats,
                        _sums,
                    ) in leaves
                ],
                *[
                    (leaf, parent_features, idx)
                    for (
                        _depth,
                        leaf,
                        parent_features,
                        _x,
                        idx,
                        _gh,
                        _stats,
                        _sums,
                    ) in new_leaves
                ],
            ]:
                value = learning_rate * leaf.value
                raw_predictions[idx, t] += value
                # select features for gradient descend
                features = list(parent_features - set(categorical_features))
                if features:
                    x = X_dev[features][:, idx].T
                    # remove features consisting of only NaNs
                    nan = x.isnan().all(0)
                    x = x[:, ~nan]
                    for i in sorted((nan.nonzero()), reverse=True):
                        features.pop(i)
                    # normalize mean
                    mean = x.nanmean(0)
                    x = x - mean
                    x[x.isnan()] = 0
                    # remove constant features
                    std = x.std(0)
                    var = std > 0
                    const = ~var
                    mean = mean[var]
                    std = std[var]
                    x = x[:, var]
                    for i in sorted((const.nonzero()), reverse=True):
                        features.pop(i)
                if not features:
                    root = root.replace(leaf, ConstLeafNode(value))
                    continue
                g = loss.gradients(loss.predictions(raw_predictions[idx]), y[idx])
                if g.min() == g.max():
                    root = root.replace(leaf, ConstLeafNode(value))
                else:
                    # normalize std
                    x /= std
                    g = x.T.matmul(g[:, t]) / len(x)
                    w = -learning_rate * g
                    raw_predictions[idx, t] += (w * x).sum(1)
                    root = root.replace(
                        leaf,
                        LinearLeafNode(
                            value,
                            mean.cpu().numpy(),
                            std.cpu().numpy(),
                            features,
                            w.cpu().numpy(),
                        ),
                    )
                del g
            trees[-1].append(root)
        if verbose:
            print(
                f"Iteration {it + 1} of {max_iter}. "
                f"Elapsed: {time() - start:.1f}s. "
                f"ETA: {(max_iter - it - 1) * (time() - start) / (it + 1):.1f}s."
            )
        for root in trees[-1]:
            if isinstance(root, (CatSplitNode, NumSplitNode)):
                break
        else:
            print(
                "No more splits with gain can be made. Stopping at iteration {it}.",
            )
            break
    return baseline.cpu().numpy(), trees


def _predict_one(X, it):
    return np.array(
        [
            np.array(
                [
                    root.predict(x)
                    for x in (
                        map(list, X.itertuples(index=False))
                        if isinstance(X, DataFrame)
                        else X
                    )
                ]
            )
            for root in it
        ]
    ).T


def predict(X, baseline, trees, *, n_jobs=None):
    raw_predictions = np.full(
        shape=(len(X), len(baseline)),
        fill_value=baseline,
    )
    for p in Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_predict_one)(X, it) for it in trees
    ):
        raw_predictions += p
        yield raw_predictions
