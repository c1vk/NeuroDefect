from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    backend: str = "numpy"


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for shape (n, 2) or (2,)."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z - np.max(z)
        e = np.exp(z)
        return (e / e.sum()).astype(np.float32)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _numpy_weights_from_torch_state(state: dict) -> list[np.ndarray]:
    """Map a Sequential Linear state_dict onto the NumPy forward weight layout.

    ``nn.Linear`` stores weight as ``(out, in)``. The NumPy network computes
    ``X @ W`` and therefore stores the transpose, ``(in, out)``.
    """
    pairs = (
        ("net.0.weight", "net.0.bias"),
        ("net.3.weight", "net.3.bias"),
        ("net.6.weight", "net.6.bias"),
    )
    weights: list[np.ndarray] = []
    for weight_key, bias_key in pairs:
        weights.append(np.asarray(state[weight_key], dtype=np.float32).T.copy())
        weights.append(np.asarray(state[bias_key], dtype=np.float32).reshape(-1).copy())
    return weights


def _class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced weights for classes {0, 1}: n / (2 * n_c)."""
    y = np.asarray(y).astype(int).reshape(-1)
    n = len(y)
    n0 = max(int((y == 0).sum()), 1)
    n1 = max(int((y == 1).sum()), 1)
    return np.array([n / (2.0 * n0), n / (2.0 * n1)], dtype=np.float32)


def _threshold_grid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    return np.unique(np.concatenate([scores, np.array([0.5]), np.linspace(0.05, 0.95, 37)]))


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Pick decision threshold on validation scores that maximises F1."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in _threshold_grid(scores):
        pred = (scores >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def _tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    mode: str = "f1",
    accuracy_floor: float = 0.65,
    fn_cost: float = 2.0,
) -> tuple[float, dict]:
    """Tune a decision threshold on validation scores.

    Modes:
      - f1: maximise F1
      - accuracy_floor: among thresholds with accuracy >= floor, maximise F1;
        if none qualify, maximise accuracy then F1 (helps large imbalanced sets)
      - cost_sensitive: minimise FP + fn_cost * FN (then break ties by F1)
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    info = {"mode": mode, "accuracy_floor": float(accuracy_floor), "fn_cost": float(fn_cost)}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, info
    grid = _threshold_grid(scores)
    best_t, best_key = 0.5, None
    for t in grid:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        acc = (tp + tn) / max(len(y_true), 1)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if mode == "accuracy_floor":
            qualifies = acc >= accuracy_floor
            key = (1 if qualifies else 0, f1 if qualifies else acc, f1)
        elif mode == "cost_sensitive":
            cost = fp + fn_cost * fn
            key = (-cost, f1)
        else:
            key = (f1,)
        if best_key is None or key > best_key:
            best_key = key
            best_t = float(t)
            info.update({"val_accuracy": acc, "val_f1": f1, "threshold": best_t})
    return best_t, info


if HAS_TORCH:

    class _TorchBPN(nn.Module):
        """Two hidden layers + 2-unit logits for softmax classification."""

        def __init__(self, n_in: int, hidden: tuple[int, int], dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden[1], 2),
            )

        def forward(self, x):
            return self.net(x)
else:

    class _TorchBPN:  # type: ignore[no-redef]
        """Unpickle stand-in used when a saved run contains a torch module and torch is not installed."""

        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, *args, **kwargs) -> None:
            return None

        def __setstate__(self, state) -> None:
            self._state = state


class BPNClassifier(BaseEstimator, ClassifierMixin):
    """Feed-forward defect predictor with a 2-unit softmax output.

    Architecture: Dense(h1)-ReLU-Dropout-Dense(h2)-ReLU-Dropout-Dense(2).
    Defect probability is the softmax probability of class 1. Training uses
    weighted categorical cross-entropy. A decision threshold can be tuned on
    a validation split for F1.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        momentum: float = 0.9,
        epochs: int = 80,
        batch_size: int = 32,
        patience: int = 12,
        random_state: int = 42,
        decision_threshold: float = 0.5,
        tune_threshold: bool = True,
        threshold_mode: str = "f1",
        accuracy_floor: float = 0.65,
        fn_cost: float = 2.0,
        balance_classes: bool = True,
    ) -> None:
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.decision_threshold = decision_threshold
        self.tune_threshold = tune_threshold
        self.threshold_mode = threshold_mode
        self.accuracy_floor = accuracy_floor
        self.fn_cost = fn_cost
        self.balance_classes = balance_classes
        self.history_ = TrainHistory()
        self.classes_ = np.array([0, 1])
        self.decision_threshold_ = float(decision_threshold)

    def fit(self, X, y, X_val=None, y_val=None, callback=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if X_val is None or y_val is None:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(X))
            cut = max(int(0.15 * len(X)), 8)
            # keep at least one of each class in val when possible
            val_idx, train_idx = idx[:cut], idx[cut:]
            X_val, y_val = X[val_idx], y[val_idx]
            X, y = X[train_idx], y[train_idx]
        else:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64).reshape(-1)

        if HAS_TORCH:
            self._fit_torch(X, y, X_val, y_val, callback)
        else:
            self._fit_numpy(X, y, X_val, y_val, callback)

        self.decision_threshold_ = float(self.decision_threshold)
        self.threshold_info_ = {"mode": "fixed", "threshold": self.decision_threshold_}
        if self.tune_threshold:
            val_scores = self.predict_proba(X_val)[:, 1]
            t, info = _tune_threshold(
                y_val,
                val_scores,
                mode=str(self.threshold_mode),
                accuracy_floor=float(self.accuracy_floor),
                fn_cost=float(self.fn_cost),
            )
            self.decision_threshold_ = t
            self.threshold_info_ = info
        return self

    def _fit_torch(self, X, y, X_val, y_val, callback):
        torch.manual_seed(self.random_state)
        device = torch.device("cpu")
        model = _TorchBPN(X.shape[1], self.hidden_sizes, self.dropout).to(device)
        class_w = _class_weights(y) if self.balance_classes else np.ones(2, dtype=np.float32)
        weights = torch.tensor(class_w, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        if self.optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            opt = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum if "momentum" in self.optimizer else 0.0,
            )
        best_state = None
        best_val = float("inf")
        stall = 0
        history = TrainHistory(backend="torch")
        n = len(X)
        order = np.arange(n)
        rng = np.random.default_rng(self.random_state)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.long)

        for epoch in range(self.epochs):
            model.train()
            rng.shuffle(order)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                xb = X_t[batch]
                yb = y_t[batch]
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                tr_logits = model(X_t)
                va_logits = model(Xv)
                tr_loss = float(loss_fn(tr_logits, y_t).item())
                va_loss = float(loss_fn(va_logits, yv).item())
                tr_prob = torch.softmax(tr_logits, dim=1)[:, 1].numpy()
                va_prob = torch.softmax(va_logits, dim=1)[:, 1].numpy()
            tr_pred = (tr_prob >= 0.5).astype(int)
            va_pred = (va_prob >= 0.5).astype(int)
            tr_f1 = float(f1_score(y, tr_pred, zero_division=0))
            va_f1 = float(f1_score(y_val, va_pred, zero_division=0))
            tr_acc = float((tr_pred == y).mean())
            history.train_loss.append(tr_loss)
            history.val_loss.append(va_loss)
            history.train_f1.append(tr_f1)
            history.val_f1.append(va_f1)
            history.train_acc.append(tr_acc)
            if callback:
                callback(epoch + 1, self.epochs, history)
            if va_loss < best_val - 1e-4:
                best_val = va_loss
                stall = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stall += 1
                if stall >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.backend_ = "torch"
        self.n_features_in_ = X.shape[1]
        self.history_ = history

    def _fit_numpy(self, X, y, X_val, y_val, callback):
        rng = np.random.default_rng(self.random_state)
        n_in = X.shape[1]
        h1, h2 = self.hidden_sizes
        W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, h1)).astype(np.float32)
        b1 = np.zeros(h1, dtype=np.float32)
        W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2)).astype(np.float32)
        b2 = np.zeros(h2, dtype=np.float32)
        W3 = rng.normal(0, np.sqrt(2 / h2), (h2, 2)).astype(np.float32)
        b3 = np.zeros(2, dtype=np.float32)
        params = [W1, b1, W2, b2, W3, b3]
        use_adam = self.optimizer == "adam"
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        vel = [np.zeros_like(p) for p in params]
        class_w = _class_weights(y) if self.balance_classes else np.ones(2, dtype=np.float32)
        best = None
        best_val = float("inf")
        stall = 0
        history = TrainHistory(backend="numpy")
        t_step = 0
        n = len(X)

        def forward(xb, train=True):
            z1 = xb @ W1 + b1
            a1 = _relu(z1)
            if train and self.dropout > 0:
                mask1 = (rng.random(a1.shape) > self.dropout).astype(np.float32)
                a1 = a1 * mask1 / (1.0 - self.dropout)
            else:
                mask1 = None
            z2 = a1 @ W2 + b2
            a2 = _relu(z2)
            if train and self.dropout > 0:
                mask2 = (rng.random(a2.shape) > self.dropout).astype(np.float32)
                a2 = a2 * mask2 / (1.0 - self.dropout)
            else:
                mask2 = None
            z3 = a2 @ W3 + b3
            p = _softmax(z3)
            cache = (xb, z1, a1, mask1, z2, a2, mask2, z3, p)
            return p, cache

        def ce_loss(p, yb):
            p = np.clip(p, 1e-6, 1.0)
            rows = np.arange(len(yb))
            w = class_w[yb.astype(int)]
            return float(np.mean(-w * np.log(p[rows, yb.astype(int)])))

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                xb, yb = X[batch], y[batch].astype(int)
                p, cache = forward(xb, train=True)
                xb, z1, a1, mask1, z2, a2, mask2, z3, _ = cache
                # dL/dz3 for weighted mean CE: w_y * (p - one_hot) / batch
                one_hot = np.zeros_like(p)
                one_hot[np.arange(len(yb)), yb] = 1.0
                w = class_w[yb].reshape(-1, 1)
                dz3 = (w * (p - one_hot)) / max(len(yb), 1)
                gW3 = a2.T @ dz3
                gb3 = dz3.sum(axis=0)
                da2 = dz3 @ W3.T
                if mask2 is not None:
                    da2 = da2 * mask2 / (1.0 - self.dropout)
                dz2 = da2 * (z2 > 0)
                gW2 = a1.T @ dz2
                gb2 = dz2.sum(axis=0)
                da1 = dz2 @ W2.T
                if mask1 is not None:
                    da1 = da1 * mask1 / (1.0 - self.dropout)
                dz1 = da1 * (z1 > 0)
                gW1 = xb.T @ dz1
                gb1 = dz1.sum(axis=0)
                grads = [gW1, gb1, gW2, gb2, gW3, gb3]
                t_step += 1
                if use_adam:
                    beta1, beta2, eps = 0.9, 0.999, 1e-8
                    for i, g in enumerate(grads):
                        m[i] = beta1 * m[i] + (1 - beta1) * g
                        v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                        mhat = m[i] / (1 - beta1**t_step)
                        vhat = v[i] / (1 - beta2**t_step)
                        params[i] -= self.learning_rate * mhat / (np.sqrt(vhat) + eps)
                else:
                    mu = self.momentum if "momentum" in self.optimizer else 0.0
                    for i, g in enumerate(grads):
                        vel[i] = mu * vel[i] - self.learning_rate * g
                        params[i] += vel[i]
                W1, b1, W2, b2, W3, b3 = params

            tr_p, _ = forward(X, train=False)
            va_p, _ = forward(X_val, train=False)
            tr_loss, va_loss = ce_loss(tr_p, y.astype(int)), ce_loss(va_p, y_val.astype(int))
            tr_pred = (tr_p[:, 1] >= 0.5).astype(int)
            va_pred = (va_p[:, 1] >= 0.5).astype(int)
            tr_f1 = float(f1_score(y, tr_pred, zero_division=0))
            va_f1 = float(f1_score(y_val, va_pred, zero_division=0))
            history.train_loss.append(tr_loss)
            history.val_loss.append(va_loss)
            history.train_f1.append(tr_f1)
            history.val_f1.append(va_f1)
            history.train_acc.append(float((tr_pred == y).mean()))
            if callback:
                callback(epoch + 1, self.epochs, history)
            if va_loss < best_val - 1e-4:
                best_val = va_loss
                stall = 0
                best = [p.copy() for p in params]
            else:
                stall += 1
                if stall >= self.patience:
                    break
        if best is not None:
            self.weights_ = best
        else:
            self.weights_ = params
        self.backend_ = "numpy"
        self.n_features_in_ = n_in
        self.history_ = history

    def _numpy_proba(self, X: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2, W3, b3 = self.weights_
        a1 = _relu(X @ W1 + b1)
        a2 = _relu(a1 @ W2 + b2)
        return _softmax(a2 @ W3 + b3)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if getattr(self, "backend_", "numpy") == "torch" and HAS_TORCH:
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(torch.tensor(X, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1).numpy()
        else:
            probs = self._numpy_proba(X)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        # renormalise after clip for numerical safety
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs.astype(np.float64)

    def predict(self, X):
        threshold = float(getattr(self, "decision_threshold_", self.decision_threshold))
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save_weights(self) -> dict:
        payload = {
            "n_features": int(self.n_features_in_),
            "hidden": self.hidden_sizes,
            "dropout": self.dropout,
            "decision_threshold": float(getattr(self, "decision_threshold_", self.decision_threshold)),
            "output": "softmax2",
        }
        if getattr(self, "backend_", "numpy") == "torch" and HAS_TORCH:
            payload.update(
                {
                    "backend": "torch",
                    "state_dict": {k: v.cpu().numpy() for k, v in self.model_.state_dict().items()},
                }
            )
            return payload
        payload.update(
            {
                "backend": "numpy",
                "weights": [w.copy() for w in self.weights_],
            }
        )
        return payload

    @classmethod
    def from_weights(cls, payload: dict, **kwargs) -> "BPNClassifier":
        model = cls(hidden_sizes=tuple(payload.get("hidden", (64, 32))), **kwargs)
        model.n_features_in_ = payload["n_features"]
        model.backend_ = payload["backend"]
        model.decision_threshold_ = float(
            payload.get("decision_threshold", kwargs.get("decision_threshold", 0.5))
        )
        if payload["backend"] == "torch" and HAS_TORCH:
            net = _TorchBPN(payload["n_features"], tuple(payload["hidden"]), payload.get("dropout", 0.2))
            state = {k: torch.tensor(v) for k, v in payload["state_dict"].items()}
            # Old single-logit checkpoints are incompatible with softmax-2; fail clearly.
            out_key = "net.6.weight" if "net.6.weight" in state else None
            if out_key is not None and state[out_key].ndim == 2 and state[out_key].shape[0] != 2:
                raise ValueError(
                    "Saved BPN weights use a single-logit output; retrain after the softmax upgrade."
                )
            net.load_state_dict(state)
            net.eval()
            model.model_ = net
        elif payload["backend"] == "torch":
            # Checkpoints trained with torch stay usable through the NumPy forward pass.
            model.weights_ = _numpy_weights_from_torch_state(payload["state_dict"])
            model.backend_ = "numpy"
            if np.asarray(model.weights_[4]).shape[1] != 2:
                raise ValueError(
                    "Saved BPN weights use a single-logit output; retrain after the softmax upgrade."
                )
        else:
            weights = payload["weights"]
            if len(weights) >= 5 and np.asarray(weights[4]).ndim == 2 and np.asarray(weights[4]).shape[1] != 2:
                raise ValueError(
                    "Saved BPN weights use a single-logit output; retrain after the softmax upgrade."
                )
            model.weights_ = weights
            model.backend_ = "numpy"
        return model
