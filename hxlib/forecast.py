"""Small single-layer LSTM implemented in NumPy (no TensorFlow/PyTorch needed) for Rf forecasting."""
from __future__ import annotations
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTM:
    def __init__(self, hidden=16, seed=0):
        r = np.random.default_rng(seed)
        H, s = hidden, 1.0 / np.sqrt(hidden)
        self.H = H
        self.p = {"Wx": r.normal(0, s, (1, 4 * H)), "Wh": r.normal(0, s, (H, 4 * H)), "b": np.zeros(4 * H),
                  "Wy": r.normal(0, s, (H, 1)), "by": np.zeros(1)}
        self.p["b"][H:2 * H] = 1.0   # forget-gate bias

    def forward(self, X):
        P, H = self.p, self.H
        B, T, _ = X.shape
        h, c, cache = np.zeros((B, H)), np.zeros((B, H)), []
        for t in range(T):
            x = X[:, t, :]
            z = x @ P["Wx"] + h @ P["Wh"] + P["b"]
            i, f = _sig(z[:, :H]), _sig(z[:, H:2 * H])
            g, o = np.tanh(z[:, 2 * H:3 * H]), _sig(z[:, 3 * H:])
            cn = f * c + i * g
            tc = np.tanh(cn)
            cache.append((x, h, c, i, f, g, o, tc))
            h, c = o * tc, cn
        return h @ P["Wy"] + P["by"], h, cache

    def backward(self, cache, h_last, dy):
        P, H = self.p, self.H
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["Wy"], G["by"] = h_last.T @ dy, dy.sum(0)
        dh, dc = dy @ P["Wy"].T, np.zeros_like(h_last)
        for x, hp, cp, i, f, g, o, tc in reversed(cache):
            do = dh * tc
            dct = dc + dh * o * (1 - tc ** 2)
            dz = np.concatenate([dct * g * i * (1 - i), dct * cp * f * (1 - f),
                                 dct * i * (1 - g ** 2), do * o * (1 - o)], axis=1)
            G["Wx"] += x.T @ dz
            G["Wh"] += hp.T @ dz
            G["b"] += dz.sum(0)
            dh, dc = dz @ P["Wh"].T, dct * f
        return G

    def predict(self, X):
        return self.forward(X)[0]


def _windows(z, w):
    X = np.stack([z[i:i + w] for i in range(len(z) - w)])[:, :, None]
    return X, z[w:][:, None]


def train_and_forecast(series, window=10, hidden=16, epochs=40, horizon=30, lr=0.01, seed=0, batch=64):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < window + 20:
        raise ValueError(f"Need at least {window + 20} points; got {len(s)}.")
    mu, sd = s.mean(), s.std() or 1.0
    z = (s - mu) / sd
    X, y = _windows(z, window)
    nv = max(5, int(0.2 * len(X)))
    Xtr, ytr, Xva, yva = X[:-nv], y[:-nv], X[-nv:], y[-nv:]
    m = NumpyLSTM(hidden, seed)
    M1 = {k: np.zeros_like(v) for k, v in m.p.items()}
    M2 = {k: np.zeros_like(v) for k, v in m.p.items()}
    rng, step, hist = np.random.default_rng(seed), 0, []
    for _ in range(epochs):
        order = rng.permutation(len(Xtr))
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            yh, hl, cache = m.forward(Xtr[idx])
            G = m.backward(cache, hl, 2 * (yh - ytr[idx]) / len(idx))
            norm = np.sqrt(sum((g ** 2).sum() for g in G.values()))
            sc = min(1.0, 1.0 / (norm + 1e-12))
            step += 1
            for key in m.p:
                g = G[key] * sc
                M1[key] = 0.9 * M1[key] + 0.1 * g
                M2[key] = 0.999 * M2[key] + 0.001 * g * g
                m.p[key] -= lr * (M1[key] / (1 - 0.9 ** step)) / (np.sqrt(M2[key] / (1 - 0.999 ** step)) + 1e-8)
        hist.append(float(np.sqrt(np.mean((m.predict(Xva) - yva) ** 2))))
    val_pred = m.predict(Xva)[:, 0] * sd + mu
    val_true = yva[:, 0] * sd + mu
    naive = Xva[:, -1, 0] * sd + mu                      # persistence baseline
    buf, fc = list(z[-window:]), []
    for _ in range(horizon):
        nxt = float(m.predict(np.array(buf[-window:])[None, :, None])[0, 0])
        fc.append(nxt)
        buf.append(nxt)
    return {"history": s, "val_pred": val_pred, "val_true": val_true, "val_start": len(s) - nv,
            "forecast": np.array(fc) * sd + mu, "rmse": float(np.sqrt(np.mean((val_pred - val_true) ** 2))),
            "naive_rmse": float(np.sqrt(np.mean((naive - val_true) ** 2))), "loss_curve": hist}
