from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .features import FEATURE_NAMES


def pick_compute_device() -> str:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(device: str | None) -> str:
    if device is None or device == "auto":
        return pick_compute_device()
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be one of: auto, cpu, mps")
    if device == "mps" and pick_compute_device() != "mps":
        raise RuntimeError("MPS requested but not available on this runtime")
    return device


def _split_group_indices(group_ids: list[str], val_fraction: float) -> tuple[list[int], list[int]]:
    if len(group_ids) == 0:
        return [], []

    unique_groups = list(dict.fromkeys(group_ids))
    if len(unique_groups) < 2 or val_fraction <= 0.0:
        all_idx = list(range(len(group_ids)))
        return all_idx, []

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(unique_groups))
    n_val_groups = max(1, int(round(len(unique_groups) * val_fraction)))
    if n_val_groups >= len(unique_groups):
        n_val_groups = len(unique_groups) - 1

    val_group_set = {unique_groups[i] for i in perm[:n_val_groups]}
    train_idx = [i for i, group_id in enumerate(group_ids) if group_id not in val_group_set]
    val_idx = [i for i, group_id in enumerate(group_ids) if group_id in val_group_set]
    return train_idx, val_idx


class _TorchRegressor(nn.Module):
    # Legacy MLP shape kept for backward compatibility with older checkpoints.
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TorchSequenceRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, mc_dropout: float = 0.1):
        super().__init__()
        self.mc_dropout_rate = mc_dropout
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=mc_dropout,
        )
        self.dropout = nn.Dropout(mc_dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        y = self.dropout(y)
        return self.head(y)


class HRModel:
    """Torch-first HR model with sequence training and legacy loader fallback."""

    MC_FORWARD_PASSES = 10

    def __init__(
        self,
        *,
        backend: str,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        model: nn.Module,
        train_device: str,
        seq_len: int = 120,
    ):
        self.backend = backend
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.model = model
        self.train_device = train_device
        self.seq_len = seq_len

    @classmethod
    def fit_from_groups(
        cls,
        x_groups: list[np.ndarray],
        y_groups: list[np.ndarray],
        *,
        group_ids: list[str] | None = None,
        device: str | None = None,
        seq_len: int = 120,
        stride: int = 10,
        epochs: int = 12,
        batch_size: int = 64,
        lr: float = 1e-3,
        lambda_dynamics: float = 0.35,
        max_drop_bpm_per_step: float = 1.2,
        max_windows: int = 40000,
        patience: int = 5,
        val_fraction: float = 0.15,
    ) -> "HRModel":
        if not x_groups or not y_groups or len(x_groups) != len(y_groups):
            raise ValueError("x_groups and y_groups must be non-empty and aligned")
        if group_ids is None:
            group_ids = [f"group_{idx}" for idx in range(len(x_groups))]
        if len(group_ids) != len(x_groups):
            raise ValueError("group_ids must be aligned with x_groups and y_groups")

        train_group_idx, val_group_idx = _split_group_indices(group_ids, val_fraction)
        if not train_group_idx:
            raise ValueError("No training groups available after validation split")

        train_groups = [(x_groups[idx], y_groups[idx]) for idx in train_group_idx]
        val_groups = [(x_groups[idx], y_groups[idx]) for idx in val_group_idx]

        all_x = np.concatenate([x for x, _ in train_groups if len(x) > 0], axis=0)
        if len(all_x) == 0:
            raise ValueError("No training rows available")

        mu = all_x.mean(axis=0)
        sigma = all_x.std(axis=0)
        sigma[sigma == 0.0] = 1.0

        train_windows = _build_windows(train_groups, mu, sigma, seq_len=seq_len, stride=stride)
        if train_windows is None:
            raise ValueError("No sequence windows available; lower seq_len or add more data")

        train_x, train_y, train_m = train_windows
        if len(train_x) > max_windows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(train_x), size=max_windows, replace=False)
            train_x = [train_x[i] for i in idx]
            train_y = [train_y[i] for i in idx]
            train_m = [train_m[i] for i in idx]

        x_train = np.stack(train_x).astype(np.float32)
        y_train = np.stack(train_y).astype(np.float32)
        m_train = np.stack(train_m).astype(np.float32)

        val_windows = _build_windows(val_groups, mu, sigma, seq_len=seq_len, stride=stride)
        has_val = val_windows is not None
        if has_val:
            val_x, val_y, val_m = val_windows
            x_val = np.stack(val_x).astype(np.float32)
            y_val = np.stack(val_y).astype(np.float32)
            m_val = np.stack(val_m).astype(np.float32)
        else:
            x_val = y_val = m_val = None

        resolved_device = _resolve_device(device)
        dev = torch.device(resolved_device)
        model = _TorchSequenceRegressor(in_dim=x_train.shape[-1], hidden_dim=64).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # --- LR Scheduler (OneCycleLR) ---
        steps_per_epoch = max(1, (x_train.shape[0] + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )

        xt = torch.from_numpy(x_train).to(dev)
        yt = torch.from_numpy(y_train).to(dev)
        mt = torch.from_numpy(m_train).to(dev)
        xv = torch.from_numpy(x_val).to(dev) if x_val is not None else None
        yv = torch.from_numpy(y_val).to(dev) if y_val is not None else None
        mv = torch.from_numpy(m_val).to(dev) if m_val is not None else None

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n = xt.shape[0]
        for _ in range(epochs):
            model.train()
            perm_t = torch.randperm(n, device=dev)
            for start in range(0, n, batch_size):
                idx = perm_t[start : start + batch_size]
                bx = xt[idx]
                by = yt[idx]
                bm = mt[idx]
                pred = model(bx)

                se = (pred - by) ** 2
                mse = (se * bm).sum() / torch.clamp_min(bm.sum(), 1.0)

                d_pred = pred[:, 1:, :] - pred[:, :-1, :]
                excess_drop = torch.relu((-d_pred) - max_drop_bpm_per_step)
                dyn = excess_drop.mean()
                loss = mse + lambda_dynamics * dyn

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

            if xv is None or yv is None or mv is None:
                continue

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_pred = model(xv)
                val_se = (val_pred - yv) ** 2
                val_loss = float((val_se * mv).sum() / torch.clamp_min(mv.sum(), 1.0))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        model = model.to("cpu")
        model.eval()
        return cls(
            backend="torch_gru",
            feature_mean=mu,
            feature_std=sigma,
            model=model,
            train_device=resolved_device,
            seq_len=seq_len,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros((0,), dtype=float)
        xz = ((x - self.feature_mean) / self.feature_std).astype(np.float32)

        with torch.no_grad():
            if self.backend == "torch_gru":
                return self._predict_gru_overlapping(xz)
            if self.backend == "torch_mlp":
                out = self.model(torch.from_numpy(xz)).squeeze(-1).cpu().numpy()
                return out.astype(float)
        raise RuntimeError(f"Unsupported backend in predict: {self.backend}")

    def _predict_gru_overlapping(self, xz: np.ndarray) -> np.ndarray:
        """Overlapping-window inference for GRU to stay close to training regime."""
        n = len(xz)
        if n <= self.seq_len:
            seq = torch.from_numpy(xz).unsqueeze(0)
            out = self.model(seq).squeeze(0).squeeze(-1).cpu().numpy()
            return out.astype(float)

        stride = max(1, self.seq_len // 2)
        accum = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)

        for start in range(0, n - self.seq_len + 1, stride):
            end = start + self.seq_len
            chunk = torch.from_numpy(xz[start:end]).unsqueeze(0)
            pred = self.model(chunk).squeeze(0).squeeze(-1).cpu().numpy()
            accum[start:end] += pred
            counts[start:end] += 1.0

        # Handle tail if last window doesn't reach the end
        last_start = n - self.seq_len
        if counts[n - 1] == 0:
            chunk = torch.from_numpy(xz[last_start:n]).unsqueeze(0)
            pred = self.model(chunk).squeeze(0).squeeze(-1).cpu().numpy()
            accum[last_start:n] += pred
            counts[last_start:n] += 1.0

        counts[counts == 0] = 1.0
        return (accum / counts).astype(float)

    def predict_with_uncertainty(self, x: np.ndarray, n_passes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """MC-Dropout inference: returns (mean_prediction, std_uncertainty)."""
        if len(x) == 0:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
        if self.backend != "torch_gru":
            pred = self.predict(x)
            return pred, np.zeros_like(pred)

        n_passes = n_passes or self.MC_FORWARD_PASSES
        xz = ((x - self.feature_mean) / self.feature_std).astype(np.float32)

        self.model.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                out = self._predict_gru_overlapping(xz)
                preds.append(out)
        self.model.eval()

        stacked = np.stack(preds, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        return mean.astype(float), std.astype(float)

    def save(self, out_dir: str | Path, metadata: dict[str, Any] | None = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "backend": self.backend,
            "feature_mean": self.feature_mean.astype(np.float32),
            "feature_std": self.feature_std.astype(np.float32),
            "feature_names": FEATURE_NAMES,
            "train_device": self.train_device,
            "seq_len": self.seq_len,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, out / "model.pt")

        config = {
            "model_type": self.backend,
            "feature_names": FEATURE_NAMES,
            "device": self.train_device,
            "seq_len": self.seq_len,
        }
        if metadata:
            config.update(metadata)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "HRModel":
        md = Path(model_dir)
        pt_path = md / "model.pt"
        if pt_path.exists():
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            backend = str(payload.get("backend", "torch_mlp"))
            mean = np.asarray(payload["feature_mean"], dtype=float)
            std = np.asarray(payload["feature_std"], dtype=float)
            train_device = str(payload.get("train_device", "cpu"))
            seq_len = int(payload.get("seq_len", 120))
            saved_names = payload.get("feature_names", [])
            in_dim = len(saved_names) if saved_names else len(FEATURE_NAMES)

            if backend == "torch_gru":
                model = _TorchSequenceRegressor(in_dim=in_dim, hidden_dim=64)
            elif backend == "torch_mlp":
                model = _TorchRegressor(in_dim=in_dim)
            else:
                raise RuntimeError(f"Unsupported backend in checkpoint: {backend}")

            model.load_state_dict(payload["state_dict"])
            model.eval()
            return cls(
                backend=backend,
                feature_mean=mean,
                feature_std=std,
                model=model,
                train_device=train_device,
                seq_len=seq_len,
            )

        # Legacy fallback (.npy ridge model from earliest version).
        weights = np.load(md / "weights.npy")
        mean = np.load(md / "feature_mean.npy")
        std = np.load(md / "feature_std.npy")
        cfg = json.loads((md / "config.json").read_text(encoding="utf-8"))
        in_dim = len(weights)
        model = _TorchRegressor(in_dim=in_dim)
        with torch.no_grad():
            model.net[0].weight.zero_()
            model.net[0].bias.zero_()
            w = torch.from_numpy(weights.astype(np.float32))
            model.net[0].weight[0, : len(w)] = w
            model.net[4].bias.fill_(float(cfg.get("bias", 0.0)))
        model.eval()
        return cls(
            backend="torch_mlp",
            feature_mean=mean,
            feature_std=std,
            model=model,
            train_device=str(cfg.get("device", "cpu")),
            seq_len=int(cfg.get("seq_len", 120)),
        )


def _build_windows(
    groups: list[tuple[np.ndarray, np.ndarray]],
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    seq_len: int,
    stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    windows_x: list[np.ndarray] = []
    windows_y: list[np.ndarray] = []
    windows_m: list[np.ndarray] = []
    for x_raw, y_raw in groups:
        if len(x_raw) < seq_len:
            continue
        xz = ((x_raw - mu) / sigma).astype(np.float32)
        y = y_raw.astype(np.float32)
        for start in range(0, len(xz) - seq_len + 1, stride):
            end = start + seq_len
            yw = y[start:end]
            mask = np.isfinite(yw).astype(np.float32)
            if mask.sum() < seq_len * 0.6:
                continue
            windows_x.append(xz[start:end])
            windows_y.append(np.nan_to_num(yw, nan=0.0).reshape(-1, 1))
            windows_m.append(mask.reshape(-1, 1))

    if not windows_x:
        return None
    return windows_x, windows_y, windows_m
