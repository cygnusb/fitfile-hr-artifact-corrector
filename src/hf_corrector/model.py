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
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.head(y)


class HRModel:
    """Torch-first HR model with sequence training and legacy loader fallback."""

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
        device: str | None = None,
        seq_len: int = 120,
        stride: int = 10,
        epochs: int = 12,
        batch_size: int = 64,
        lr: float = 1e-3,
        lambda_dynamics: float = 0.35,
        max_drop_bpm_per_step: float = 1.2,
        max_windows: int = 40000,
    ) -> "HRModel":
        if not x_groups or not y_groups or len(x_groups) != len(y_groups):
            raise ValueError("x_groups and y_groups must be non-empty and aligned")

        all_x = np.concatenate([x for x in x_groups if len(x) > 0], axis=0)
        if len(all_x) == 0:
            raise ValueError("No training rows available")

        mu = all_x.mean(axis=0)
        sigma = all_x.std(axis=0)
        sigma[sigma == 0.0] = 1.0

        windows_x: list[np.ndarray] = []
        windows_y: list[np.ndarray] = []
        windows_m: list[np.ndarray] = []
        for x_raw, y_raw in zip(x_groups, y_groups):
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
            raise ValueError("No sequence windows available; lower seq_len or add more data")

        if len(windows_x) > max_windows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(windows_x), size=max_windows, replace=False)
            windows_x = [windows_x[i] for i in idx]
            windows_y = [windows_y[i] for i in idx]
            windows_m = [windows_m[i] for i in idx]

        x_arr = np.stack(windows_x).astype(np.float32)
        y_arr = np.stack(windows_y).astype(np.float32)
        m_arr = np.stack(windows_m).astype(np.float32)

        resolved_device = _resolve_device(device)
        dev = torch.device(resolved_device)
        model = _TorchSequenceRegressor(in_dim=x_arr.shape[-1], hidden_dim=64).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        x_tensor = torch.from_numpy(x_arr).to(dev)
        y_tensor = torch.from_numpy(y_arr).to(dev)
        m_tensor = torch.from_numpy(m_arr).to(dev)

        n = x_tensor.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                bx = x_tensor[idx]
                by = y_tensor[idx]
                bm = m_tensor[idx]
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
                seq = torch.from_numpy(xz).unsqueeze(0)
                out = self.model(seq).squeeze(0).squeeze(-1).cpu().numpy()
                return out.astype(float)
            if self.backend == "torch_mlp":
                out = self.model(torch.from_numpy(xz)).squeeze(-1).cpu().numpy()
                return out.astype(float)
        raise RuntimeError(f"Unsupported backend in predict: {self.backend}")

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

            if backend == "torch_gru":
                model = _TorchSequenceRegressor(in_dim=len(FEATURE_NAMES), hidden_dim=64)
            elif backend == "torch_mlp":
                model = _TorchRegressor(in_dim=len(FEATURE_NAMES))
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
        model = _TorchRegressor(in_dim=len(FEATURE_NAMES))
        with torch.no_grad():
            # Approximate legacy ridge as first-layer linear projection.
            # This fallback is only for compatibility and is not used for new training.
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
