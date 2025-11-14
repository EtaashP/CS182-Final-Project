from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class TrainingVisualizer:
    """
    Robust logger for training.

    Features:
      - Global (run-level) hyperparameters: saved once, injected into every epoch.
      - Per-epoch metrics: train_loss, val_acc, lr, batch_size, etc.
      - Safe scalar conversion, but containers (list/tuple/dict) preserved.
      - Guaranteed stable column schema across epochs.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._globals: Dict[str, Any] = {}    # global hyperparameters

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def set_global(self, **kwargs: Any) -> None:
        """
        Set run-level (global) hyperparameters.
        These will be injected into every epoch's record.
        """
        for k, v in kwargs.items():
            self._globals[k] = self._safe_scalar(v)

    def update(self, epoch: int, **kwargs: Any) -> None:
        """
        Log per-epoch metrics (train_loss, val_acc, lr, etc.)
        Automatically includes global hyperparameters.
        """
        rec: Dict[str, Any] = {"epoch": int(epoch)}

        # add global hyperparameters EVERY epoch
        rec.update(self._globals)

        # add per-epoch values
        for k, v in kwargs.items():
            rec[k] = v
        self._records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame with a stable schema."""
        if not self._records:
            return pd.DataFrame(columns=["epoch"])

        # Use from_records: more stable than pd.DataFrame(list_of_dicts)
        df = pd.DataFrame.from_records(self._records)

        # Guarantee column ordering: epoch first, globals next, epoch metrics last
        cols = ["epoch"] + sorted(self._globals.keys()) + \
               [c for c in df.columns if c not in self._globals and c != "epoch"]
        df = df[cols]
        return df

    def save_csv(self, filename: str, index: bool = False) -> str:
        df = self.to_dataframe()
        out_path = Path(filename)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=index)
        return str(out_path.resolve())

    def reset(self) -> None:
        self._records = []
        self._globals = {}

    def get_best_epoch(self, metric: str = "val_acc") -> Optional[Dict[str, Any]]:
        df = self.to_dataframe()
        if df.empty or metric not in df.columns:
            return None
        idx = df[metric].idxmax()
        return df.loc[idx].to_dict()

    # ------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------
    @staticmethod
    def _safe_scalar(v: Any) -> Any:
        """
        Convert torch/numpy scalar to Python scalar when appropriate.
        Preserve list/tuple/dict containers exactly.
        """

        # Preserve containers (important for Adam betas)
        if isinstance(v, (list, tuple, dict)):
            return v

        # Strings unchanged
        if isinstance(v, (str, bytes)):
            return v

        # torch/numpy scalar via .item()
        try:
            if hasattr(v, "item") and callable(v.item):
                # Only use item() if it's 0-d (true scalar)
                if getattr(v, "ndim", 0) == 0:
                    return v.item()
        except Exception:
            pass

        # numpy 0-d array
        try:
            if isinstance(v, np.ndarray) and v.ndim == 0:
                return v.item()
        except Exception:
            pass

        # fallback (non-scalar object preserved)
        return v
