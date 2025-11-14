# vit_visualizer.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

class TrainingVisualizer:
    """
    Lightweight logger that collects per-epoch metrics and hyperparameters.
    - Call `update(epoch, **kwargs)` each epoch (kwargs can include train_loss, train_acc, val_acc,
      lr, weight_decay, batch_size, drop_path_rate, etc.).
    - After training, call `to_dataframe()` or `save_csv(filename)`.

    This class:
    - Converts simple torch/numpy scalars via .item() when possible.
    - Keeps raw lists of records which convert to a DataFrame on demand.
    - Provides helper `get_best_epoch()` to find the best epoch by a metric.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    # -------------------------
    # Public API
    # -------------------------
    def update(self, epoch: int, **kwargs: Any) -> None:
        """
        Record one epoch. `epoch` should be an int (1-based preferred).
        kwargs: arbitrary keys (train_loss, train_acc, val_acc, lr, weight_decay, ...).
        Scalars that have `.item()` (torch / numpy scalars) will be converted to Python numbers.
        """
        rec: Dict[str, Any] = {"epoch": int(epoch)}
        for k, v in kwargs.items():
            rec[k] = self._safe_scalar(v)
        self._records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame with all collected records (may be empty)."""
        if not self._records:
            # return empty DataFrame with a predictable schema
            return pd.DataFrame(columns=["epoch"])
        return pd.DataFrame(self._records)

    def save_csv(self, filename: str, index: bool = False) -> str:
        """
        Save collected records to CSV. Creates parent directories if necessary.
        Returns the final filename used (absolute path as string).
        """
        df = self.to_dataframe()
        out_path = Path(filename)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=index)
        return str(out_path.resolve())

    def reset(self) -> None:
        """Clear all collected records (so the same object can be reused for another run)."""
        self._records = []

    def get_best_epoch(self, metric: str = "val_acc") -> Optional[Dict[str, Any]]:
        """
        Return the record (dict) of the row with max(metric).
        If metric is not present or no records, returns None.
        """
        df = self.to_dataframe()
        if df.empty or metric not in df.columns:
            return None
        idx = df[metric].idxmax()
        return df.loc[idx].to_dict()

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def _safe_scalar(v: Any) -> Any:
        """
        Convert torch.tensor / numpy scalar to Python scalar if possible.
        Leave lists/dicts/strings alone.
        """
        # strings/bytes: keep as-is
        if isinstance(v, (str, bytes)):
            return v

        # try .item() for torch scalar or numpy scalar or 0-d numpy array
        try:
            if hasattr(v, "item") and callable(v.item):
                # Some objects (large arrays) have item but will raise if not scalar,
                # so guard with try/except.
                try:
                    return v.item()
                except Exception:
                    print("error does not exist:", v)
        except Exception:
            pass

        # if it's a one-element list/tuple/np.ndarray convert to scalar, otherwise keep structure
        try:
            import numpy as _np
            if isinstance(v, _np.ndarray) and v.size == 1:
                return v.item()
        except Exception:
            print("error does not exist:", v)

        return v
