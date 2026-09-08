"""
config_loader.py
=================

Loads configs/config.yaml into a single `Config` object that the rest of
the project uses instead of re-parsing YAML or hardcoding parameters.

Why a class instead of a raw dict
----------------------------------
A thin wrapper around the parsed dict gives us:
- Dot-notation access (`config.detection.confidence_threshold`) which is
  far more readable than `config["detection"]["confidence_threshold"]`
  once configs get deep, as ours will by Phase 6-7.
- A single validated place to resolve the device ("auto" -> actual
  torch device), instead of every module re-implementing that check.
- Fail-fast behavior: if config.yaml is malformed or missing, the whole
  pipeline should refuse to start rather than fail confusingly deep inside
  a training loop.

Usage
-----
    from utils.config_loader import Config

    cfg = Config("configs/config.yaml")
    print(cfg.detection.confidence_threshold)   # 0.6
    print(cfg.resolved_device)                  # "cuda" or "cpu"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class _AttrDict(dict):
    """
    A dict subclass that also allows attribute-style access
    (d.key as well as d["key"]), applied recursively to nested dicts.

    This is intentionally minimal (no external dependency like
    `addict` or `omegaconf`) to keep the dependency list lean, per the
    project's "keep it modular but not bloated" spirit.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = _AttrDict(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no field '{name}'. Check configs/config.yaml."
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Config:
    """
    Loads and validates configs/config.yaml, exposing every section as an
    attribute (e.g. `self.detection`, `self.tracking`, `self.training`).

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML is empty or fails to parse into a dictionary.
    """

    REQUIRED_SECTIONS = (
        "project",
        "device",
        "detection",
        "tracking",
        "dataset",
        "quality_filters",
        "embedding",
        "training",
        "recognition",
        "performance",
        "logging",
    )

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found at '{path}'. "
                f"Every module expects configs/config.yaml to exist."
            )

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        if not raw or not isinstance(raw, dict):
            raise ValueError(f"Config file '{path}' is empty or malformed.")

        self._validate_sections(raw)
        self._data = _AttrDict(raw)
        self._config_path = path

    def _validate_sections(self, raw: Dict[str, Any]) -> None:
        """Fail fast if a required top-level section is missing."""
        missing = [s for s in self.REQUIRED_SECTIONS if s not in raw]
        if missing:
            raise ValueError(
                f"Config file is missing required section(s): {missing}. "
                f"See configs/config.yaml for the expected schema."
            )

    def __getattr__(self, name: str) -> Any:
        # Delegates attribute access (cfg.detection, cfg.training, ...)
        # to the underlying _AttrDict once __init__ has set self._data.
        data = self.__dict__.get("_data")
        if data is not None and name in data:
            return data[name]
        raise AttributeError(f"Config has no section '{name}'.")

    @property
    def resolved_device(self) -> str:
        """
        Resolve the 'device.mode' setting ("auto" / "cuda" / "cpu") into a
        concrete torch device string. Imports torch lazily so that modules
        which only need path/config utilities don't require torch installed.
        """
        mode = self.device.mode

        if mode == "cpu":
            return "cpu"

        import torch  # local import: keep torch optional for non-DL utilities

        if mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Config requests device.mode='cuda' but no CUDA device "
                    "is available. Set device.mode to 'auto' or 'cpu'."
                )
            return f"cuda:{self.device.gpu_id}"

        # mode == "auto"
        return f"cuda:{self.device.gpu_id}" if torch.cuda.is_available() else "cpu"

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"Config(path='{self._config_path}')"
