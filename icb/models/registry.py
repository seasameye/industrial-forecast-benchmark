# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from .lstm import LSTMEncDec
from .official_dag import OfficialDAGAdapter
from .official_timexer import OfficialTimeXerAdapter
from .persistence import Persistence
from .tslib import TSLibAdapter


def _default_tsl_repo() -> str:
    return "third_party_repos/DAG/ts_benchmark/baselines/time_series_library"


def available_models() -> list[str]:
    return [
        "persistence",
        "lstm",
        "crosslinear",
        "tide",
        "timexer",
        "timemixer",
        "timemixer_adapter",
        "patchtst",
        "dlinear",
        "dag",
        "kite",
    ]


def build_model(name: str, d_y: int, d_u: int, L: int, H: int,
                **kwargs: Any):
    key = name.lower()
    if key == "persistence":
        return Persistence()
    if key == "lstm":
        return LSTMEncDec(d_y, d_u, hidden=kwargs.get("hidden", 128),
                          layers=kwargs.get("layers", 2),
                          dropout=kwargs.get("dropout", 0.1))
    if key == "crosslinear":
        return TSLibAdapter("CrossLinear", d_y, d_u, L, H, **kwargs)
    if key == "tide":
        return TSLibAdapter("TiDE", d_y, d_u, L, H, **kwargs)
    if key == "timexer":
        return OfficialTimeXerAdapter(d_y, d_u, L, H, **kwargs)
    if key in ("timemixer", "timemixer_adapter"):
        kwargs.setdefault("channel_independence", 0)
        kwargs.setdefault("repo", _default_tsl_repo())
        return TSLibAdapter("TimeMixer", d_y, d_u, L, H, **kwargs)
    if key == "patchtst":
        kwargs.setdefault("repo", _default_tsl_repo())
        kwargs.setdefault("exo_mode", "hist")
        return TSLibAdapter("PatchTST", d_y, d_u, L, H, **kwargs)
    if key == "dlinear":
        kwargs.setdefault("repo", _default_tsl_repo())
        kwargs.setdefault("exo_mode", "hist")
        return TSLibAdapter("DLinear", d_y, d_u, L, H, **kwargs)
    if key == "dag":
        return OfficialDAGAdapter(d_y, d_u, L, H, **kwargs)
    if key == "kite":
        from .kite import KITEOfficialAdapter
        return KITEOfficialAdapter(d_y, d_u, L, H, **kwargs)
    raise ValueError(f"unknown model {name}. Available: {available_models()}")
