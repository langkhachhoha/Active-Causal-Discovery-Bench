"""Run bookkeeping: event traces, incremental CSV, checkpoints, aggregation."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class TraceWriter:
    """Append-only JSONL event log; every LLM call and every step lands here."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"ts": utc_now(), "event_type": event_type, "payload": payload}
        self._fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True, default=_jsonable) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


class CsvSink:
    """Row-at-a-time CSV writer with a fixed header (missing keys become empty)."""

    def __init__(self, path: Path, header: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._header = header
        self._exists = path.exists()
        if self._exists:
            self._migrate()

    def _migrate(self) -> None:
        """Rewrite an existing file whose header predates the current one.

        Resuming a run after new columns were added used to append wide rows under a
        narrow header, producing a file no CSV reader will parse. Any column the old
        file has and the new header does not is kept on the end, so a migration never
        loses data.
        """
        with self._path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            existing = list(reader.fieldnames or [])
            if existing == self._header:
                return
            rows = list(reader)
        merged = self._header + [c for c in existing if c not in self._header]
        self._header = merged
        with self._path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=merged)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in merged})

    def write(self, row: dict[str, Any]) -> None:
        clean = {key: _cell(row.get(key, "")) for key in self._header}
        with self._path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._header)
            if not self._exists:
                writer.writeheader()
                self._exists = True
            writer.writerow(clean)


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return json.dumps(value, separators=(",", ":"), default=_jsonable)
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_jsonable), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def load_checkpoint(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    completed = payload.get("completed", {})
    return {str(k): str(v) for k, v in completed.items()}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def aggregate(
    rows: list[dict[str, str]],
    group_keys: Iterable[str],
    metrics: Iterable[str],
    out_csv: Path,
) -> None:
    """Mean / sd / 95% CI per group, plus success counts. One row per group."""
    group_keys = list(group_keys)
    metrics = list(metrics)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    header = list(group_keys) + ["n_total", "n_success", "n_failed"]
    for metric in metrics:
        header += [f"{metric}_mean", f"{metric}_sd", f"{metric}_ci95_lo", f"{metric}_ci95_hi"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for key, group in sorted(grouped.items()):
            success = [r for r in group if r.get("status") == "success"]
            out: dict[str, Any] = dict(zip(group_keys, key))
            out["n_total"] = len(group)
            out["n_success"] = len(success)
            out["n_failed"] = len(group) - len(success)
            for metric in metrics:
                values = [v for v in (_as_float(r.get(metric)) for r in success) if v is not None]
                if not values:
                    out[f"{metric}_mean"] = ""
                    out[f"{metric}_sd"] = ""
                    out[f"{metric}_ci95_lo"] = ""
                    out[f"{metric}_ci95_hi"] = ""
                    continue
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
                out[f"{metric}_mean"] = round(mean, 6)
                out[f"{metric}_sd"] = round(sd, 6)
                out[f"{metric}_ci95_lo"] = round(mean - half, 6)
                out[f"{metric}_ci95_hi"] = round(mean + half, 6)
            writer.writerow(out)
