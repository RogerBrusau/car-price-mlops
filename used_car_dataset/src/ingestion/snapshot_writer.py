from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _utc_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"


@dataclass
class SnapshotWriter:
    out_base: Path
    source: str
    scrape_date: str
    rows_format: str = "parquet"  # parquet|csv|jsonl.gz
    flush_every: int = 200
    run_id: str = ""

    _buffer: List[Dict[str, Any]] = None
    _part: int = 0
    _parquet_ok: bool = False

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = _utc_run_id()
        self._buffer = []
        self.rows_format = (self.rows_format or "parquet").lower()

        if self.rows_format == "parquet":
            try:
                import pyarrow  # noqa: F401
                import pyarrow.parquet  # noqa: F401
                self._parquet_ok = True
            except Exception:
                self._parquet_ok = False

    def add(self, row: Dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= max(1, int(self.flush_every)):
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self._buffer:
            return None

        part_dir = self.out_base / f"source={self.source}" / f"scrape_date={self.scrape_date}"
        _ensure_dir(part_dir)

        self._part += 1
        buf = self._buffer
        self._buffer = []

        if self.rows_format == "parquet" and self._parquet_ok:
            import pyarrow as pa
            import pyarrow.parquet as pq

            out_path = part_dir / f"snapshots_{self.run_id}_{self._part:05d}.parquet"
            table = pa.Table.from_pylist(buf)
            pq.write_table(table, out_path)
            return out_path

        if self.rows_format == "jsonl.gz":
            out_path = part_dir / f"snapshots_{self.run_id}_{self._part:05d}.jsonl.gz"
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                for r in buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return out_path

        # csv fallback (default cuando parquet no está disponible)
        out_path = part_dir / f"snapshots_{self.run_id}_{self._part:05d}.csv"
        fieldnames = sorted({k for r in buf for k in r.keys()})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(buf)
        return out_path

    def close(self) -> None:
        self.flush()
