# src/ingestion/scheduler.py
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def find_latest_discovery_parquet(discovery_root: Path, source: str) -> Path:
    root = discovery_root / f"source={source}"
    parts = sorted(root.glob("scrape_date=*/discovered_urls.parquet"))
    if not parts:
        raise FileNotFoundError(f"No discovery parquet found under {root}")
    return parts[-1]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS listing_registry (
          listing_id TEXT PRIMARY KEY,
          source TEXT NOT NULL,
          source_listing_key TEXT NOT NULL,
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          canonical_url TEXT,
          UNIQUE(source, source_listing_key)
        );

        CREATE TABLE IF NOT EXISTS url_queue (
          listing_id TEXT PRIMARY KEY,
          source TEXT NOT NULL,
          url TEXT NOT NULL,
          next_fetch_at TEXT NOT NULL,
          priority INTEGER NOT NULL DEFAULT 0,
          tries INTEGER NOT NULL DEFAULT 0,
          last_status INTEGER,
          last_fetch_at TEXT,
          last_error TEXT,
          locked_at TEXT,
          lock_token TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_url_queue_next
        ON url_queue(next_fetch_at, priority);

        CREATE TABLE IF NOT EXISTS listing_snapshots (
          snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
          listing_id TEXT NOT NULL,
          source TEXT NOT NULL,
          scrape_datetime TEXT NOT NULL,
          status_code INTEGER,
          final_url TEXT,
          bytes INTEGER,
          html_sha256 TEXT,
          stored_path TEXT,
          content_type TEXT,
          error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_listing
        ON listing_snapshots(listing_id, scrape_datetime);
        """
    )
    conn.commit()


def chunked(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


@dataclass(frozen=True)
class SchedulerConfig:
    ttl_hours: int
    priority: int
    max_per_run: int
    dead_status_cooldown_days: int


def load_scheduler_config(cfg: dict) -> SchedulerConfig:
    q = cfg.get("queue", {}) if isinstance(cfg, dict) else {}
    return SchedulerConfig(
        ttl_hours=int(q.get("fetch_ttl_hours", 24)),
        priority=int(q.get("priority_default", 0)),
        max_per_run=int(q.get("max_per_run", 0)),
        dead_status_cooldown_days=int(q.get("dead_status_cooldown_days", 30)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML source config")
    ap.add_argument("--discovery-root", required=True, help="data/bronze/discovery")
    ap.add_argument("--db", required=True, help="SQLite path, e.g. data/bronze/queue/queue.sqlite")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = cfg["source"]
    scfg = load_scheduler_config(cfg)

    discovery_root = Path(args.discovery_root)
    parquet_path = find_latest_discovery_parquet(discovery_root, source)

    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["source", "source_listing_key", "url"])
    df = df.drop_duplicates(subset=["source", "source_listing_key"], keep="last")

    now_iso = utcnow_iso()
    now_dt = datetime.fromisoformat(now_iso)

    df["listing_id"] = df.apply(
        lambda r: sha1_hex(f"{r['source']}:{r['source_listing_key']}"),
        axis=1,
    )

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    init_db(conn)

    with conn:
        conn.executemany(
            """
            INSERT INTO listing_registry(listing_id, source, source_listing_key, first_seen_at, last_seen_at, canonical_url)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(listing_id) DO UPDATE SET
              last_seen_at=excluded.last_seen_at,
              canonical_url=COALESCE(excluded.canonical_url, listing_registry.canonical_url)
            """,
            [
                (
                    row["listing_id"],
                    row["source"],
                    row["source_listing_key"],
                    now_iso,
                    now_iso,
                    row.get("url_canonical", None),
                )
                for _, row in df.iterrows()
            ],
        )

    listing_ids = df["listing_id"].astype(str).tolist()

    existing: dict[str, sqlite3.Row] = {}
    for ch in chunked(listing_ids, 900):
        qs = ",".join(["?"] * len(ch))
        rows = conn.execute(
            f"""
            SELECT listing_id, last_fetch_at, last_status
            FROM url_queue
            WHERE listing_id IN ({qs})
            """,
            ch,
        ).fetchall()
        for r in rows:
            existing[r["listing_id"]] = r

    ttl = timedelta(hours=scfg.ttl_hours)
    dead_cooldown = timedelta(days=scfg.dead_status_cooldown_days)
    dead_status = {404, 410}

    to_enqueue: list[tuple[str, str, str, str, int]] = []
    for _, row in df.iterrows():
        lid = row["listing_id"]
        url = row["url"]
        ex = existing.get(lid)

        should = False
        if ex is None:
            should = True
        else:
            lf = ex["last_fetch_at"]
            ls = ex["last_status"]
            if lf is None:
                should = True
            else:
                lf_dt = datetime.fromisoformat(lf)
                expired = lf_dt <= (now_dt - ttl)
                if expired:
                    if ls in dead_status and lf_dt > (now_dt - dead_cooldown):
                        should = False
                    else:
                        should = True

        if should:
            to_enqueue.append((lid, source, url, now_iso, scfg.priority))

    if scfg.max_per_run and scfg.max_per_run > 0:
        to_enqueue = to_enqueue[: scfg.max_per_run]

    with conn:
        conn.executemany(
            """
            INSERT INTO url_queue(listing_id, source, url, next_fetch_at, priority)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(listing_id) DO UPDATE SET
              url=excluded.url,
              next_fetch_at=excluded.next_fetch_at,
              priority=excluded.priority
            """,
            to_enqueue,
        )

    print(f"discovery_parquet={parquet_path}")
    print(f"registry_upserted={len(df)}")
    print(f"enqueued={len(to_enqueue)}")
    conn.close()


if __name__ == "__main__":
    main()
