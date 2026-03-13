# src/ingestion/fetch_detail.py
from __future__ import annotations

import json
import argparse
import gzip
import hashlib
import os
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import importlib
import re

import httpx
import yaml

from src.ingestion.url_identity import canonicalize_url, listing_id_from, deterministic_sample
from src.ingestion.snapshot_writer import SnapshotWriter


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class FetchConfig:
    user_agent: str
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    rate_limit_rps: float
    refresh_hours_ok: int
    retry_days_404: int
    lock_timeout_minutes: int
    max_bytes: int
    robots_cache_dir: Path
    robots_ttl_hours: int


def load_fetch_config(cfg: dict) -> FetchConfig:
    robots = cfg.get("robots", {}) if isinstance(cfg, dict) else {}
    f = cfg.get("fetch_detail", {}) if isinstance(cfg, dict) else {}

    return FetchConfig(
        user_agent=str(robots.get("user_agent", "projecte-cotxes-bot/1.0")),
        timeout_s=float(f.get("timeout_s", 20)),
        max_retries=int(f.get("max_retries", 3)),
        backoff_base_s=float(f.get("backoff_base_s", 2.0)),
        rate_limit_rps=float(f.get("rate_limit_rps", 0.5)),
        refresh_hours_ok=int(f.get("refresh_interval_hours", 24)),
        retry_days_404=int(f.get("retry_days_404", 14)),
        lock_timeout_minutes=int(f.get("lock_timeout_minutes", 30)),
        max_bytes=int(f.get("max_bytes", 5_000_000)),
        robots_cache_dir=Path(robots.get("cache_dir", ".cache/robots")),
        robots_ttl_hours=int(robots.get("cache_ttl_hours", 24)),
    )


class HostRateLimiter:
    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / max(rps, 1e-9)
        self.next_time: dict[str, float] = {}

    def wait(self, host: str) -> None:
        now = time.monotonic()
        t = self.next_time.get(host, now)
        if t > now:
            time.sleep(t - now)
        self.next_time[host] = time.monotonic() + self.min_interval


class RobotsCache:
    def __init__(self, cache_dir: Path, ttl: timedelta) -> None:
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, netloc: str) -> Path:
        safe = netloc.replace(":", "_")
        return self.cache_dir / f"{safe}.robots.txt"

    def get(self, client: httpx.Client, base_url: str, user_agent: str) -> RobotFileParser:
        u = urlparse(base_url)
        netloc = u.netloc
        rp = RobotFileParser()
        rp.set_url(f"{u.scheme}://{netloc}/robots.txt")

        p = self._cache_path(netloc)
        now = utcnow()

        use_cache = False
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if now - mtime <= self.ttl:
                use_cache = True

        if use_cache:
            txt = p.read_text(encoding="utf-8", errors="replace")
        else:
            headers = {"User-Agent": user_agent}
            try:
                r = client.get(rp.url, headers=headers, timeout=10)
                txt = r.text if r.status_code == 200 else ""
            except Exception:
                txt = ""
            p.write_text(txt, encoding="utf-8")

        rp.parse(txt.splitlines())
        return rp


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(
        """
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


def claim_batch(conn: sqlite3.Connection, batch_size: int, lock_timeout: timedelta) -> list[sqlite3.Row]:
    token = uuid.uuid4().hex
    now = utcnow()
    now_iso = now.isoformat()
    stale_before = (now - lock_timeout).isoformat()

    with conn:
        rows = conn.execute(
            """
            SELECT listing_id
            FROM url_queue
            WHERE next_fetch_at <= ?
              AND (locked_at IS NULL OR locked_at <= ?)
            ORDER BY priority DESC, next_fetch_at ASC
            LIMIT ?
            """,
            (now_iso, stale_before, batch_size),
        ).fetchall()

        if not rows:
            return []

        ids = [r["listing_id"] for r in rows]
        qs = ",".join(["?"] * len(ids))
        conn.execute(
            f"""
            UPDATE url_queue
            SET locked_at = ?, lock_token = ?
            WHERE listing_id IN ({qs})
            """,
            [now_iso, token, *ids],
        )

        claimed = conn.execute(
            """
            SELECT listing_id, source, url, tries, priority
            FROM url_queue
            WHERE lock_token = ?
            """,
            (token,),
        ).fetchall()

    return claimed


def write_gzip(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz:
            gz.write(content)


def compute_backoff_seconds(base: float, tries: int) -> float:
    exp = min(tries, 8)
    jitter = random.uniform(0.85, 1.15)
    return base * (2**exp) * jitter


_NUM_RX = re.compile(r"(\d[\d\.\s]*\d|\d)")


def _to_int(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x)
    m = _NUM_RX.search(s.replace("\xa0", " "))
    if not m:
        return None
    val = m.group(1).replace(".", "").replace(" ", "")
    try:
        return int(val)
    except Exception:
        return None


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("\xa0", " ")
    m = _NUM_RX.search(s)
    if not m:
        return None
    raw = m.group(1).replace(" ", "").replace(".", "")
    try:
        return float(raw)
    except Exception:
        return None


def _extract_jsonld_objects(html_text: str):
    objs = []
    for m in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
        re.DOTALL | re.IGNORECASE,
    ):
        blob = m.group(1).strip()
        if not blob:
            continue
        try:
            data = json.loads(blob)
            if isinstance(data, list):
                objs.extend([x for x in data if isinstance(x, (dict, list))])
            elif isinstance(data, dict):
                objs.append(data)
        except Exception:
            continue
    return objs


def _walk_json(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk_json(v)
    elif isinstance(x, list):
        for it in x:
            yield from _walk_json(it)


def _pick_vehicle_like(objs):
    best = None
    for o in objs:
        for d in _walk_json(o):
            t = d.get("@type")
            if isinstance(t, list):
                t = " ".join([str(z) for z in t])
            t = (str(t) if t is not None else "").lower()
            if "vehicle" in t or "car" in t or "product" in t:
                best = d
                if "vehicle" in t:
                    return best
    return best


def _get_name_field(x):
    if isinstance(x, dict):
        return x.get("name") or x.get("@id") or None
    return None


def parse_min_fields_generic(html_bytes: bytes, final_url: str):
    html_text = (html_bytes or b"").decode("utf-8", errors="ignore")
    out = {}

    objs = _extract_jsonld_objects(html_text)
    cand = _pick_vehicle_like(objs) if objs else None

    if cand:
        offers = cand.get("offers")
        if isinstance(offers, list):
            offers = offers[0] if offers else None
        if isinstance(offers, dict):
            out["price_eur"] = _to_int(
                offers.get("price") or offers.get("lowPrice") or offers.get("highPrice")
            )

        brand = cand.get("brand")
        out["make"] = _get_name_field(brand) if brand else None
        out["model"] = cand.get("model") or None
        out["trim_raw"] = cand.get("vehicleConfiguration") or None

        out["first_registration_date"] = (
            cand.get("dateVehicleFirstRegistered") or cand.get("productionDate") or None
        )
        if not out["first_registration_date"]:
            y = cand.get("vehicleModelDate")
            if y:
                out["first_registration_date"] = str(y)

        odo = cand.get("mileageFromOdometer")
        if isinstance(odo, dict):
            out["mileage_km"] = _to_int(odo.get("value"))
        else:
            out["mileage_km"] = _to_int(odo)

        out["fuel"] = cand.get("fuelType") or None
        out["transmission"] = cand.get("vehicleTransmission") or None

        eng = cand.get("vehicleEngine")
        power_cv = None
        if isinstance(eng, dict):
            p = eng.get("enginePower")
            if isinstance(p, dict):
                val = _to_float(p.get("value"))
                unit = str(p.get("unitText") or p.get("unitCode") or "").lower()
                if val is not None:
                    if "kw" in unit:
                        power_cv = int(round(val * 1.35962))
                    else:
                        power_cv = int(round(val))
            else:
                power_cv = _to_int(p)
        out["power_cv"] = power_cv

        addr = cand.get("address") or None
        if isinstance(addr, dict):
            out["province"] = addr.get("addressRegion") or addr.get("addressLocality") or None

    if out.get("price_eur") is None:
        m = re.search(r"(\d[\d\.\s]{3,})\s*€", html_text)
        if m:
            out["price_eur"] = _to_int(m.group(1))

    if out.get("mileage_km") is None:
        m = re.search(r"(\d[\d\.\s]{3,})\s*km", html_text, re.IGNORECASE)
        if m:
            out["mileage_km"] = _to_int(m.group(1))

    if out.get("power_cv") is None:
        m = re.search(r"(\d{2,4})\s*cv", html_text, re.IGNORECASE)
        if m:
            out["power_cv"] = _to_int(m.group(1))

    if isinstance(out.get("make"), str):
        out["make"] = out["make"].strip()
    if isinstance(out.get("model"), str):
        out["model"] = out["model"].strip()

    return out


def load_source_parser(source: str):
    for modname in (f"src.parsers.{source}", f"src.ingestion.parsers.{source}"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "parse_listing"):
                return mod, getattr(mod, "__version__", None), mod.parse_listing
            if hasattr(mod, "parse"):
                return mod, getattr(mod, "__version__", None), mod.parse
        except Exception:
            continue
    return None, None, None


def cleanup_raw_by_mtime(out_root: Path, retention_days: int):
    if retention_days is None:
        return
    cutoff = datetime.now(timezone.utc).timestamp() - retention_days * 86400
    for p in out_root.rglob("*.gz"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        except Exception:
            continue


def _call_parser(parser_fn, body: bytes, final_url: str, cfg: dict) -> dict:
    if parser_fn is None:
        return {}
    try:
        out = parser_fn(body, url=final_url, config=cfg)
        return out if isinstance(out, dict) else {}
    except TypeError:
        pass
    try:
        out = parser_fn(body, url=final_url)
        return out if isinstance(out, dict) else {}
    except TypeError:
        pass
    try:
        out = parser_fn(body, cfg)
        return out if isinstance(out, dict) else {}
    except TypeError:
        pass
    out = parser_fn(body)
    return out if isinstance(out, dict) else {}


def _should_store_raw(
    store_raw: str,
    sample_rate: float,
    stable_listing_id: str,
    scrape_date: str,
    parse_ok: bool,
    http_ok: bool,
    any_error: Optional[str],
) -> bool:
    store_raw = (store_raw or "all").lower()

    if store_raw == "none":
        return False
    if store_raw == "all":
        return True
    if store_raw == "sample":
        return deterministic_sample(stable_listing_id, scrape_date, float(sample_rate or 0.0))
    if store_raw == "errors":
        # guarda si hay error de fetch, status != 200, o parse_ok false
        return (not http_ok) or (not parse_ok) or (any_error is not None)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML source config")
    ap.add_argument("--db", required=True, help="SQLite path")
    ap.add_argument("--out-root", required=True, help="data/bronze/raw_html")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=0, help="0 = ilimitado en esta ejecución")

    ap.add_argument(
        "--out-rows",
        type=str,
        default="data/silver/listings",
        help="Directorio base para snapshots silver (partitioned por source/scrape_date).",
    )
    ap.add_argument(
        "--rows-format",
        type=str,
        default="parquet",
        choices=["parquet", "csv", "jsonl.gz"],
        help="Formato para snapshots: parquet (si hay pyarrow), si no cae a csv.",
    )
    ap.add_argument(
        "--store-raw",
        type=str,
        default="errors",
        choices=["none", "errors", "sample", "all"],
        help="Política de guardado de raw HTML.",
    )
    ap.add_argument(
        "--sample-rate",
        type=float,
        default=0.0,
        help="Probabilidad en [0,1] para store-raw=sample (muestreo determinista).",
    )
    ap.add_argument(
        "--raw-retention-days",
        type=int,
        default=None,
        help="Si se define, borra raws con mtime más viejo que N días.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = cfg.get("source") or cfg.get("name")
    if not source:
        raise ValueError("YAML debe incluir 'source' o 'name'")
    fcfg = load_fetch_config(cfg)

    db_path = Path(args.db)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    init_db(conn)

    out_root = Path(args.out_root)

    limiter = HostRateLimiter(fcfg.rate_limit_rps)
    robots_cache = RobotsCache(fcfg.robots_cache_dir, ttl=timedelta(hours=fcfg.robots_ttl_hours))

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    client = httpx.Client(follow_redirects=True, timeout=fcfg.timeout_s, limits=limits)

    parser_mod, parser_version, parser_fn = load_source_parser(source)
    if parser_fn is None:
        parser_version = "generic_jsonld_v1"
    else:
        parser_version = parser_version or "custom_parser"

    rows_base = Path(args.out_rows)
    current_writer_date: Optional[str] = None
    writer: Optional[SnapshotWriter] = None

    def ensure_writer(scrape_date: str) -> SnapshotWriter:
        nonlocal writer, current_writer_date
        if writer is None or current_writer_date != scrape_date:
            if writer is not None:
                writer.close()
            writer = SnapshotWriter(
                out_base=rows_base,
                source=source,
                scrape_date=scrape_date,
                rows_format=args.rows_format,
                flush_every=max(50, int(args.batch_size or 50)),
            )
            current_writer_date = scrape_date
        return writer

    processed = 0
    lock_timeout = timedelta(minutes=fcfg.lock_timeout_minutes)

    try:
        while True:
            if args.max_items and processed >= args.max_items:
                break

            batch = claim_batch(conn, args.batch_size, lock_timeout)
            if not batch:
                break

            for row in batch:
                if args.max_items and processed >= args.max_items:
                    break

                queue_listing_id = row["listing_id"]
                url = row["url"]
                tries = int(row["tries"] or 0)

                scrape_dt = utcnow()
                scrape_iso = scrape_dt.isoformat()
                scrape_date = scrape_dt.date().isoformat()

                final_url = None
                status_code = None
                body = b""
                content_type = None
                err = None

                host = urlparse(url).netloc
                limiter.wait(host)

                rp = robots_cache.get(client, base_url=url, user_agent=fcfg.user_agent)
                if not rp.can_fetch(fcfg.user_agent, url):
                    err = "robots_disallow"
                    next_fetch = (scrape_dt + timedelta(days=7)).isoformat()

                    # silver snapshot aunque sea robots
                    canonical_url = canonicalize_url(url)
                    stable_listing_id = queue_listing_id
                    w = ensure_writer(scrape_date)
                    w.add(
                        {
                            "source": source,
                            "url": url,
                            "listing_id": stable_listing_id,
                            "scrape_datetime": scrape_iso,
                            "price_eur": None,
                            "make": None,
                            "model": None,
                            "trim_raw": None,
                            "first_registration_date": None,
                            "year": None,
                            "mileage_km": None,
                            "fuel": None,
                            "transmission": None,
                            "power_cv": None,
                            "province": None,
                            "seller_type": None,
                            "parse_ok": False,
                            "error": err,
                            "parser_version": parser_version,
                            "status_code": None,
                            "final_url": None,
                        }
                    )

                    with conn:
                        conn.execute(
                            """
                            UPDATE url_queue
                            SET last_status = NULL,
                                last_fetch_at = ?,
                                last_error = ?,
                                next_fetch_at = ?,
                                locked_at = NULL,
                                lock_token = NULL
                            WHERE listing_id = ?
                            """,
                            (scrape_iso, err, next_fetch, queue_listing_id),
                        )
                        conn.execute(
                            """
                            INSERT INTO listing_snapshots(listing_id, source, scrape_datetime, status_code, final_url, bytes, html_sha256, stored_path, content_type, error)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (queue_listing_id, source, scrape_iso, None, None, 0, None, None, None, err),
                        )

                    processed += 1
                    continue

                attempt = 0
                while True:
                    attempt += 1
                    try:
                        headers = {
                            "User-Agent": fcfg.user_agent,
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            "Accept-Language": "es-ES,es;q=0.9,en;q=0.7",
                        }
                        r = client.get(url, headers=headers)
                        status_code = int(r.status_code)
                        final_url = str(r.url)
                        content_type = r.headers.get("content-type")

                        raw = r.content or b""
                        if len(raw) > fcfg.max_bytes:
                            raw = raw[: fcfg.max_bytes]
                            err = "body_truncated"

                        body = raw
                        break
                    except Exception as e:
                        err = f"fetch_error:{type(e).__name__}"
                        if attempt >= fcfg.max_retries:
                            break
                        backoff = compute_backoff_seconds(fcfg.backoff_base_s, tries + attempt)
                        time.sleep(backoff)

                html_hash = sha256_hex(body) if body else None
                bytes_len = len(body)

                http_ok = (status_code == 200)
                last_error = None

                if http_ok:
                    next_fetch_at = (scrape_dt + timedelta(hours=fcfg.refresh_hours_ok)).isoformat()
                    new_tries = 0
                    last_error = err if err is not None else None
                elif status_code in (404, 410):
                    next_fetch_at = (scrape_dt + timedelta(days=fcfg.retry_days_404)).isoformat()
                    new_tries = tries + 1
                    last_error = err or f"http_{status_code}"
                elif status_code in (429, 500, 502, 503, 504) or status_code is None:
                    backoff_s = compute_backoff_seconds(fcfg.backoff_base_s, tries + 1)
                    next_fetch_at = (scrape_dt + timedelta(seconds=backoff_s)).isoformat()
                    new_tries = tries + 1
                    last_error = err or (f"http_{status_code}" if status_code else "no_status")
                else:
                    backoff_s = compute_backoff_seconds(fcfg.backoff_base_s, tries + 1)
                    next_fetch_at = (scrape_dt + timedelta(seconds=backoff_s)).isoformat()
                    new_tries = tries + 1
                    last_error = err or f"http_{status_code}"

                # parsing mínimo + snapshot silver siempre
                used_url_for_id = final_url or url
                canonical_url = canonicalize_url(used_url_for_id)
                stable_listing_id = queue_listing_id

                parsed = {}
                parse_err = None

                if http_ok and body:
                    try:
                        if parser_fn is not None:
                            parsed = _call_parser(parser_fn, body, used_url_for_id, cfg)
                        else:
                            parsed = parse_min_fields_generic(body, used_url_for_id)
                    except Exception as e:
                        parse_err = f"parse_error:{type(e).__name__}:{e}"
                else:
                    if status_code is None:
                        parse_err = "no_status_no_parse"
                    elif not http_ok:
                        parse_err = f"http_{status_code}_no_parse"
                    elif not body:
                        parse_err = "empty_body_no_parse"

                price_eur = parsed.get("price_eur")
                parse_ok = bool(http_ok and (parse_err is None) and (price_eur is not None))

                # error final para silver
                silver_error = parse_err or last_error

                w = ensure_writer(scrape_date)
                w.add(
                    {
                        "source": source,
                        "url": used_url_for_id,
                        "listing_id": stable_listing_id,
                        "scrape_datetime": scrape_iso,
                        "price_eur": price_eur,
                        "make": parsed.get("make"),
                        "model": parsed.get("model"),
                        "trim_raw": parsed.get("trim_raw"),
                        "first_registration_date": parsed.get("first_registration_date"),
                        "year": parsed.get("year"),
                        "mileage_km": parsed.get("mileage_km"),
                        "fuel": parsed.get("fuel"),
                        "transmission": parsed.get("transmission"),
                        "power_cv": parsed.get("power_cv"),
                        "province": parsed.get("province"),
                        "seller_type": parsed.get("seller_type"),
                        "parse_ok": parse_ok,
                        "error": silver_error,
                        "parser_version": parser_version,
                        "status_code": status_code,
                        "final_url": final_url,
                    }
                )

                # decidir guardado raw
                save_raw = _should_store_raw(
                    store_raw=args.store_raw,
                    sample_rate=float(args.sample_rate or 0.0),
                    stable_listing_id=stable_listing_id,
                    scrape_date=scrape_date,
                    parse_ok=parse_ok,
                    http_ok=http_ok,
                    any_error=silver_error,
                )

                stored_path = None
                if save_raw and body:
                    rel_dir = Path(f"source={source}") / f"scrape_date={scrape_date}"
                    fname = f"listing_id={stable_listing_id}.html.gz"
                    rel_path = rel_dir / fname
                    abs_path = out_root / rel_path
                    write_gzip(abs_path, body)
                    stored_path = str(rel_path)

                with conn:
                    conn.execute(
                        """
                        UPDATE url_queue
                        SET last_status = ?,
                            last_fetch_at = ?,
                            last_error = ?,
                            next_fetch_at = ?,
                            tries = ?,
                            locked_at = NULL,
                            lock_token = NULL
                        WHERE listing_id = ?
                        """,
                        (status_code, scrape_iso, last_error, next_fetch_at, new_tries, queue_listing_id),
                    )

                    conn.execute(
                        """
                        INSERT INTO listing_snapshots(listing_id, source, scrape_datetime, status_code, final_url, bytes, html_sha256, stored_path, content_type, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            queue_listing_id,
                            source,
                            scrape_iso,
                            status_code,
                            final_url,
                            bytes_len,
                            html_hash,
                            stored_path,
                            content_type,
                            last_error,
                        ),
                    )

                processed += 1

    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass

        if args.raw_retention_days:
            cleanup_raw_by_mtime(out_root, int(args.raw_retention_days))

        client.close()
        conn.close()

    print(f"processed={processed}")


if __name__ == "__main__":
    main()
