from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import pandas as pd
import yaml
from lxml import etree


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_url(url: str) -> str:
    """
    Canonicalización conservadora:
    normaliza esquema y host, elimina fragmento, mantiene query.
    """
    u = urlparse(url)
    scheme = (u.scheme or "https").lower()
    netloc = u.netloc.lower()
    path = u.path.rstrip("/")
    u = u._replace(scheme=scheme, netloc=netloc, path=path, fragment="")
    return urlunparse((u.scheme, u.netloc, u.path, u.params, u.query, ""))


def sha1_hex(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class HttpConfig:
    user_agent: str
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    backoff_max_s: float


@dataclass
class PolitenessConfig:
    rate_limit_seconds: float
    respect_robots_txt: bool
    robots_cache_ttl_hours: int


@dataclass
class DiscoveryConfig:
    type: str  # "sitemap" | "pagination"

    sitemap_urls: list[str] | None = None
    max_urls: int = 200000

    # nuevos para debug y control
    max_sitemaps: int = 2000
    progress_every: int = 1000
    verbose: bool = True

    # filtros de sitemaps hijos (nuevo)
    sitemap_include_regex: list[str] | None = None
    sitemap_exclude_regex: list[str] | None = None

    # filtros de URLs finales
    url_include_regex: list[str] | None = None
    url_exclude_regex: list[str] | None = None

    listing_key_regex: str | None = None

    # pagination (opcional)
    list_url_template: str | None = None
    start_page: int | None = None
    end_page: int | None = None
    listing_link_xpath: str | None = None
    next_page_xpath: str | None = None
    stop_if_no_new_urls: bool = True


@dataclass
class SourceConfig:
    name: str
    base_url: str
    http: HttpConfig
    politeness: PolitenessConfig
    discovery: DiscoveryConfig


class RateLimiter:
    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last = 0.0

    def wait(self):
        dt = time.time() - self._last
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last = time.time()


class RobotsRules:
    """
    Parser mínimo que soporta User-agent: * con Allow/Disallow y patrones con * y $.
    Regla: match más largo gana; si empate, Allow gana.
    """
    def __init__(self):
        self.rules: list[tuple[bool, re.Pattern, int, str]] = []

    @staticmethod
    def _pat_to_regex(pat: str) -> re.Pattern:
        pat_esc = re.escape(pat)
        pat_esc = pat_esc.replace(r"\*", ".*")
        if pat.endswith("$"):
            pat_esc = pat_esc[:-2] + "$"
        else:
            pat_esc = "^" + pat_esc
        return re.compile(pat_esc)

    def add(self, allow: bool, pat: str):
        pat = pat.strip()
        if pat == "":
            return
        rx = self._pat_to_regex(pat)
        self.rules.append((allow, rx, len(pat), pat))

    def allowed(self, path_with_query: str) -> bool:
        best = None  # (raw_len, allow)
        for allow, rx, raw_len, _ in self.rules:
            if rx.search(path_with_query):
                cand = (raw_len, allow)
                if best is None:
                    best = cand
                else:
                    if cand[0] > best[0]:
                        best = cand
                    elif cand[0] == best[0] and cand[1] is True and best[1] is False:
                        best = cand
        if best is None:
            return True
        return bool(best[1])


def load_source_config(path: str) -> SourceConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    http = d["http"]
    pol = d["politeness"]
    disc = d["discovery"]

    return SourceConfig(
        name=d["name"],
        base_url=d["base_url"].rstrip("/"),
        http=HttpConfig(
            user_agent=http["user_agent"],
            timeout_s=float(http.get("timeout_s", 25)),
            max_retries=int(http.get("max_retries", 5)),
            backoff_base_s=float(http.get("backoff_base_s", 1.0)),
            backoff_max_s=float(http.get("backoff_max_s", 30.0)),
        ),
        politeness=PolitenessConfig(
            rate_limit_seconds=float(pol.get("rate_limit_seconds", 10)),
            respect_robots_txt=bool(pol.get("respect_robots_txt", True)),
            robots_cache_ttl_hours=int(pol.get("robots_cache_ttl_hours", 24)),
        ),
        discovery=DiscoveryConfig(
            type=disc["type"],
            sitemap_urls=disc.get("sitemap_urls"),
            max_urls=int(disc.get("max_urls", 200000)),

            max_sitemaps=int(disc.get("max_sitemaps", 2000)),
            progress_every=int(disc.get("progress_every", 1000)),
            verbose=bool(disc.get("verbose", True)),

            sitemap_include_regex=disc.get("sitemap_include_regex"),
            sitemap_exclude_regex=disc.get("sitemap_exclude_regex"),

            url_include_regex=disc.get("url_include_regex"),
            url_exclude_regex=disc.get("url_exclude_regex"),

            listing_key_regex=disc.get("listing_key_regex"),

            list_url_template=disc.get("list_url_template"),
            start_page=disc.get("start_page"),
            end_page=disc.get("end_page"),
            listing_link_xpath=disc.get("listing_link_xpath"),
            next_page_xpath=disc.get("next_page_xpath"),
            stop_if_no_new_urls=bool(disc.get("stop_if_no_new_urls", True)),
        ),
    )


def http_get_with_retries(
    client: httpx.Client,
    rl: RateLimiter,
    url: str,
    http_cfg: HttpConfig,
) -> httpx.Response:
    backoff = http_cfg.backoff_base_s
    last_exc = None
    for _attempt in range(1, http_cfg.max_retries + 1):
        try:
            rl.wait()
            r = client.get(url, timeout=http_cfg.timeout_s, follow_redirects=True)

            if r.status_code in (429, 503):
                sleep_s = min(http_cfg.backoff_max_s, backoff) + random.random() * 0.25
                time.sleep(sleep_s)
                backoff *= 2
                continue

            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            sleep_s = min(http_cfg.backoff_max_s, backoff) + random.random() * 0.25
            time.sleep(sleep_s)
            backoff *= 2

    raise RuntimeError(f"Failed GET {url} after retries. Last error: {last_exc}")


def robots_cache_path(cache_dir: Path, source_name: str, base_url: str) -> Path:
    host = urlparse(base_url).netloc
    return cache_dir / f"robots_{source_name}_{sha1_hex(host)}.json"


def fetch_robots_rules(
    client: httpx.Client,
    rl: RateLimiter,
    cfg: SourceConfig,
    cache_dir: Path,
) -> RobotsRules:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = robots_cache_path(cache_dir, cfg.name, cfg.base_url)

    now_ts = time.time()
    ttl_s = cfg.politeness.robots_cache_ttl_hours * 3600

    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if now_ts - payload.get("fetched_ts", 0) < ttl_s:
            rr = RobotsRules()
            for item in payload.get("rules", []):
                rr.add(bool(item["allow"]), item["pattern"])
            return rr

    robots_url = urljoin(cfg.base_url + "/", "robots.txt")
    r = http_get_with_retries(client, rl, robots_url, cfg.http)
    txt = r.text

    rr = RobotsRules()

    active = False
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "user-agent":
            active = (val == "*")
        elif active and key in ("allow", "disallow"):
            rr.add(key == "allow", val)

    cache_file.write_text(
        json.dumps(
            {
                "fetched_ts": now_ts,
                "rules": [{"allow": a, "pattern": p} for (a, _, _, p) in rr.rules],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rr


def compile_regex_list(pats: Optional[list[str]]) -> list[re.Pattern]:
    if not pats:
        return []
    return [re.compile(p) for p in pats]


def url_allowed_by_filters(url: str, include: list[re.Pattern], exclude: list[re.Pattern]) -> bool:
    if include and not any(rx.search(url) for rx in include):
        return False
    if exclude and any(rx.search(url) for rx in exclude):
        return False
    return True


def iter_sitemap_locs(xml_bytes: bytes) -> Iterable[str]:
    """
    Extrae todos los <loc> sin depender del namespace.
    Sirve tanto para urlset como para sitemapindex.
    """
    root = etree.fromstring(xml_bytes)
    locs = root.xpath("//*[local-name()='loc']/text()")
    for loc in locs:
        loc = (loc or "").strip()
        if loc:
            yield loc


def fetch_bytes_maybe_gzip(r: httpx.Response) -> bytes:
    b = r.content
    if r.url.path.endswith(".gz"):
        try:
            return gzip.decompress(b)
        except Exception:
            return b
    return b


def discover_from_sitemaps(
    client: httpx.Client,
    rl: RateLimiter,
    cfg: SourceConfig,
    robots: RobotsRules,
) -> list[dict]:
    disc = cfg.discovery

    # filtros para URLs finales
    include = compile_regex_list(disc.url_include_regex)
    exclude = compile_regex_list(disc.url_exclude_regex)

    # filtros para sitemaps hijos (nuevo)
    sm_include = compile_regex_list(disc.sitemap_include_regex)
    sm_exclude = compile_regex_list(disc.sitemap_exclude_regex)

    seen_urls: set[str] = set()
    out: list[dict] = []

    queue = list(disc.sitemap_urls or [])
    visited_sitemaps: set[str] = set()
    sitemaps_fetched = 0

    while queue and len(out) < disc.max_urls and sitemaps_fetched < disc.max_sitemaps:
        sm_url = queue.pop(0)
        if sm_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sm_url)

        sm_url = canonicalize_url(sm_url)

        if cfg.politeness.respect_robots_txt:
            pathq = urlparse(sm_url).path
            if not robots.allowed(pathq):
                if disc.verbose:
                    print(f"[{cfg.name}] robots bloquea sitemap: {sm_url}")
                continue

        if disc.verbose:
            print(f"[{cfg.name}] sitemap {sitemaps_fetched+1}/{disc.max_sitemaps} queue={len(queue)} out={len(out)} fetch={sm_url}")

        r = http_get_with_retries(client, rl, sm_url, cfg.http)
        sitemaps_fetched += 1

        xml = fetch_bytes_maybe_gzip(r)

        for loc in iter_sitemap_locs(xml):
            loc = canonicalize_url(loc)

            is_sitemap = loc.endswith(".xml") or loc.endswith(".xml.gz")
            if is_sitemap:
                # nuevo: filtrado de sitemaps hijos
                if url_allowed_by_filters(loc, sm_include, sm_exclude):
                    if loc not in visited_sitemaps:
                        queue.append(loc)
                continue

            if loc in seen_urls:
                continue
            seen_urls.add(loc)

            if not url_allowed_by_filters(loc, include, exclude):
                continue

            if cfg.politeness.respect_robots_txt:
                u = urlparse(loc)
                pathq = u.path + (("?" + u.query) if u.query else "")
                if not robots.allowed(pathq):
                    continue

            listing_key = None
            if disc.listing_key_regex:
                m = re.search(disc.listing_key_regex, loc)
                if m:
                    listing_key = m.group(1)

            out.append(
                {
                    "source": cfg.name,
                    "discovered_datetime": utc_now_iso(),
                    "discovered_from": sm_url,
                    "url": loc,
                    "url_canonical": loc,
                    "source_listing_key": listing_key,
                }
            )

            if disc.verbose and disc.progress_every > 0 and (len(out) % disc.progress_every == 0):
                print(f"[{cfg.name}] discovered={len(out)}")

            if len(out) >= disc.max_urls:
                break

    if disc.verbose:
        print(f"[{cfg.name}] finished: discovered={len(out)} sitemaps_fetched={sitemaps_fetched} sitemaps_visited={len(visited_sitemaps)}")

    return out


def discover_from_pagination(
    client: httpx.Client,
    rl: RateLimiter,
    cfg: SourceConfig,
    robots: RobotsRules,
) -> list[dict]:
    disc = cfg.discovery
    if not disc.list_url_template or disc.start_page is None or disc.end_page is None or not disc.listing_link_xpath:
        raise ValueError("Pagination discovery requiere list_url_template, start_page, end_page, listing_link_xpath")

    include = compile_regex_list(disc.url_include_regex)
    exclude = compile_regex_list(disc.url_exclude_regex)

    seen: set[str] = set()
    out: list[dict] = []

    for page in range(disc.start_page, disc.end_page + 1):
        list_url = disc.list_url_template.format(page=page)
        list_url = canonicalize_url(urljoin(cfg.base_url + "/", list_url))

        if disc.verbose:
            print(f"[{cfg.name}] page={page} fetch={list_url} out={len(out)}")

        if cfg.politeness.respect_robots_txt:
            u = urlparse(list_url)
            pathq = u.path + (("?" + u.query) if u.query else "")
            if not robots.allowed(pathq):
                if disc.verbose:
                    print(f"[{cfg.name}] robots bloquea listado, paro: {list_url}")
                break

        r = http_get_with_retries(client, rl, list_url, cfg.http)
        tree = etree.HTML(r.content)

        hrefs = tree.xpath(disc.listing_link_xpath)
        page_urls = 0

        for href in hrefs:
            if not href:
                continue
            abs_url = canonicalize_url(urljoin(cfg.base_url + "/", str(href)))
            if abs_url in seen:
                continue
            seen.add(abs_url)

            if not url_allowed_by_filters(abs_url, include, exclude):
                continue

            if cfg.politeness.respect_robots_txt:
                u2 = urlparse(abs_url)
                pathq2 = u2.path + (("?" + u2.query) if u2.query else "")
                if not robots.allowed(pathq2):
                    continue

            listing_key = None
            if disc.listing_key_regex:
                m = re.search(disc.listing_key_regex, abs_url)
                if m:
                    listing_key = m.group(1)

            out.append(
                {
                    "source": cfg.name,
                    "discovered_datetime": utc_now_iso(),
                    "discovered_from": list_url,
                    "url": abs_url,
                    "url_canonical": abs_url,
                    "source_listing_key": listing_key,
                }
            )
            page_urls += 1

            if disc.verbose and disc.progress_every > 0 and (len(out) % disc.progress_every == 0):
                print(f"[{cfg.name}] discovered={len(out)}")

            if len(out) >= disc.max_urls:
                break

        if disc.stop_if_no_new_urls and page_urls == 0:
            if disc.verbose:
                print(f"[{cfg.name}] sin nuevas urls en page={page}, paro")
            break

        if len(out) >= disc.max_urls:
            break

    if disc.verbose:
        print(f"[{cfg.name}] finished: discovered={len(out)}")

    return out


def write_discovery_output(rows: list[dict], out_dir: Path, source: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    scrape_date = datetime.now(timezone.utc).date().isoformat()

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "source",
                "discovered_datetime",
                "discovered_from",
                "url",
                "url_canonical",
                "source_listing_key",
            ]
        )

    part_dir = out_dir / f"source={source}" / f"scrape_date={scrape_date}"
    part_dir.mkdir(parents=True, exist_ok=True)
    out_path = part_dir / "discovered_urls.parquet"
    df.to_parquet(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta al YAML de la fuente")
    ap.add_argument("--out", required=True, help="Directorio de salida (bronze/discovery)")
    ap.add_argument("--robots-cache", default=".cache/robots", help="Directorio cache robots")
    args = ap.parse_args()

    cfg = load_source_config(args.config)
    rl = RateLimiter(cfg.politeness.rate_limit_seconds)

    headers = {
        "User-Agent": cfg.http.user_agent,
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    with httpx.Client(headers=headers) as client:
        robots = RobotsRules()
        if cfg.politeness.respect_robots_txt:
            robots = fetch_robots_rules(client, rl, cfg, Path(args.robots_cache))

        if cfg.discovery.type == "sitemap":
            rows = discover_from_sitemaps(client, rl, cfg, robots)
        elif cfg.discovery.type == "pagination":
            rows = discover_from_pagination(client, rl, cfg, robots)
        else:
            raise ValueError(f"Unknown discovery.type: {cfg.discovery.type}")

    write_discovery_output(rows, Path(args.out), cfg.name)


if __name__ == "__main__":
    main()
