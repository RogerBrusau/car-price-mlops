from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


_DROP_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "yclid",
    "mc_cid", "mc_eid",
    "_ga", "_gid", "_gl",
    "gbraid", "wbraid",
    "ref", "referrer", "source", "cmp",
}


def canonicalize_url(url: str) -> str:
    """
    Canoniza URL para estabilidad de listing_id:
    1) lower de scheme/host
    2) sin fragment
    3) elimina query params de tracking (utm_*, gclid, fbclid, etc.)
    4) ordena query restante
    5) normaliza trailing slash en path
    """
    s = urlsplit(url.strip())
    scheme = (s.scheme or "https").lower()
    netloc = s.netloc.lower()

    path = s.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    kept = []
    for k, v in parse_qsl(s.query, keep_blank_values=True):
        kl = k.lower()
        if kl.startswith("utm_"):
            continue
        if kl in _DROP_QUERY_KEYS:
            continue
        kept.append((k, v))

    kept.sort(key=lambda kv: (kv[0], kv[1]))
    query = urlencode(kept, doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def listing_id_from(source: str, canonical_url: str) -> str:
    """
    listing_id estable: sha1("source|canonical_url").
    """
    h = hashlib.sha1()
    h.update(f"{source}|{canonical_url}".encode("utf-8"))
    return h.hexdigest()


def deterministic_sample(listing_id: str, scrape_date: str, sample_rate: float) -> bool:
    """
    Muestreo determinista por (listing_id, scrape_date) para reproducibilidad.
    sample_rate en [0,1].
    """
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    h = hashlib.sha1(f"{listing_id}|{scrape_date}".encode("utf-8")).hexdigest()
    x = int(h, 16) / float(2**160)
    return x < sample_rate
