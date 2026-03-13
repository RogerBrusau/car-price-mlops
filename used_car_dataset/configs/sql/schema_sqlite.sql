PRAGMA journal_mode=WAL;

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
