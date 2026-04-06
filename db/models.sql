-- db/models.sql
-- Single source of truth for SQLite schema

PRAGMA foreign_keys = ON;

-- =========================
-- faces: registered face embeddings
-- =========================
CREATE TABLE IF NOT EXISTS faces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    embedding_blob  BLOB NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);

-- =========================
-- plates: registered license plates
-- =========================
CREATE TABLE IF NOT EXISTS plates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    owner           TEXT NOT NULL,
    plate_text_norm TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_plates_text_norm ON plates(plate_text_norm);
CREATE INDEX IF NOT EXISTS idx_plates_owner ON plates(owner);

-- =========================
-- jobs: processing jobs created from web/API
-- =========================
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts    TEXT NOT NULL,
    end_ts      TEXT NOT NULL,
    source      TEXT NOT NULL, -- file path / camera url / etc
    status      TEXT NOT NULL DEFAULT 'queued', -- queued/running/done/failed
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);

-- =========================
-- results: output artifacts per job
-- =========================
CREATE TABLE IF NOT EXISTS results (
    job_id      INTEGER PRIMARY KEY,
    output_path TEXT NOT NULL,
    thumb_path  TEXT,
    meta_json   TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at);
