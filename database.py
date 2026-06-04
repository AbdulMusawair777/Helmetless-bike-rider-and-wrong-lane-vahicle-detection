# ╔══════════════════════════════════════════════════════════════════════╗
# ║  database.py  —  SQLite Violation Store                              ║
# ║  By Musaawar Khan                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join('output', 'violations.db')

# ─────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────

_SQL_CREATE_VIOLATIONS = """
CREATE TABLE IF NOT EXISTS violations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,

    -- session / timing
    session_id       TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    video_source     TEXT    NOT NULL,
    frame_number     INTEGER NOT NULL,
    frame_time_sec   REAL    NOT NULL,

    -- vehicle
    track_id         INTEGER NOT NULL,
    vehicle_class    TEXT    NOT NULL,

    -- rule broken
    violation_type   TEXT    NOT NULL,   -- WRONG_DIRECTION | NO_HELMET | BOTH

    -- details
    direction        TEXT    DEFAULT '',
    helmet_status    TEXT    DEFAULT '',

    -- ── LICENSE PLATE ─────────────────────────────────────────────
    plate_detected   INTEGER DEFAULT 0,  -- 1 = plate box found by model
    plate_text       TEXT    DEFAULT '',  -- OCR text (best read so far)
    plate_updated_at TEXT    DEFAULT '',  -- timestamp of last OCR update

    -- evidence
    snapshot_path    TEXT    DEFAULT '',
    confidence       REAL    DEFAULT 0.0,
    bbox_x1          INTEGER DEFAULT 0,
    bbox_y1          INTEGER DEFAULT 0,
    bbox_x2          INTEGER DEFAULT 0,
    bbox_y2          INTEGER DEFAULT 0,

    -- review
    reviewed         INTEGER DEFAULT 0,
    notes            TEXT    DEFAULT '',

    -- ── ONE ROW PER (session, track, violation type) ───────────────
    UNIQUE (session_id, track_id, violation_type)
);
"""

_SQL_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT    PRIMARY KEY,
    started_at        TEXT    NOT NULL,
    ended_at          TEXT    DEFAULT '',
    video_source      TEXT    NOT NULL,
    correct_direction TEXT    NOT NULL,
    total_frames      INTEGER DEFAULT 0,
    total_violations  INTEGER DEFAULT 0
);
"""

_SQL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_viol_session ON violations(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_viol_type    ON violations(violation_type);",
    "CREATE INDEX IF NOT EXISTS idx_viol_track   ON violations(track_id);",
    "CREATE INDEX IF NOT EXISTS idx_viol_plate   ON violations(plate_text);",
    "CREATE INDEX IF NOT EXISTS idx_viol_time    ON violations(timestamp);",
]


# ─────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


# ─────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables and indexes. Safe to call repeatedly."""
    with _conn() as c:
        c.execute(_SQL_CREATE_VIOLATIONS)
        c.execute(_SQL_CREATE_SESSIONS)
        for idx in _SQL_INDEXES:
            c.execute(idx)
    print(f'  [DB] Ready → {DB_PATH}')


# ─────────────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────────────

def start_session(session_id: str, video_source: str, correct_dir: str):
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO sessions"
            " (session_id, started_at, video_source, correct_direction)"
            " VALUES (?,?,?,?)",
            (session_id,
             datetime.now().isoformat(sep=' ', timespec='seconds'),
             video_source, correct_dir)
        )


def end_session(session_id: str, total_frames: int, total_violations: int):
    with _conn() as c:
        c.execute(
            "UPDATE sessions"
            " SET ended_at=?, total_frames=?, total_violations=?"
            " WHERE session_id=?",
            (datetime.now().isoformat(sep=' ', timespec='seconds'),
             total_frames, total_violations, session_id)
        )


# ─────────────────────────────────────────────────────────────────────
# INSERT VIOLATION
# ─────────────────────────────────────────────────────────────────────

def insert_violation(
    session_id:     str,
    video_source:   str,
    frame_number:   int,
    frame_time_sec: float,
    track_id:       int,
    vehicle_class:  str,
    violation_type: str,      # WRONG_DIRECTION | NO_HELMET | BOTH
    direction:      str,
    helmet_status:  str,
    plate_detected: bool,
    plate_text:     str,      # OCR text at time of first flag (may be '')
    snapshot_path:  str,
    confidence:     float,
    bbox:           tuple,    # (x1, y1, x2, y2)
) -> int:
    """
    Insert one violation row.  Returns the new row id, or the id of the
    existing row if this (session_id, track_id, violation_type) has already
    been recorded (INSERT OR IGNORE + SELECT fallback).

    This is the DB-level safety net — pipeline.py's ViolationRecorder
    already prevents duplicate calls, but this guard ensures correctness
    even if called directly or during testing.
    """
    ts   = datetime.now().isoformat(sep=' ', timespec='seconds')
    x1, y1, x2, y2 = bbox
    pt   = plate_text.upper().strip()
    p_ts = ts if pt else ''

    with _conn() as c:
        cur = c.execute("""
            INSERT OR IGNORE INTO violations (
                session_id, timestamp, video_source,
                frame_number, frame_time_sec,
                track_id, vehicle_class,
                violation_type, direction, helmet_status,
                plate_detected, plate_text, plate_updated_at,
                snapshot_path, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2
            ) VALUES (?,?,?, ?,?, ?,?, ?,?,?, ?,?,?, ?,?, ?,?,?,?)
        """, (
            session_id, ts, video_source,
            frame_number, round(frame_time_sec, 2),
            track_id, vehicle_class,
            violation_type, direction, helmet_status,
            int(plate_detected), pt, p_ts,
            snapshot_path, round(confidence, 4),
            x1, y1, x2, y2,
        ))
        if cur.lastrowid:
            return cur.lastrowid
        # Row already existed — return its id
        row = c.execute(
            "SELECT id FROM violations"
            " WHERE session_id=? AND track_id=? AND violation_type=?",
            (session_id, track_id, violation_type)
        ).fetchone()
        return row['id'] if row else -1


# ─────────────────────────────────────────────────────────────────────
# UPDATE PLATE TEXT  ← key new function
# Called whenever a better OCR read arrives for a track_id that is
# already flagged.  Updates ALL violation rows for that track in this
# session so the plate is never empty when OCR succeeds later.
# ─────────────────────────────────────────────────────────────────────

def update_plate_text(session_id: str, track_id: int, plate_text: str):
    """
    Retrofit plate_text into every existing violation row for this
    (session_id, track_id) pair.  Only overwrites if the current
    stored value is empty or shorter than the new read.
    """
    pt = plate_text.upper().strip()
    if not pt:
        return
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    with _conn() as c:
        c.execute("""
            UPDATE violations
               SET plate_text       = ?,
                   plate_detected   = 1,
                   plate_updated_at = ?
             WHERE session_id = ?
               AND track_id   = ?
               AND (plate_text = '' OR length(plate_text) < length(?))
        """, (pt, ts, session_id, track_id, pt))


# ─────────────────────────────────────────────────────────────────────
# QUERIES
# ─────────────────────────────────────────────────────────────────────

def fetch_violations(session_id: str = None,
                     violation_type: str = None) -> list:
    filters, params = [], []
    if session_id:
        filters.append("session_id=?");  params.append(session_id)
    if violation_type:
        filters.append("violation_type=?"); params.append(violation_type.upper())
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM violations {where} ORDER BY frame_number",
            params
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_summary(session_id: str = None) -> dict:
    where  = f"WHERE session_id=?" if session_id else ""
    params = [session_id] if session_id else []
    with _conn() as c:
        rows = c.execute(
            f"SELECT violation_type, COUNT(*) as n"
            f" FROM violations {where} GROUP BY violation_type",
            params
        ).fetchall()
    return {r['violation_type']: r['n'] for r in rows}


def fetch_sessions() -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def plate_lookup(text: str) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM violations WHERE plate_text LIKE ? ORDER BY timestamp",
            (f'%{text.upper().strip()}%',)
        ).fetchall()
    return [dict(r) for r in rows]