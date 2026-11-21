"""SQLite-Speicher: jede Generierung mit allen Parametern und der Bewertung des Users.

Das ist der Datensatz, aus dem später gelernt wird (Prompt-Optimierer, Bewertungsmodell, LoRA/DPO).
Bilder einer Anfrage teilen sich eine batch_id -> daraus entstehen Präferenzpaare (gleicher Prompt, 👍 vs 👎).
"""
import sqlite3
import time

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    prompt      TEXT NOT NULL,
    negative    TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL,
    steps       INTEGER NOT NULL,
    guidance    REAL NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    seed        INTEGER NOT NULL,
    file        TEXT NOT NULL,
    duration_s  REAL,
    rating      INTEGER NOT NULL DEFAULT 0,   -- -1 schlecht, 0 unbewertet, 1 gut
    favorite    INTEGER NOT NULL DEFAULT 0,
    note        TEXT NOT NULL DEFAULT '',
    rated_at    REAL,
    score       REAL                          -- vorhergesagte Wahrscheinlichkeit für 👍 (Bewertungsmodell), NULL = nicht bewertet
);
CREATE INDEX IF NOT EXISTS idx_images_batch ON images(batch_id);
CREATE INDEX IF NOT EXISTS idx_images_rating ON images(rating);
CREATE TABLE IF NOT EXISTS embeddings (        -- CLIP-Bildvektoren (float32, normiert), einmal pro Bild berechnet
    image_id INTEGER PRIMARY KEY REFERENCES images(id),
    vec      BLOB NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    if "score" not in {r["name"] for r in con.execute("PRAGMA table_info(images)")}:
        con.execute("ALTER TABLE images ADD COLUMN score REAL")   # Migration für Datenbanken vor dem Bewertungsmodell
        con.commit()
    return con


def insert_image(con, **row) -> int:
    row.setdefault("created_at", time.time())
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    cur = con.execute(f"INSERT INTO images ({cols}) VALUES ({marks})", tuple(row.values()))
    con.commit()
    return cur.lastrowid


def rate(con, image_id: int, rating: int | None, favorite: bool | None, note: str | None) -> dict | None:
    fields, vals = [], []
    if rating is not None:
        fields.append("rating = ?"); vals.append(rating)
    if favorite is not None:
        fields.append("favorite = ?"); vals.append(int(favorite))
    if note is not None:
        fields.append("note = ?"); vals.append(note)
    if fields:
        fields.append("rated_at = ?"); vals.append(time.time())
        con.execute(f"UPDATE images SET {', '.join(fields)} WHERE id = ?", (*vals, image_id))
        con.commit()
    return get(con, image_id)


def get(con, image_id: int) -> dict | None:
    r = con.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    return dict(r) if r else None


def list_images(con, filter_: str = "all", limit: int = 60, offset: int = 0) -> list[dict]:
    where = {
        "all": "1=1",
        "unrated": "rating = 0",
        "good": "rating = 1",
        "bad": "rating = -1",
        "favorite": "favorite = 1",
        "top": "rating = 0 AND score IS NOT NULL",   # unbewertete Bilder, vom Modell vorsortiert
    }.get(filter_, "1=1")
    order = "score DESC, id DESC" if filter_ == "top" else "created_at DESC, id DESC"
    rows = con.execute(
        f"SELECT * FROM images WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def stats(con) -> dict:
    r = con.execute(
        "SELECT COUNT(*) total, SUM(rating=1) good, SUM(rating=-1) bad, SUM(rating=0) unrated, "
        "SUM(favorite) favorites FROM images"
    ).fetchone()
    return {k: (r[k] or 0) for k in r.keys()}
