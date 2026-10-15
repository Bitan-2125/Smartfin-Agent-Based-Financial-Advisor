"""Financial data store: local SQLite or an external PostgreSQL database."""

import json
import os
import sqlite3
import time
import hashlib
from contextlib import contextmanager


PLAN_LIFETIME_SECONDS = 7 * 24 * 60 * 60


def is_postgres(path):
    return path.startswith(("postgresql://", "postgres://"))


class PostgresConnection:
    """Keep parameterized queries portable between sqlite3 and psycopg."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query, parameters=()):
        if query == "BEGIN IMMEDIATE":
            return None
        return self.connection.execute(query.replace("?", "%s"), parameters)

    def executemany(self, query, rows):
        with self.connection.cursor() as cursor:
            cursor.executemany(query.replace("?", "%s"), rows)

    def commit(self):
        self.connection.commit()

    def close(self):
        self.connection.close()


@contextmanager
def database(path):
    postgres = is_postgres(path)
    if postgres:
        import psycopg
        connection = PostgresConnection(psycopg.connect(path, connect_timeout=10))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
    identity = "BIGSERIAL PRIMARY KEY" if postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
    foreign_id = "BIGINT" if postgres else "INTEGER"
    try:
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS users (
                id {identity},
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                recovery_hash TEXT NOT NULL,
                session_version INTEGER NOT NULL DEFAULT 1,
                consented_at BIGINT NOT NULL,
                created_at BIGINT NOT NULL
            )
        """)
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS plans (
                token TEXT PRIMARY KEY,
                user_id {foreign_id} REFERENCES users(id) ON DELETE CASCADE,
                data TEXT NOT NULL,
                expires_at BIGINT NOT NULL,
                created_at BIGINT NOT NULL DEFAULT 0
            )
        """)
        if postgres:
            connection.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE")
            connection.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS created_at BIGINT NOT NULL DEFAULT 0")
        else:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(plans)")}
            if "user_id" not in columns:
                connection.execute("ALTER TABLE plans ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
            if "created_at" not in columns:
                connection.execute("ALTER TABLE plans ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0")
        connection.execute("CREATE INDEX IF NOT EXISTS plans_by_owner ON plans(user_id, expires_at)")
        connection.execute("CREATE INDEX IF NOT EXISTS plans_by_expiry ON plans(expires_at)")
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS messages (
                id {identity},
                token TEXT NOT NULL REFERENCES plans(token) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS anonymous_usage (
                ip_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                occurred_at BIGINT NOT NULL
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS anonymous_usage_window ON anonymous_usage(ip_hash, kind, occurred_at)")
        yield connection
        connection.commit()
    finally:
        connection.close()


def allow_anonymous_usage(path, ip, kind, limit, seconds=3600):
    """Bound free API use without requiring an account or storing the raw IP."""
    now = int(time.time())
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()
    with database(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if is_postgres(path):
            lock_id = int.from_bytes(hashlib.sha256(ip_hash.encode()).digest()[:8], "big", signed=True)
            connection.execute("SELECT pg_advisory_xact_lock(?)", (lock_id,))
        connection.execute("DELETE FROM anonymous_usage WHERE occurred_at < ?", (now - 86400,))
        count = connection.execute(
            "SELECT COUNT(*) FROM anonymous_usage WHERE ip_hash = ? AND kind = ? AND occurred_at >= ?",
            (ip_hash, kind, now - seconds),
        ).fetchone()[0]
        if count >= limit:
            return False
        connection.execute(
            "INSERT INTO anonymous_usage (ip_hash, kind, occurred_at) VALUES (?, ?, ?)",
            (ip_hash, kind, now),
        )
        return True


def save_plan(path, token, data):
    now = int(time.time())
    with database(path) as connection:
        connection.execute("DELETE FROM plans WHERE expires_at <= ?", (now,))
        connection.execute(
            "INSERT INTO plans (token, user_id, data, expires_at, created_at) VALUES (?, ?, ?, ?, ?)",
            (token, None, json.dumps(data), now + PLAN_LIFETIME_SECONDS, time.time_ns()),
        )


def prune_expired(connection, now):
    expired = connection.execute("SELECT 1 FROM plans WHERE expires_at <= ? LIMIT 1", (now,)).fetchone()
    if expired:
        connection.execute("DELETE FROM plans WHERE expires_at <= ?", (now,))


def load_plan(path, token):
    if not token:
        return None
    with database(path) as connection:
        prune_expired(connection, int(time.time()))
        row = connection.execute(
            "SELECT data FROM plans WHERE token = ? AND user_id IS NULL AND expires_at > ?",
            (token, int(time.time())),
        ).fetchone()
    return json.loads(row[0]) if row else None


def load_history(path, token, limit=20):
    if not token:
        return []
    with database(path) as connection:
        rows = connection.execute(
            "SELECT m.role, m.content FROM messages m JOIN plans p ON p.token = m.token "
            "WHERE m.token = ? AND p.user_id IS NULL AND p.expires_at > ? "
            "ORDER BY id DESC LIMIT ?",
            (token, int(time.time()), limit),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def save_exchange(path, token, query, response):
    with database(path) as connection:
        owned = connection.execute(
            "SELECT 1 FROM plans WHERE token = ? AND user_id IS NULL AND expires_at > ?", (token, int(time.time()))
        ).fetchone()
        if not owned:
            raise ValueError("Plan is unavailable")
        connection.executemany(
            "INSERT INTO messages (token, role, content) VALUES (?, ?, ?)",
            [(token, "user", query), (token, "assistant", response)],
        )


def delete_plan(path, token):
    if not token:
        return
    with database(path) as connection:
        connection.execute("DELETE FROM plans WHERE token = ? AND user_id IS NULL", (token,))
