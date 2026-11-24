"""Optional SQLite persistence. No raw prompts or responses are retained."""
import json
import sqlite3
from contextlib import closing


class Storage:
    def __init__(self, path):
        self.path = path
        with closing(self.connect()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT PRIMARY KEY, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage (
                    request_id TEXT PRIMARY KEY, model TEXT NOT NULL,
                    worker_id TEXT NOT NULL, latency_ms REAL NOT NULL,
                    completed_at REAL NOT NULL
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def save_model(self, metadata):
        with closing(self.connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO models VALUES (?, ?)",
                         (metadata["name"], json.dumps(metadata)))

    def delete_model(self, name):
        with closing(self.connect()) as conn, conn:
            conn.execute("DELETE FROM models WHERE name = ?", (name,))

    def models(self):
        with closing(self.connect()) as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT metadata FROM models ORDER BY name")]

    def record_usage(self, request_id, model, worker_id, latency_ms, completed_at):
        with closing(self.connect()) as conn, conn:
            conn.execute("INSERT OR IGNORE INTO usage VALUES (?, ?, ?, ?, ?)",
                         (request_id, model, worker_id, latency_ms, completed_at))

    def usage_summary(self):
        with closing(self.connect()) as conn:
            count, avg = conn.execute("SELECT COUNT(*), AVG(latency_ms) FROM usage").fetchone()
            return {"recorded_requests": count, "avg_latency_ms": avg or 0.0}
