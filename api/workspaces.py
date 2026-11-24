"""Opt-in local beta accounts, scoped sessions, private text and daily quotas.

Not a complete public authentication service: require TLS and external
rate limiting, monitoring, recovery and security review before exposure.
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone

from inference.grounded import retrieve


class AccessDenied(RuntimeError):
    pass


class QuotaExceeded(RuntimeError):
    pass


class WorkspaceStore:
    def __init__(self, path, daily_limit=100):
        self.path = path
        self.daily_limit = daily_limit
        if daily_limit < 1:
            raise ValueError("daily limit must be positive")
        with closing(self.connect()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                    salt BLOB NOT NULL, password_hash BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    digest TEXT PRIMARY KEY, account_id INTEGER NOT NULL,
                    expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    account_id INTEGER NOT NULL, id TEXT NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY(account_id, id)
                );
                CREATE TABLE IF NOT EXISTS quota (
                    account_id INTEGER NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL,
                    PRIMARY KEY(account_id, day)
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    @staticmethod
    def _hash(password, salt):
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600000)

    @staticmethod
    def _digest(token):
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _validate(username, password):
        if not isinstance(username, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{3,40}", username):
            raise ValueError("username must contain 3–40 letters, numbers, dots, hyphens or underscores")
        if not isinstance(password, str) or not 12 <= len(password) <= 128:
            raise ValueError("password must contain 12–128 characters")
        return username.casefold()

    def register(self, username, password):
        username = self._validate(username, password)
        salt = os.urandom(16)
        try:
            with closing(self.connect()) as conn, conn:
                conn.execute("INSERT INTO accounts(username,salt,password_hash) VALUES(?,?,?)",
                             (username, salt, self._hash(password, salt)))
        except sqlite3.IntegrityError as exc:
            raise ValueError("Account unavailable") from exc
        return {"username": username}

    def login(self, username, password):
        username = self._validate(username, password)
        with closing(self.connect()) as conn, conn:
            row = conn.execute("SELECT id,salt,password_hash FROM accounts WHERE username=?", (username,)).fetchone()
            salt = row[1] if row else b"\0" * 16
            digest = self._hash(password, salt)
            if not row or not hmac.compare_digest(digest, row[2]):
                raise AccessDenied("Invalid credentials")
            token = secrets.token_urlsafe(32)
            conn.execute("DELETE FROM sessions WHERE expires <= ?", (time.time(),))
            conn.execute("INSERT INTO sessions VALUES(?,?,?)",
                         (self._digest(token), row[0], time.time() + 8 * 3600))
            return {"token": token, "expires_in_seconds": 8 * 3600}

    def authenticate(self, token):
        if not isinstance(token, str) or not 20 <= len(token) <= 128:
            raise AccessDenied("Session required")
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT account_id FROM sessions WHERE digest=? AND expires>?",
                               (self._digest(token), time.time())).fetchone()
        if not row:
            raise AccessDenied("Session invalid or expired")
        return row[0]

    def logout(self, token):
        with closing(self.connect()) as conn, conn:
            conn.execute("DELETE FROM sessions WHERE digest=?", (self._digest(token),))

    def save_document(self, account_id, doc_id, text):
        if not isinstance(doc_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", doc_id):
            raise ValueError("Invalid document id")
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise ValueError("Document must contain 1–20000 characters")
        with closing(self.connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            count = conn.execute("SELECT COUNT(*) FROM documents WHERE account_id=?",
                                 (account_id,)).fetchone()[0]
            existing = conn.execute("SELECT 1 FROM documents WHERE account_id=? AND id=?",
                                    (account_id, doc_id)).fetchone()
            if count >= 20 and not existing:
                raise ValueError("Workspace document limit reached")
            conn.execute("INSERT OR REPLACE INTO documents VALUES(?,?,?)", (account_id, doc_id, text))
        return {"id": doc_id}

    def documents(self, account_id):
        with closing(self.connect()) as conn:
            return [{"id": row[0], "text": row[1]} for row in conn.execute(
                "SELECT id,text FROM documents WHERE account_id=? ORDER BY id", (account_id,)
            )]

    def delete_document(self, account_id, doc_id):
        with closing(self.connect()) as conn, conn:
            cursor = conn.execute("DELETE FROM documents WHERE account_id=? AND id=?", (account_id, doc_id))
            return {"deleted": cursor.rowcount == 1}

    def consume(self, account_id):
        day = datetime.now(timezone.utc).date().isoformat()
        with closing(self.connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT count FROM quota WHERE account_id=? AND day=?", (account_id, day)).fetchone()
            count = row[0] if row else 0
            if count >= self.daily_limit:
                raise QuotaExceeded("Daily workspace quota exceeded")
            conn.execute("INSERT OR REPLACE INTO quota VALUES(?,?,?)", (account_id, day, count + 1))
        return self.daily_limit - count - 1

    def ask(self, account_id, question):
        docs = self.documents(account_id)
        if not docs:
            raise ValueError("Add a document first")
        remaining = self.consume(account_id)
        return {**retrieve(question, docs), "remaining_daily_queries": remaining}
