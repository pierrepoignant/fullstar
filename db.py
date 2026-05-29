#!/usr/bin/env python3
"""MySQL store for recipes saved from the Fullstar Recipe Studio.

A recipe is persisted as its full JSON payload (the dict returned by
``chopper_recipes.make_recipe``) plus the author's name/email and a
user-given title.

Connection settings come from the OVH managed-MySQL env vars (these are
injected from the Kubernetes secret), with ``MYSQL_*`` and local defaults as
fallbacks for development:

    DATABASE_OVH__HOST / MYSQL_HOST          (default 127.0.0.1)
    DATABASE_OVH__PORT / MYSQL_PORT          (default 3306)
    DATABASE_OVH__USER / MYSQL_USER          (default fullstar)
    DATABASE_OVH__PASSWORD / MYSQL_PASSWORD  (default "")
    DATABASE_OVH__NAME / MYSQL_DATABASE      (default fullstar)

OVH managed databases require TLS, so the connection uses SSL by default
(set DB_SSL=0 to disable for a plaintext local server). Provide a CA bundle
path via DATABASE_OVH__SSL_CA to verify the server certificate.

The ``recipes`` table is created on demand (CREATE TABLE IF NOT EXISTS), so
the app survives MySQL not being ready at boot — the first save/read will
create it once MySQL is reachable.
"""
import json
import os
import re
import ssl as ssl_lib

import pymysql
from pymysql.cursors import DictCursor

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TRUTHY = ("1", "true", "on", "yes")

_initialized = False


def _env(*names, default=None):
    """First non-empty value among the given env var names."""
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


def _ssl_ctx():
    if (os.environ.get("DB_SSL", "1") or "").lower() not in TRUTHY:
        return None
    ca = os.environ.get("DATABASE_OVH__SSL_CA")
    ctx = ssl_lib.create_default_context(cafile=ca) if ca else ssl_lib.create_default_context()
    if not ca:  # TLS required by OVH, but no CA pinned -> encrypt without verifying.
        ctx.check_hostname = False
        ctx.verify_mode = ssl_lib.CERT_NONE
    return ctx


def _connect():
    return pymysql.connect(
        host=_env("DATABASE_OVH__HOST", "MYSQL_HOST", default="127.0.0.1"),
        port=int(_env("DATABASE_OVH__PORT", "MYSQL_PORT", default="3306")),
        user=_env("DATABASE_OVH__USER", "MYSQL_USER", default="fullstar"),
        password=_env("DATABASE_OVH__PASSWORD", "MYSQL_PASSWORD", default=""),
        database=_env("DATABASE_OVH__NAME", "MYSQL_DATABASE", default="fullstar"),
        charset="utf8mb4",
        autocommit=True,
        cursorclass=DictCursor,
        ssl=_ssl_ctx(),
        connect_timeout=10,
    )


def init_db():
    """Create the recipes table if it does not exist."""
    global _initialized
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS recipes(
                     id           BIGINT AUTO_INCREMENT PRIMARY KEY,
                     created_at   VARCHAR(32)  NOT NULL,
                     author_name  VARCHAR(255) NOT NULL,
                     author_email VARCHAR(255) NOT NULL,
                     title        VARCHAR(255) NOT NULL,
                     recipe_json  LONGTEXT     NOT NULL
                   ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    finally:
        conn.close()
    _initialized = True


def _ensure_init():
    if not _initialized:
        init_db()


def valid_email(email):
    return bool(EMAIL_RE.match((email or "").strip()))


def save_recipe(author_name, author_email, title, recipe):
    """Persist one recipe; returns the new row id."""
    from datetime import datetime, timezone
    _ensure_init()
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO recipes(created_at,author_name,author_email,title,recipe_json)"
                " VALUES(%s,%s,%s,%s,%s)",
                (created, author_name.strip(), author_email.strip(),
                 title.strip(), json.dumps(recipe)))
            return cur.lastrowid
    finally:
        conn.close()


def count_recipes(author_email=None):
    _ensure_init()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            if author_email:
                cur.execute("SELECT COUNT(*) AS cnt FROM recipes WHERE author_email=%s",
                            (author_email,))
            else:
                cur.execute("SELECT COUNT(*) AS cnt FROM recipes")
            return cur.fetchone()["cnt"]
    finally:
        conn.close()


def list_recipes(page=1, per_page=50, author_email=None):
    """Saved recipes, newest first, one page at a time.

    Pass ``author_email`` to return only that cook's recipes (the "My recipes"
    filter); omit it to return everyone's. Each item is the stored recipe dict
    augmented with the user's title, author name, formatted timestamp and id.
    """
    _ensure_init()
    offset = max(0, (page - 1) * per_page)
    conn = _connect()
    try:
        with conn.cursor() as cur:
            if author_email:
                cur.execute(
                    "SELECT * FROM recipes WHERE author_email=%s"
                    " ORDER BY id DESC LIMIT %s OFFSET %s",
                    (author_email, per_page, offset))
            else:
                cur.execute(
                    "SELECT * FROM recipes ORDER BY id DESC LIMIT %s OFFSET %s",
                    (per_page, offset))
            rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        r = json.loads(row["recipe_json"])
        r["title"] = row["title"]
        r["author_name"] = row["author_name"]
        r["created_at"] = row["created_at"]
        r["id"] = row["id"]
        out.append(r)
    return out
