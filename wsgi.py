"""Gunicorn entrypoint.

`gunicorn --preload wsgi:app` imports this once in the master process, which
loads the Epicure core model a single time and shares it with the forked
workers via copy-on-write.
"""
from app import app  # noqa: F401
