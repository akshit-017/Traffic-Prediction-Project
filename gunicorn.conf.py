"""
Gunicorn configuration file — automatically picked up by gunicorn when
present in the working directory.

This lets the simple ``gunicorn app:app`` start command work on Render
without needing ``--bind 0.0.0.0:$PORT`` on the command line.
"""

import os

# Render injects a PORT env var; default to 8000 locally.
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# MUST be 1: app.py uses in-memory dicts (edge_counters, active_sessions)
# for fleet state. Multiple workers would fragment this state.
workers = 1

# Use threads for concurrency instead of multiple processes.
threads = 4

# ML model loading can take a while; give it room.
timeout = 120
