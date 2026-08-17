"""
Gunicorn configuration file — automatically picked up by gunicorn when
present in the working directory.

This lets the simple ``gunicorn app:app`` start command work on Render
without needing ``--bind 0.0.0.0:$PORT`` on the command line.
"""

import os

# Render injects a PORT env var; default to 8000 locally.
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Keep workers low for free-tier Render instances (512 MB RAM).
workers = 2

# ML model loading can take a while; give it room.
timeout = 120
