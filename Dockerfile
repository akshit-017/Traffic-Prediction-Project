# ═══════════════════════════════════════════════════════════════════
# Dockerfile — Bengaluru Traffic Prediction System
# Target: Hugging Face Spaces (Docker SDK, Free Tier)
# ═══════════════════════════════════════════════════════════════════
#
# CRITICAL: This app uses in-memory dicts (edge_counters, active_sessions)
# for fleet state. It MUST run on a SINGLE worker process to avoid
# state fragmentation. The CMD below enforces this.
# ═══════════════════════════════════════════════════════════════════

FROM python:3.10-slim

# ── System deps (slim base has everything we need) ───────────

# ── Non-root user (mandatory for HF Spaces) ─────────────────────
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ─────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────────
# .dockerignore excludes Raw/, notebooks/, data/, outputs/,
# train_models.py, and the 727 MB best_traffic_model.pkl if desired
COPY --chown=user . .

# ── Expose port 7860 (mandatory for Hugging Face Spaces) ────────
EXPOSE 7860

# ── Environment ─────────────────────────────────────────────────
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# ── Run Flask dev server (single process for stateful dicts) ────
# Using Flask's built-in server guarantees a single process.
# For production-grade single-worker, use gunicorn:
#   CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "4", "--timeout", "120", "app:app"]
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
