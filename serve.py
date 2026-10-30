"""
SOALM V2 production entrypoint.

Runs the Flask app behind the Waitress WSGI server (works on Windows and Linux)
instead of the Flask development server. Use this for hosting the model as a
service that a website can call.

Environment variables:
    SOALM_HOST            Bind address              (default: 0.0.0.0)
    SOALM_PORT            Port to listen on         (default: 5000)
    SOALM_THREADS         Waitress worker threads   (default: 8)
    SOALM_CORS_ORIGINS    Comma-separated origins   (default: * )

Run locally:
    python serve.py

Note: the chat/LLM feature has been removed from this deployment. The
/api/chat route returns a "disabled" response; all prediction endpoints work.
"""

from __future__ import annotations

import os

from waitress import serve

# Importing soalm_api loads all 29 models once at import time.
from soalm_api import app, log, MODELS, MODEL_COUNT
from weather.weather_updater import start_weather_updater


def main() -> None:
    host = os.environ.get("SOALM_HOST", "0.0.0.0")
    # A platform-provided PORT (Cloud Run, Render, Heroku) wins, because the
    # platform routes traffic to that port. Otherwise SOALM_PORT, else 5000.
    port = int(os.environ.get("PORT") or os.environ.get("SOALM_PORT") or "5000")
    threads = int(os.environ.get("SOALM_THREADS", "8"))

    # Keep the offline weather cache fresh in the background.
    start_weather_updater()

    log("=" * 72)
    log("SOALM V2 API SERVER (production / waitress)")
    log(f"Models loaded: {len(MODELS)}/{MODEL_COUNT}")
    log(f"Worker threads: {threads}")
    log(f"Listening on http://{host}:{port}")
    log("=" * 72)

    serve(app, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
