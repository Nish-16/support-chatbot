"""
Web frontend for the demo: serves web/index.html and one JSON endpoint.

Same three stages as 30_demo.py, same production functions -- this only puts a
browser in front of them. Deliberately stdlib-only (http.server, no Flask or
FastAPI) so the frontend adds nothing to requirements.txt and `git clone` plus
the existing install still runs everything.

  GET  /              -> web/index.html
  GET  /<file>        -> anything else under web/
  POST /api/message   -> {"text": "...", "no_reply": false} -> run_once() as JSON

Costs API calls per message (1 classify, +1 draft, +1 more if the placeholder
guard regenerates). Needs GROQ_API_KEY in .env.

Usage:
  ./.venv/Scripts/python.exe scripts/31_serve.py
  ./.venv/Scripts/python.exe scripts/31_serve.py --port 8080
"""
import argparse
import importlib.util
import json
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from groq import APIError, APIConnectionError, APITimeoutError, RateLimitError

from groq_lib import MODEL, make_client
from vector_retrieval import get_retriever

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEB = os.path.join(os.path.dirname(_HERE), "web")

# 30_demo.py starts with a digit, so it cannot be imported by name.
_spec = importlib.util.spec_from_file_location("demo_mod", os.path.join(_HERE, "30_demo.py"))
_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_demo)

MAX_BODY = 64 * 1024          # a tweet-length message; refuse anything absurd
MAX_TEXT = 4000

_client = None
_retriever = None
# One Groq client and one fitted TF-IDF matrix shared across requests; the lock
# keeps two browser tabs from interleaving inside the retriever's state.
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = os.path.normpath(os.path.join(_WEB, rel))
        # Refuse anything that escapes web/ -- this serves the local filesystem.
        if not target.startswith(_WEB) or not os.path.isfile(target):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        with open(target, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/message":
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if length <= 0 or length > MAX_BODY:
            self._json(400, {"error": "empty or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", "")).strip()
            no_reply = bool(payload.get("no_reply"))
        except (ValueError, UnicodeDecodeError) as e:
            self._json(400, {"error": f"bad JSON: {e}"})
            return
        if not text:
            self._json(400, {"error": "no text supplied"})
            return
        if len(text) > MAX_TEXT:
            self._json(400, {"error": f"text too long (max {MAX_TEXT} characters)"})
            return

        try:
            with _lock:
                result = _demo.run_once(_client, text, _retriever, want_reply=not no_reply)
        except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as e:
            self._json(502, {"error": f"Groq call failed: {e}"})
            return
        except Exception as e:                      # noqa: BLE001 -- a demo server
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        self._json(200, result)


def main():
    global _client, _retriever
    p = argparse.ArgumentParser(description="Serve the web demo.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="default binds to localhost only")
    p.add_argument("--retriever", choices=["tfidf", "vector"], default="tfidf")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.path.isfile(os.path.join(_WEB, "index.html")):
        print(f"web/index.html not found at {_WEB}", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Put it in .env (see .env.example).", file=sys.stderr)
        sys.exit(1)

    _client = make_client()
    print(f"fitting the {args.retriever} retriever...")
    _retriever = get_retriever(args.retriever)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  model:     {MODEL}")
    print(f"  retriever: {args.retriever}")
    print(f"  serving:   http://{args.host}:{args.port}\n")
    print("Ctrl-C to stop. Each message sent from the page costs API calls.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()


if __name__ == "__main__":
    main()
