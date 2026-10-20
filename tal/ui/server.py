"""A small stdlib HTTP server for the local TAL UI.

Deliberately no framework: the lab machine may be offline, and the UI only
needs JSON endpoints plus three static files. It binds to loopback by default;
there is no authentication, so do not expose it on a public interface.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .. import __version__
from ..errors import SpecError
from .jobs import JobManager
from .services import (
    chat,
    probe,
    read_report,
    resume_experiment,
    start_batch_probe,
    start_capability,
    start_experiment,
    start_interpret,
)
from .store import Workspace

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 8 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class UiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], workspace: Workspace):
        super().__init__(address, UiHandler)
        self.workspace = workspace
        self.jobs = JobManager()
        self.static = STATIC_DIR
        self._lock = threading.Lock()


class UiHandler(BaseHTTPRequestHandler):
    server: UiServer
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def log_message(self, fmt: str, *args: Any) -> None:  # keep the console readable
        if args and str(args[1]) not in ("200", "304"):
            print(f"[ui] {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str, *, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message, "status": status}, status=status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ApiError(413, "request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError(400, "request body must be a JSON object")
        return data

    def _parts(self) -> list[str]:
        path = unquote(urlparse(self.path).path)
        return [part for part in path.split("/") if part]

    # ------------------------------------------------------------------ #
    # dispatch
    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def _dispatch(self, method: str) -> None:
        try:
            parts = self._parts()
            if parts[:1] == ["api"]:
                self._api(method, parts[1:])
            elif method == "GET":
                self._static(parts)
            else:
                raise ApiError(405, "method not allowed")
        except ApiError as exc:
            self._error(exc.status, exc.message)
        except SpecError as exc:
            self._error(400, str(exc))
        except KeyError as exc:
            self._error(404, f"not found: {exc}")
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 - last resort for the UI
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def _api(self, method: str, rest: list[str]) -> None:
        ws = self.server.workspace
        jobs = self.server.jobs

        if rest == ["health"]:
            return self._json({"ok": True, "version": __version__})
        if rest == ["state"] and method == "GET":
            return self._json(ws.summary())

        # models ---------------------------------------------------------
        if rest[:1] == ["models"]:
            if len(rest) == 1:
                if method == "GET":
                    return self._json({"models": ws.models()})
                if method == "POST":
                    return self._json({"model": ws.add_model(self._body())}, status=201)
            if len(rest) == 2:
                model_id = rest[1]
                if method == "PATCH":
                    return self._json({"model": ws.update_model(model_id, self._body())})
                if method == "DELETE":
                    ws.delete_model(model_id)
                    return self._json({"deleted": model_id})
                if method == "GET":
                    return self._json({"model": ws.get_model(model_id)})
            if len(rest) == 3 and rest[2] == "probe" and method == "POST":
                return self._json(probe(ws, rest[1]))

        # prompt libraries ----------------------------------------------
        if rest[:1] == ["prompts"]:
            if len(rest) == 1:
                if method == "GET":
                    return self._json({"prompts": ws.list_prompts()})
                if method == "POST":
                    return self._json({"prompt": ws.add_prompt(self._body())}, status=201)
            if len(rest) == 2:
                prompt_id = rest[1]
                if method == "GET":
                    data = ws.get_prompt(prompt_id)
                    return self._json(
                        {
                            "id": data["id"],
                            "name": data["name"],
                            "kind": data["kind"],
                            "format": data["format"],
                            "count": len(data.get("items") or []),
                            "items": data.get("items") or [],
                        }
                    )
                if method == "DELETE":
                    ws.delete_prompt(prompt_id)
                    return self._json({"deleted": prompt_id})

        # chat -----------------------------------------------------------
        if rest == ["chat"] and method == "POST":
            return self._json(chat(ws, self._body()))

        # jobs -----------------------------------------------------------
        if rest[:1] == ["jobs"]:
            if len(rest) == 1 and method == "GET":
                return self._json({"jobs": [j.to_dict(logs=False) for j in jobs.list()]})
            if len(rest) == 2 and method == "GET":
                job = jobs.get(rest[1])
                return self._json({"job": job.to_dict(logs=True, limit=600)})

        # experiments ----------------------------------------------------
        if rest[:1] == ["experiments"]:
            if len(rest) == 1:
                if method == "GET":
                    return self._json({"experiments": ws.runs()})
                if method == "POST":
                    return self._json(start_experiment(ws, jobs, self._body()), status=202)
            if len(rest) == 2 and method == "GET":
                run = ws.get_run(rest[1])
                related = [j.to_dict(logs=False) for j in jobs.list(kind="experiment") if j.run_id == rest[1]]
                return self._json({"run": run, "jobs": related})
            if len(rest) == 3 and rest[2] == "resume" and method == "POST":
                return self._json(resume_experiment(ws, jobs, rest[1], self._body()), status=202)

        # batch probes ---------------------------------------------------
        if rest[:1] == ["probes"]:
            if len(rest) == 1:
                if method == "GET":
                    return self._json({"probes": ws.probe_results()})
                if method == "POST":
                    return self._json(start_batch_probe(ws, jobs, self._body()), status=202)
            if len(rest) == 2 and method == "GET":
                return self._json(ws.get_probe(rest[1]))

        # capability -----------------------------------------------------
        if rest[:1] == ["capability"]:
            if len(rest) == 1:
                if method == "GET":
                    return self._json({"results": ws.capability_results()})
                if method == "POST":
                    return self._json(start_capability(ws, jobs, self._body()), status=202)
            if len(rest) == 2 and method == "GET":
                path = ws.capability_dir / f"{rest[1]}.json"
                if not path.exists():
                    raise KeyError(rest[1])
                return self._json(json.loads(path.read_text(encoding="utf-8")))

        # layer inspection ----------------------------------------------
        if rest[:1] == ["interpret"]:
            if len(rest) == 1:
                if method == "POST":
                    return self._json(start_interpret(ws, jobs, self._body()), status=202)
                if method == "GET":
                    results = []
                    for path in sorted(
                        ws.interpret_dir.glob("*.json"),
                        key=lambda item: item.stat().st_mtime,
                        reverse=True,
                    ):
                        try:
                            data = json.loads(path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            continue
                        results.append(
                            {
                                "id": data.get("id") or path.stem,
                                "created_at": data.get("created_at"),
                                "model": data.get("model"),
                                "prompt": data.get("prompt"),
                                "summary": data.get("summary"),
                                "duration_s": data.get("duration_s"),
                            }
                        )
                    return self._json({"results": results})
            if len(rest) == 2 and method == "GET":
                path = ws.interpret_dir / f"{rest[1]}.json"
                if not path.exists():
                    raise KeyError(rest[1])
                return self._json(json.loads(path.read_text(encoding="utf-8")))

        # senbonzakura ---------------------------------------------------
        if rest == ["senbonzakura"] and method == "POST":
            body = self._body()
            body["engine"] = "senbonzakura"
            return self._json(start_capability(ws, jobs, body), status=202)

        # reports --------------------------------------------------------
        if rest[:1] == ["report"] and len(rest) >= 3 and method == "GET":
            run_id = rest[1]
            filename = "/".join(rest[2:])
            text = read_report(ws, run_id, filename)
            return self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")

        raise ApiError(404, f"unknown endpoint: {method} /{'/'.join(rest)}")

    # ------------------------------------------------------------------ #
    # static
    # ------------------------------------------------------------------ #
    def _static(self, parts: list[str]) -> None:
        if not parts:
            parts = ["index.html"]
        name = parts[-1]
        if name not in ("index.html", "app.js", "style.css"):
            raise ApiError(404, "not found")
        path = self.server.static / name
        if not path.exists():
            raise ApiError(500, f"UI asset missing: {path}")
        mime = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8"}.get(
            name
        ) or mimetypes.guess_type(name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), mime)


def serve(
    workspace: str | Path = "ui-data",
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> int:
    ws = Workspace(workspace)
    server = UiServer((host, port), ws)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"TAL UI  workspace: {ws.root}")
    print(f"TAL UI  listening: {url}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: the UI has no authentication; do not expose it to untrusted networks.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nTAL UI stopped.")
    finally:
        server.server_close()
    return 0
