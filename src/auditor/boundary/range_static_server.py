"""Serve a local static annotation packet with seekable MP4 byte ranges."""
from __future__ import annotations

import argparse
import email.utils
import http.server
import re
import urllib.parse
from pathlib import Path


RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def _local_path(self) -> Path:
        request_path = urllib.parse.urlsplit(self.path).path
        return Path(self.translate_path(request_path))

    def _range(self, size: int) -> tuple[int, int] | None:
        match = RANGE_RE.fullmatch((self.headers.get("Range") or "").strip())
        if not match:
            return None
        lo, hi = match.groups()
        if lo:
            start = int(lo)
            end = min(int(hi) if hi else size - 1, size - 1)
        else:
            length = min(int(hi or 0), size)
            start, end = size - length, size - 1
        if start < 0 or start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return (-1, -1)
        return start, end

    def do_GET(self) -> None:
        path = self._local_path()
        if not path.is_file() or not self.headers.get("Range"):
            return super().do_GET()
        size = path.stat().st_size
        byte_range = self._range(size)
        if byte_range is None:
            return super().do_GET()
        start, end = byte_range
        if start < 0:
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header(
            "Last-Modified", email.utils.formatdate(path.stat().st_mtime, usegmt=True)
        )
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    block = handle.read(min(1 << 16, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_HEAD(self) -> None:
        path = self._local_path()
        if not path.is_file():
            return super().do_HEAD()
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.send_header(
            "Last-Modified", email.utils.formatdate(path.stat().st_mtime, usegmt=True)
        )
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.bind not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("refusing a non-loopback bind")
    root = args.directory.resolve()
    if not root.is_dir():
        raise SystemExit(f"directory does not exist: {root}")
    handler = lambda *a, **kw: RangeRequestHandler(*a, directory=str(root), **kw)
    server = http.server.ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"serving {root} at http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
