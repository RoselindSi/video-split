"""A static server that answers Range requests, so video in a packet can be scrubbed.

WHY NOT `python -m http.server`. It replies 200 with the whole file to every
request, Range header or not. A browser cannot seek in an mp4 that way: it can
only play from the start, and everything the annotation page does through
`video.currentTime` -- cutting a segment, nudging a boundary, jumping to a
segment -- stops working. The videos themselves are fine; they are faststart,
with `moov` ahead of `mdat`.

This matters only for previewing a packet over HTTP. An annotator opens the
page from the unzipped folder over `file://`, where the browser seeks the
local file directly and no server is involved.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
import socketserver

RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_head(self):
        header = self.headers.get("Range")
        if not header:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            stream = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(stream.fileno()).st_size
        m = RANGE.match(header.strip())
        if not m:
            stream.close()
            self.send_error(400, "bad Range")
            return None
        start, end = m.group(1), m.group(2)
        if start == "":
            # A suffix range: the last N bytes.
            n = int(end or 0)
            start, end = max(0, size - n), size - 1
        else:
            start = int(start)
            end = int(end) if end else size - 1
        if start >= size or start > end:
            stream.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        end = min(end, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        stream.seek(start)
        self._remaining = end - start + 1
        return stream

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        self._remaining = None
        while n > 0:
            block = source.read(min(64 * 1024, n))
            if not block:
                break
            outputfile.write(block)
            n -= len(block)

    def end_headers(self):
        if "Accept-Ranges" not in self._headers_buffer_text():
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _headers_buffer_text(self):
        return b"".join(getattr(self, "_headers_buffer", [])).decode(
            "latin-1", "replace")

    def log_message(self, fmt, *args):
        pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--directory", required=True)
    ap.add_argument("--port", type=int, default=8731)
    a = ap.parse_args()
    handler = functools.partial(Handler, directory=a.directory)
    with Server(("127.0.0.1", a.port), handler) as httpd:
        print("serving %s on http://127.0.0.1:%d" % (a.directory, a.port),
              flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
