import functools
import http.client
import tempfile
import threading
import unittest
from pathlib import Path

from src.auditor.boundary.range_static_server import RangeRequestHandler


class RangeStaticServerTest(unittest.TestCase):
    def test_serves_partial_content_for_video_seek(self):
        import http.server

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.mp4").write_bytes(b"0123456789")
            handler = functools.partial(RangeRequestHandler, directory=str(root))
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection(*server.server_address)
                connection.request("GET", "/sample.mp4", headers={"Range": "bytes=2-5"})
                response = connection.getresponse()
                self.assertEqual(response.status, 206)
                self.assertEqual(response.getheader("Accept-Ranges"), "bytes")
                self.assertEqual(response.getheader("Content-Range"), "bytes 2-5/10")
                self.assertEqual(response.read(), b"2345")
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
