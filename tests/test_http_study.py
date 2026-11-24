import threading
import unittest

from api.app import create_server
from inferbench.http_study import percentile, run_study


class TestHTTPStudy(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1, 3, 2], 50), 2)

    def test_real_http_requests(self):
        server = create_server(host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = run_study("http://127.0.0.1:" + str(server.server_port),
                               concurrency=(1, 2), requests=4)
            for run in result["runs"]:
                self.assertEqual(run["successful"], 4)
                self.assertEqual(run["failed"], 0)
                self.assertGreater(run["successful_requests_per_second"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
