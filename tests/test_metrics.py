"""Verifies that displayed telemetry comes only from real completed requests."""
import unittest
from unittest.mock import patch

from api.app import AppState


class TestMeasuredMetrics(unittest.TestCase):
    def test_empty_history_contains_no_fake_data(self):
        state = AppState()
        self.assertEqual(state.get_latency_history(), [])
        self.assertEqual(state.get_throughput_history(), [])

    def test_request_metrics_come_from_observations(self):
        state = AppState()
        state.start_time = 1000.0
        with patch("api.app.time.time", return_value=1010.0):
            state.record_latency(10.0)
            state.record_latency(30.0)
            self.assertEqual(state.requests_per_second(), 0.2)
            self.assertEqual(state.metrics["avg_latency_ms"], 20.0)
            latency = state.get_latency_history()
            throughput = state.get_throughput_history()
        self.assertEqual(len(latency), 1)
        self.assertEqual(latency[0]["p50"], 20.0)
        self.assertEqual(latency[0]["p95"], 29.0)
        self.assertEqual(throughput, [
            {"timestamp": 1010, "requests_per_second": 2},
        ])

    def test_expired_data_not_reported_as_recent(self):
        state = AppState()
        state._completed_events = [(100.0, 5.0)]
        with patch("api.app.time.time", return_value=1000.0):
            self.assertEqual(state.requests_per_second(), 0.0)
            self.assertEqual(state.get_latency_history(), [])
            self.assertEqual(state.get_throughput_history(), [])


if __name__ == "__main__":
    unittest.main()
