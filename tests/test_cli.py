import contextlib
import io
import unittest
from unittest.mock import patch

from apicalai_cli import main


class TestCLI(unittest.TestCase):
    def test_infer_command(self):
        with patch("apicalai_cli.request", return_value={"label": "positive"}) as call:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["infer", "wonderful"]), 0)
            self.assertIn("positive", output.getvalue())
        self.assertEqual(call.call_args.args[1], "/api/inference")
        self.assertEqual(call.call_args.args[2]["input"]["text"], "wonderful")
