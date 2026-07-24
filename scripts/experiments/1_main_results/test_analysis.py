#!/usr/bin/env python3
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from parse_results_bench import parse_log  # noqa: E402


RESULT_TEMPLATE = """\
================ Serving Benchmark Result ================
Successful requests: {requests}
Failed requests: 0
Maximum request concurrency: 4
Output token throughput (tok/s): {throughput}
==========================================================
"""


class AnalysisTests(unittest.TestCase):
    def test_parse_result_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline_c4.log"
            path.write_text(
                RESULT_TEMPLATE.format(requests=8, throughput=10.0),
                encoding="utf-8",
            )
            result, error = parse_log(path)

        self.assertIsNone(error)
        self.assertEqual(result["mode"], "baseline")
        self.assertEqual(result["Configured concurrency"], "4")
        self.assertEqual(result["Successful requests"], "8")

    def test_parser_fails_if_any_requested_log_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "baseline_c4.log"
            bad = root / "lossless_c4.log"
            output = root / "results.csv"
            good.write_text(
                RESULT_TEMPLATE.format(requests=8, throughput=10.0),
                encoding="utf-8",
            )
            bad.write_text("incomplete log\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "parse_results_bench.py"),
                    "--fail-on-error",
                    "--output",
                    str(output),
                    str(good),
                    str(bad),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Failed:                1", completed.stderr)

    def test_summary_enforces_modes_and_prompt_count(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "results.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "filename",
                        "mode",
                        "Maximum request concurrency",
                        "Successful requests",
                        "Failed requests",
                        "Output token throughput (tok/s)",
                    ],
                )
                writer.writeheader()
                for index, mode in enumerate(("baseline", "lossless", "optimum")):
                    writer.writerow(
                        {
                            "filename": f"{mode}_c4.log",
                            "mode": mode,
                            "Maximum request concurrency": 4,
                            "Successful requests": 8,
                            "Failed requests": 0,
                            "Output token throughput (tok/s)": 10 + index,
                        }
                    )

            base_command = [
                sys.executable,
                str(SCRIPT_DIR / "summarize_results.py"),
                "--strict-protocol",
                str(csv_path),
                "--required-modes",
                "baseline",
                "lossless",
                "optimum",
            ]
            valid = subprocess.run(
                [*base_command, "--expected-prompt-multiplier", "2"],
                check=False,
                capture_output=True,
                text=True,
            )
            invalid = subprocess.run(
                [*base_command, "--expected-prompt-multiplier", "3"],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(valid.returncode, 0)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("expected 12 successful requests", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
