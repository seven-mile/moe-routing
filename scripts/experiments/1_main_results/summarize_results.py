#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


RUN_NAME_RE = re.compile(
    r"(?P<mode>baseline|lossless|optimum)_c(?P<concurrency>\d+)\.log$"
)
METRIC = "Output token throughput (tok/s)"


def infer_mode(row):
    if row.get("mode"):
        return row["mode"]
    match = RUN_NAME_RE.search(row.get("filename", ""))
    return match.group("mode") if match else None


def number(row, key):
    value = row.get(key, "").replace(",", "").split()[0]
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize main-result throughput")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-protocol",
        action="store_true",
        help="Fail on request-count mismatches, failed requests, or missing baseline",
    )
    parser.add_argument(
        "--required-modes",
        nargs="+",
        default=[],
        help="Policy modes that must occur exactly once at every concurrency",
    )
    parser.add_argument(
        "--expected-prompt-multiplier",
        type=int,
        help="Require successful requests to equal concurrency times this value",
    )
    args = parser.parse_args()
    if args.expected_prompt_multiplier is not None and args.expected_prompt_multiplier <= 0:
        parser.error("--expected-prompt-multiplier must be positive")

    with args.csv_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped = defaultdict(dict)
    for row in rows:
        mode = infer_mode(row)
        if mode is None:
            parser.error(f"cannot infer mode from row: {row.get('filename')}")
        concurrency = int(number(row, "Maximum request concurrency"))
        if mode in grouped[concurrency]:
            parser.error(
                f"duplicate {mode} row at concurrency {concurrency}: "
                f"{row.get('filename', '<unknown>')}"
            )
        grouped[concurrency][mode] = row

    lines = [
        "# Main-result summary",
        "",
        "| Concurrency | Baseline tok/s | Lossless tok/s | Lossless speedup | "
        "Optimum tok/s | Optimum speedup | Requests by mode |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    protocol_errors = []
    for concurrency, modes in sorted(grouped.items()):
        missing_modes = sorted(set(args.required_modes) - set(modes))
        if missing_modes:
            protocol_errors.append(
                f"concurrency {concurrency}: required modes are missing {missing_modes}"
            )
        if "baseline" not in modes:
            protocol_errors.append(f"concurrency {concurrency}: baseline is missing")
            continue
        baseline = number(modes["baseline"], METRIC)
        if baseline <= 0:
            protocol_errors.append(
                f"concurrency {concurrency}: baseline throughput is not positive"
            )
            continue
        lossless = number(modes["lossless"], METRIC) if "lossless" in modes else None
        optimum = number(modes["optimum"], METRIC) if "optimum" in modes else None
        requests = {
            mode: int(number(row, "Successful requests"))
            for mode, row in sorted(modes.items())
        }
        failed_requests = {
            mode: int(number(row, "Failed requests"))
            for mode, row in sorted(modes.items())
            if row.get("Failed requests", "").strip()
        }
        missing_failure_counts = sorted(set(modes) - set(failed_requests))
        if missing_failure_counts:
            protocol_errors.append(
                f"concurrency {concurrency}: failed-request counts are missing "
                f"for {missing_failure_counts}"
            )
        if len(set(requests.values())) > 1:
            protocol_errors.append(
                f"concurrency {concurrency}: unequal successful requests {requests}"
            )
        if args.expected_prompt_multiplier is not None:
            expected_requests = concurrency * args.expected_prompt_multiplier
            unexpected_requests = {
                mode: count
                for mode, count in requests.items()
                if count != expected_requests
            }
            if unexpected_requests:
                protocol_errors.append(
                    f"concurrency {concurrency}: expected {expected_requests} successful "
                    f"requests, found {unexpected_requests}"
                )
        nonzero_failures = {
            mode: count for mode, count in failed_requests.items() if count != 0
        }
        if nonzero_failures:
            protocol_errors.append(
                f"concurrency {concurrency}: failed requests {nonzero_failures}"
            )

        requests_text = ", ".join(f"{mode}={count}" for mode, count in requests.items())
        lines.append(
            "| {c} | {b:.2f} | {l} | {ls} | {o} | {os} | {r} |".format(
                c=concurrency,
                b=baseline,
                l=f"{lossless:.2f}" if lossless is not None else "-",
                ls=f"{lossless / baseline:.3f}x" if lossless is not None else "-",
                o=f"{optimum:.2f}" if optimum is not None else "-",
                os=f"{optimum / baseline:.3f}x" if optimum is not None else "-",
                r=requests_text,
            )
        )

    if protocol_errors:
        lines.extend(["", "## Protocol warnings", ""])
        lines.extend(f"- {error}" for error in protocol_errors)

    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    for error in protocol_errors:
        print(f"WARNING: {error}", file=sys.stderr)
    if args.strict_protocol and protocol_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
