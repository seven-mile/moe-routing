#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import TextIO


RESULT_BLOCK_RE = re.compile(
    r"^=+\s+Serving Benchmark Result\s+=+\s*$\n"
    r"(?P<body>.*?)(?=^=+\s*$)",
    re.DOTALL | re.MULTILINE,
)
RUN_NAME_RE = re.compile(
    r"(?P<mode>baseline|lossless|optimum)_c(?P<concurrency>\d+)\.log$"
)


def parse_log(file_path: Path):
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"Read error: {exc}"

    block_match = RESULT_BLOCK_RE.search(content)
    if not block_match:
        return None, "Format error: benchmark result block not found"

    data = {"filename": str(file_path)}
    run_match = RUN_NAME_RE.search(file_path.name)
    if run_match:
        data["mode"] = run_match.group("mode")
        data["Configured concurrency"] = run_match.group("concurrency")

    for raw_line in block_match.group("body").strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-"):
            continue

        position_match = re.match(r"Position (\d+):\s+([\d.]+)", line)
        if position_match:
            data[f"Pos_{position_match.group(1)}_Acceptance(%)"] = (
                position_match.group(2)
            )
            continue

        if ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            if value:
                data[key] = value
            continue

        parts = re.split(r"\s{2,}", line)
        if len(parts) >= 2:
            data[parts[0].strip()] = parts[1].strip()

    return data, None


def write_results(results, output_format: str, output: TextIO):
    if output_format == "json":
        json.dump(results, output, indent=2)
        output.write("\n")
        return

    all_keys = set().union(*(result.keys() for result in results))
    preferred = ["filename", "mode", "Configured concurrency"]
    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Collect vLLM serving benchmark data")
    parser.add_argument("files", nargs="+", type=Path, help="Benchmark logs to parse")
    parser.add_argument(
        "--format", choices=("csv", "json"), default="csv", help="Output format"
    )
    parser.add_argument(
        "--output", type=Path, help="Write results to this file instead of stdout"
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return nonzero if any requested log cannot be parsed",
    )
    args = parser.parse_args()

    results = []
    errors = []
    all_keys = set()
    for file_path in args.files:
        result, error = parse_log(file_path)
        if error:
            errors.append((file_path, error))
            continue
        results.append(result)
        all_keys.update(result.keys())

    if not results:
        print(f"Error: no valid data extracted from {len(args.files)} files", file=sys.stderr)
        for file_path, error in errors:
            print(f"  [{file_path}]: {error}", file=sys.stderr)
        raise SystemExit(1)

    warnings = []
    for result in results:
        missing = all_keys - set(result.keys())
        if missing:
            warnings.append((result["filename"], sorted(missing)))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as output:
            write_results(results, args.format, output)
    else:
        write_results(results, args.format, sys.stdout)

    print("\n" + "=" * 30, file=sys.stderr)
    print("ANALYSIS SUMMARY", file=sys.stderr)
    print("=" * 30, file=sys.stderr)
    print(f"Total files processed: {len(args.files)}", file=sys.stderr)
    print(f"Successful:            {len(results)}", file=sys.stderr)
    print(f"Failed:                {len(errors)}", file=sys.stderr)
    for file_path, error in errors:
        print(f"  ERROR {file_path}: {error}", file=sys.stderr)
    for filename, missing in warnings:
        print(f"  WARNING {filename}: missing {missing}", file=sys.stderr)
    print("=" * 30, file=sys.stderr)

    if args.fail_on_error and errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
