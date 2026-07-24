#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


POLICIES = {
    "baseline": {
        # Preserve the historical runner exactly: this range masks every
        # Qwen3-30B MoE layer back to the model's base expert count.
        "function": "spec_with_list_layer_range",
        "args": [
            [
                16.22663120720821,
                16.22663120720821,
                11.861973917295447,
                11.861973917295447,
                7.394487721217839,
            ],
            [0, 100],
        ],
    },
    "lossless": {
        "function": "spec_with_list_layer_range",
        "args": [
            [
                16.342797244301977,
                16.28686882612569,
                16.28686882612569,
                14.62508916637383,
            ],
            [0, 0],
        ],
    },
    "optimum": {
        "function": "spec_with_list_layer_range",
        "args": [
            [
                16.22663120720821,
                16.22663120720821,
                11.861973917295447,
                11.861973917295447,
                7.394487721217839,
            ],
            [0, 0],
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Build a Spec-K request policy")
    parser.add_argument("mode", choices=POLICIES)
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args()

    config_file = args.config_file.resolve()
    if not config_file.is_file():
        parser.error(f"policy implementation not found: {config_file}")

    config = {"file": str(config_file), **POLICIES[args.mode]}
    print(json.dumps(config, separators=(",", ":")))


if __name__ == "__main__":
    main()
