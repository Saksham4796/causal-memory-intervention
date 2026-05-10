from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.analysis.make_artifacts import make_all_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    make_all_artifacts(args.run_dir)
    print(f"Wrote paper artifacts from {args.run_dir}")


if __name__ == "__main__":
    main()
