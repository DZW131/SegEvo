"""Entrypoint used by the `segevo-dashboard` console script."""

from __future__ import annotations

import argparse

from segevo.dashboard import run_dashboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dashboard(args.run)


main()

