"""Command line entrypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from segevo.demo import generate_demo_run


def demo_main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic SegEvo demo run.")
    parser.add_argument("--out", default="runs/demo", help="Output run directory.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of demo epochs.")
    parser.add_argument("--cases", type=int, default=3, help="Number of demo cases.")
    args = parser.parse_args()

    out = generate_demo_run(args.out, epochs=args.epochs, cases=args.cases)
    print(f"Demo run written to {out}")


def dashboard_main() -> None:
    parser = argparse.ArgumentParser(description="Start the SegEvo dashboard.")
    parser.add_argument("--run", required=True, help="SegEvo run directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host.")
    parser.add_argument("--port", type=int, default=7860, help="Dashboard port.")
    args = parser.parse_args()

    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install dashboard extras with: "
            'pip install -e ".[dashboard]"'
        ) from exc

    app_path = Path(__file__).with_name("_streamlit_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--",
        "--run",
        args.run,
    ]
    raise SystemExit(stcli.main())

