"""Minimal explicit logging example.

Run:
    python examples/minimal_logging.py
    segevo-dashboard --run runs/minimal --port 7860
"""

from __future__ import annotations

import numpy as np

from segevo import SegEvoLogger


def main() -> None:
    logger = SegEvoLogger(
        "runs/minimal",
        manifest={
            "project": "minimal example",
            "classes": ["background", "target"],
        },
    )

    yy, xx = np.mgrid[:96, :96]
    image = np.sin(xx / 10.0) + np.cos(yy / 13.0)
    gt = ((yy - 48) ** 2 + (xx - 50) ** 2 < 20**2).astype(np.uint8)

    for epoch in range(5):
        radius = 26 - epoch * 2
        pred = ((yy - 46) ** 2 + (xx - 47) ** 2 < radius**2).astype(np.uint8)
        logger.log_case(epoch=epoch, case_id="case_001", image=image, gt=gt, pred=pred)


if __name__ == "__main__":
    main()

