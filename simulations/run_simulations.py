#!/usr/bin/env python3
"""Run the editable paper-figure suite using the implementation in src."""

import os
import sys
from pathlib import Path

_threads = os.environ.get("SLURM_CPUS_PER_TASK", "4")
os.environ.setdefault("OMP_NUM_THREADS", _threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _threads)

_source = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_source))

from con_enh_quant_learning.paper_simulations import main  # noqa: E402

if __name__ == "__main__":
    main(default_config=Path(__file__).with_name("config.txt"))
