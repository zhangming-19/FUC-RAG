#!/usr/bin/env python3

import os
import random
import runpy

import numpy as np
import torch

seed = int(os.environ.get("PARAMMUTE_EVAL_SEED", "42"))

print("=" * 70)
print(f"PARAMMUTE_EVAL_SEED={seed}")
print("=" * 70)

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Keep the official evaluator unchanged.
runpy.run_path(
    "src/3_evaluate/eval_CoConflictQA.py",
    run_name="__main__",
)
