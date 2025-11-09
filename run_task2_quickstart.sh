
#!/usr/bin/env bash
set -euo pipefail

# === Task 2 Quickstart: Stage A non-learned affine calibration ===
# 1) Render a tiny cache (few minibatches) to collect (rendered, GT) pairs
# 2) Solve per-camera 3x4 affine on linear RGB (robust)
# 3) Re-run short training with correction enabled

SCENE=${SCENE:-Skyfall-GS/data/datasets_JAX/JAX_068}
OUT=${OUT:-Skyfall-GS/output/task2_stageA_quick}
ITERS=${ITERS:-1000}
SAMPLES=${SAMPLES:-200000}

echo "[1/3] Collect samples (render cache)"
python -u photometric_affine_calib.py   --mode collect   --scene "$SCENE"   --out "$OUT/render_cache"   --max_batches 40

echo "[2/3] Solve per-camera affine"
python -u photometric_affine_calib.py   --mode solve   --cache "$OUT/render_cache"   --out "$OUT/affine_params.json"   --samples $SAMPLES

echo "[3/3] Short training with correction"
python -u Skyfall-GS/train.py   -s "$SCENE" -m "$OUT/train"   --iterations $ITERS   --photometric_affine 1   --photometric_affine_path "$OUT/affine_params.json"   --test_iterations 200 500 1000   --save_iterations 1000
