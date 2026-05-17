"""
Re-evaluate PointNet++ baseline (seed=42, best_from_cv) with
soft-argmax (= raw), snap-pt, and snap-mesh metrics.
"""
import sys, pathlib
import numpy as np
import torch

TRAINING_DIR = pathlib.Path(__file__).parent / "training"
PROJECT_ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(TRAINING_DIR.parent))
sys.path.insert(0, str(PROJECT_ROOT))

import train_heatmap_joint_flip_v3 as base

# The checkpoint was saved when pointnet2_ops was NOT installed (legacy fallback
# with _impl keys).  Block the package so the model rebuilds identical key names.
import sys as _sys
_sys.modules["pointnet2_ops"] = None                       # causes ImportError on import
_sys.modules["pointnet2_ops.pointnet2_modules"] = None
for _k in list(_sys.modules):                              # force re-import of model
    if "pointnet2_reg" in _k:
        del _sys.modules[_k]

from models.pointnet2_reg import PointNet2RegMSG

device       = base.device
N_LANDMARKS  = base.N_LANDMARKS
LANDMARK_NAMES = base.LANDMARK_NAMES
OUTPUT_DIM   = N_LANDMARKS * 3


def build_model():
    return PointNet2RegMSG(output_dim=OUTPUT_DIM, normal_channel=False).to(device)


def predict(model, X, batch_size=8):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1, N_LANDMARKS, 3)


def main():
    model_path = PROJECT_ROOT / "models" / "pointnet2_baseline_kfold_best.pth"

    # ── load data ─────────────────────────────────────────────────────────
    X, Y, scales, names, full_clouds, triangles = base.load_data()

    # same seed=42 80/20 patient holdout
    rng = np.random.RandomState(42)
    n_test = max(1, int(len(X) * 0.20))
    test_idx = rng.choice(len(X), n_test, replace=False)

    Xte = X[test_idx]
    Yte = Y[test_idx]
    Ste = scales[test_idx]
    full_clouds_te = [full_clouds[i] for i in test_idx]
    triangles_te   = [triangles[i]   for i in test_idx] if triangles else None

    print(f"Test set: N={len(Xte)}")

    # ── load model ────────────────────────────────────────────────────────
    model = build_model()
    sd = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    print(f"Loaded: {model_path}")

    # ── predict ───────────────────────────────────────────────────────────
    pred = predict(model, Xte)

    # ── errors ────────────────────────────────────────────────────────────
    # PointNet++ is direct regression → raw == soft-argmax
    errs_raw = np.linalg.norm(pred - Yte, axis=2) * Ste[:, None]

    snap_pt_all = np.empty_like(pred)
    snap_sf_all = np.empty_like(pred)
    for i in range(len(Xte)):
        pc  = full_clouds_te[i]
        tri = triangles_te[i] if triangles_te else None
        for k in range(N_LANDMARKS):
            snap_pt_all[i, k] = base.snap_to_nearest(pred[i, k], pc)
            snap_sf_all[i, k] = base.snap_to_mesh(pred[i, k], pc, tri)

    errs_snap_pt = np.linalg.norm(snap_pt_all - Yte, axis=2) * Ste[:, None]
    errs_snap_sf = np.linalg.norm(snap_sf_all - Yte, axis=2) * Ste[:, None]

    # ── print ─────────────────────────────────────────────────────────────
    def stats(arr, label):
        f = arr.flatten()
        print(f"  {label:<26} mean={f.mean():.3f}  median={np.median(f):.3f}  P90={np.percentile(f,90):.3f}")

    print(f"\n{'='*70}")
    print(f"  PointNet++ baseline  seed=42  best_from_cv  (N={len(Xte)} scans)")
    print(f"{'='*70}")
    stats(errs_raw,     "soft-argmax (= raw)")
    stats(errs_snap_pt, "snap-pt (nearest)")
    stats(errs_snap_sf, "snap-mesh")

    print(f"\n  {'landmark':12s}  {'soft-argmax':>12}  {'snap-pt':>8}  {'snap-mesh':>10}")
    print("  " + "-"*50)
    for k, lm in enumerate(LANDMARK_NAMES):
        r  = errs_raw[:, k].mean()
        sp = errs_snap_pt[:, k].mean()
        sm = errs_snap_sf[:, k].mean()
        print(f"  {lm:12s}  {r:12.3f}  {sp:8.3f}  {sm:10.3f}")
    print()


if __name__ == "__main__":
    main()
