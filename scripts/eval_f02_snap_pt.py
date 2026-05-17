"""
Re-evaluate F02 seed=42 final model with soft-argmax / snap-pt / snap-mesh.
Note: F02 is a direct regression model (no heatmap), so soft-argmax == raw prediction.
"""
import sys, pathlib
import numpy as np
import torch

TRAINING_DIR = pathlib.Path(__file__).parent / "training"
PROJECT_ROOT = pathlib.Path(__file__).parents[1]           # PointFeatureProject/
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(TRAINING_DIR.parent))               # scripts/
sys.path.insert(0, str(PROJECT_ROOT))                      # for models/

import train_heatmap_joint_flip_v3 as base
from models.point_transformer_reg import PointTransformerReg

device = base.device
N_LANDMARKS = base.N_LANDMARKS
LANDMARK_NAMES = base.LANDMARK_NAMES

def build_model():
    return PointTransformerReg(
        output_dim=N_LANDMARKS * 3,
        dropout=0.3,
        dims=(128, 256, 512),
        n_pts=(2048, 512, 128),
        k=16,
        pre_fps_n=4096,
        normal_channel=False,
    ).to(device)


def predict(model, X, batch_size=8):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            out.append(model(xb).cpu().numpy())
    pred = np.concatenate(out, axis=0).reshape(-1, N_LANDMARKS, 3)
    return pred


def main():
    model_path = (
        pathlib.Path(__file__).parents[1]
        / "results/paper/F02/F02_pt_single_patient80_20_mesh16384_seed42"
        / "F02_pt_single_patient80_20_mesh16384_seed42_best_from_cv.pth"
    )

    # ── load data ────────────────────────────────────────────────────────────
    X, Y, scales, names, full_clouds, triangles = base.load_data()

    seed = 42
    rng = np.random.RandomState(seed)
    n_total = len(X)
    n_test = max(1, int(n_total * 0.20))
    test_idx = rng.choice(n_total, n_test, replace=False)

    Xte = X[test_idx]
    Yte = Y[test_idx]
    Ste = scales[test_idx]
    full_clouds_te = [full_clouds[i] for i in test_idx]
    triangles_te   = [triangles[i]   for i in test_idx] if triangles else None

    print(f"Test set: N={len(Xte)}")

    # ── load model ───────────────────────────────────────────────────────────
    model = build_model()
    sd = torch.load(model_path, map_location=device, weights_only=False)
    # F02 saves raw state_dict (no wrapper dict)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    print(f"Loaded model from {model_path}")

    # ── predict ──────────────────────────────────────────────────────────────
    pred = predict(model, Xte)

    # ── compute errors ───────────────────────────────────────────────────────
    # For direct regression, raw pred == soft-argmax (no heatmap decoding step)
    errs_raw     = np.linalg.norm(pred   - Yte, axis=2) * Ste[:, None]

    snap_pt_all  = np.empty_like(pred)
    snap_sf_all  = np.empty_like(pred)
    for i in range(len(Xte)):
        pc  = full_clouds_te[i]
        tri = triangles_te[i] if triangles_te else None
        for k in range(N_LANDMARKS):
            snap_pt_all[i, k] = base.snap_to_nearest(pred[i, k], pc)
            snap_sf_all[i, k] = base.snap_to_mesh(pred[i, k], pc, tri)

    errs_snap_pt = np.linalg.norm(snap_pt_all - Yte, axis=2) * Ste[:, None]
    errs_snap_sf = np.linalg.norm(snap_sf_all - Yte, axis=2) * Ste[:, None]

    # ── print summary ────────────────────────────────────────────────────────
    def stats(arr, label):
        flat = arr.flatten()
        print(f"  {label:<22} mean={flat.mean():.3f}  median={np.median(flat):.3f}  "
              f"P90={np.percentile(flat,90):.3f}")

    print(f"\n{'='*65}")
    print(f"  F02 seed=42  best_from_cv  (N={len(Xte)} scans)")
    print(f"{'='*65}")
    stats(errs_raw,     "soft-argmax")
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
