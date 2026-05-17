"""
5-fold Inner Ensemble evaluation for V1 seed=42.

Loads all 5 inner fold models, runs each on the holdout test set,
averages their soft-argmax (refined) coordinates, then computes
soft-argmax / snap-pt / snap-mesh errors.

Compares against the single retrain-final model for reference.
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
from scripts.training.train_script_pointtransformer_c2f import (
    UnifiedCoarseFineNet,
    predict_unified,
    RADIUS_MIN_BY_LM,
    RADIUS_MIN_DEFAULT,
)

device = base.device
N_LANDMARKS = base.N_LANDMARKS
LANDMARK_NAMES = base.LANDMARK_NAMES

# ── defaults (overridden from checkpoint config when loading) ─────────────────
SIGMA      = 0.04
PATCH_PTS  = 512   # matches training config (patch_points=512)
COND_DIM   = 128
COND_DROP  = 0.1

INNER_MODELS = [
    PROJECT_ROOT / f"models/joint_hybrid_pt_patch_floor_seed42_sigma0.04_inner{i}.pth"
    for i in range(1, 6)
]
FINAL_MODEL = PROJECT_ROOT / "models/joint_hybrid_pt_patch_floor_seed42_sigma0.04.pth"


def remap_impl_keys(sd):
    """
    Remap checkpoint keys saved with pointnet2_ops native extension (_impl structure)
    to the fallback pure-PyTorch structure (mlps/bns sequential indices).

    _impl.conv_blocks.X.Y.weight  ->  mlps.X.{3*Y}.weight
    _impl.bn_blocks.X.Y.*         ->  mlps.X.{3*Y+1}.*
    """
    new_sd = {}
    import re
    for k, v in sd.items():
        # skip conv bias — current model uses bias=False (BN follows)
        if re.match(r".+\._impl\.conv_blocks\.\d+\.\d+\.bias$", k):
            continue
        m_conv = re.match(r"(.+)\._impl\.conv_blocks\.(\d+)\.(\d+)\.weight$", k)
        m_bn   = re.match(r"(.+)\._impl\.bn_blocks\.(\d+)\.(\d+)\.(.+)$", k)
        if m_conv:
            prefix, x, y = m_conv.group(1), int(m_conv.group(2)), int(m_conv.group(3))
            new_sd[f"{prefix}.mlps.{x}.{3*y}.weight"] = v
        elif m_bn:
            prefix, x, y, suffix = m_bn.group(1), int(m_bn.group(2)), int(m_bn.group(3)), m_bn.group(4)
            if suffix == "num_batches_tracked":
                continue  # skip, not needed
            new_sd[f"{prefix}.mlps.{x}.{3*y+1}.{suffix}"] = v
        else:
            new_sd[k] = v
    return new_sd


def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # read hyper-params from checkpoint config if available
    cfg = ckpt.get("config", {})
    sigma       = cfg.get("sigma",       SIGMA)
    patch_pts   = cfg.get("patch_points", PATCH_PTS)
    cond_dim    = cfg.get("cond_dim",    COND_DIM)
    cond_drop   = cfg.get("cond_dropout", COND_DROP)
    use_cond    = not cfg.get("disable_conditioning", False)

    init_radii = [float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)) for lm in LANDMARK_NAMES]
    model = UnifiedCoarseFineNet(
        radii=init_radii,
        sigma=sigma,
        patch_points=patch_pts,
        s2_dropout=base.DROPOUT_S2,
        cond_dim=cond_dim,
        cond_dropout=cond_drop,
        use_conditioning=use_cond,
    ).to(device)

    # remap if saved with pointnet2_ops native extension
    if any("_impl" in k for k in sd):
        print(f"    (remapping _impl keys → mlps keys, patch_points={patch_pts})")
        sd = remap_impl_keys(sd)
    model.load_state_dict(sd)
    print(f"  Loaded: {pathlib.Path(path).name}  [sigma={sigma}, patch_pts={patch_pts}]")
    return model


def compute_errors(refined_pred, Yte, Ste, full_clouds_te, triangles_te):
    """Returns (errs_sa, errs_snap_pt, errs_snap_sf) each shape [N, 9]."""
    errs_sa = np.linalg.norm(refined_pred - Yte, axis=2) * Ste[:, None]

    snap_pt_all = np.empty_like(refined_pred)
    snap_sf_all = np.empty_like(refined_pred)
    for i in range(len(refined_pred)):
        pc  = full_clouds_te[i]
        tri = triangles_te[i] if triangles_te else None
        for k in range(N_LANDMARKS):
            snap_pt_all[i, k] = base.snap_to_nearest(refined_pred[i, k], pc)
            snap_sf_all[i, k] = base.snap_to_mesh(refined_pred[i, k], pc, tri)

    errs_snap_pt = np.linalg.norm(snap_pt_all - Yte, axis=2) * Ste[:, None]
    errs_snap_sf = np.linalg.norm(snap_sf_all - Yte, axis=2) * Ste[:, None]
    return errs_sa, errs_snap_pt, errs_snap_sf


def print_table(errs_sa, errs_snap_pt, errs_snap_sf, label):
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    print(f"  {'Landmark':<18}  {'soft-argmax':>11}  {'snap-pt':>9}  {'snap-mesh':>9}")
    print(f"  {'─'*18}  {'─'*11}  {'─'*9}  {'─'*9}")
    for k, name in enumerate(LANDMARK_NAMES):
        print(f"  {name:<18}  {errs_sa[:,k].mean():>11.3f}  "
              f"{errs_snap_pt[:,k].mean():>9.3f}  {errs_snap_sf[:,k].mean():>9.3f}")
    print(f"  {'─'*18}  {'─'*11}  {'─'*9}  {'─'*9}")
    print(f"  {'OVERALL':<18}  {errs_sa.mean():>11.3f}  "
          f"{errs_snap_pt.mean():>9.3f}  {errs_snap_sf.mean():>9.3f}")
    print(f"{'─'*62}")


def main():
    # ── load data ─────────────────────────────────────────────────────────────
    X, Y, scales, names, full_clouds, triangles = base.load_data()

    seed = 42
    rng  = np.random.RandomState(seed)
    n_total = len(X)
    n_test  = max(1, int(n_total * 0.20))
    test_idx = rng.choice(n_total, n_test, replace=False)

    Xte            = X[test_idx]
    Yte            = Y[test_idx]
    Ste            = scales[test_idx]
    full_clouds_te = [full_clouds[i] for i in test_idx]
    triangles_te   = [triangles[i] for i in test_idx] if triangles else None
    print(f"Holdout test set: N={len(Xte)}")

    # ── inner ensemble ────────────────────────────────────────────────────────
    print("\n[Ensemble] Loading 5 inner fold models...")
    all_refined = []
    for path in INNER_MODELS:
        model = load_model(path)
        _, refined = predict_unified(model, Xte, batch_size=8)
        all_refined.append(refined)
        del model
        torch.cuda.empty_cache()

    ensemble_refined = np.mean(all_refined, axis=0)  # [N, 9, 3]
    print("  Ensemble prediction ready (mean of 5 fold predictions)")

    e_sa, e_spt, e_ssf = compute_errors(ensemble_refined, Yte, Ste, full_clouds_te, triangles_te)
    print_table(e_sa, e_spt, e_ssf, "5-FOLD INNER ENSEMBLE (seed=42)")

    # ── single final model (retrain on full train set) ────────────────────────
    print("\n[Single] Loading final retrain model...")
    final_model = load_model(FINAL_MODEL)
    _, final_refined = predict_unified(final_model, Xte, batch_size=8)
    f_sa, f_spt, f_ssf = compute_errors(final_refined, Yte, Ste, full_clouds_te, triangles_te)
    print_table(f_sa, f_spt, f_ssf, "SINGLE RETRAIN MODEL (seed=42)")

    # ── comparison ────────────────────────────────────────────────────────────
    print("\n[Comparison]")
    print(f"  {'Method':<32}  {'soft-argmax':>11}  {'snap-pt':>9}  {'snap-mesh':>9}")
    print(f"  {'─'*32}  {'─'*11}  {'─'*9}  {'─'*9}")
    print(f"  {'5-fold ensemble':<32}  {e_sa.mean():>11.3f}  {e_spt.mean():>9.3f}  {e_ssf.mean():>9.3f}")
    print(f"  {'Single retrain (full data)':<32}  {f_sa.mean():>11.3f}  {f_spt.mean():>9.3f}  {f_ssf.mean():>9.3f}")
    print(f"  {'Difference (ensemble - single)':<32}  {e_sa.mean()-f_sa.mean():>+11.3f}  "
          f"{e_spt.mean()-f_spt.mean():>+9.3f}  {e_ssf.mean()-f_ssf.mean():>+9.3f}")


if __name__ == "__main__":
    main()
