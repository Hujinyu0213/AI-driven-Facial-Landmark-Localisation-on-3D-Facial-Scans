"""
PointNet++ Regression Baseline — K-Fold + Patient-Based Split
==============================================================
Architecture : PointNet2RegMSG (MSG), output_dim = 27 (9 landmarks × 3)
Data         : data/pointcloud/<sample>/pointcloud_full.npy
               data/pointcloud/<sample>/nose_landmarks.npy  (last 9 of 21)
               data/pointcloud/<sample>/triangles.npy        (optional, for mesh-uniform sampling)
Preprocessing: PC centroid + std normalisation (same as all recent scripts)
Sampling     : mesh-uniform to 16 384 pts (fallback: GPU FPS)
Split        : patient-based 80/20 train/test  — extracts patient-ID from folder
               name prefix so that both scans of one patient always land in the
               same partition (no data leakage).
K-Fold       : 5-fold CV on patient level within the 80 % train pool.
Loss         : SmoothL1
Metric       : per-landmark & overall L2 error in mm (unscaled)
"""

import os
import sys
import json
import copy
import shutil
from datetime import datetime

# Set before any CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(os.path.dirname(BASE_DIR))
UTILS_DIR  = os.path.join(ROOT_DIR, "scripts", "utils")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
for p in (ROOT_DIR, UTILS_DIR, MODELS_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import shared data-loading utilities (same preprocessing as all recent scripts)
import scripts.training.train_script_pointnet2_c2f as base

from models.pointnet2_reg import PointNet2RegMSG

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NUM_LANDMARKS  = 9
OUTPUT_DIM     = NUM_LANDMARKS * 3   # 27
LANDMARK_NAMES = base.LANDMARK_NAMES  # ['glabella', 'nasion', ...]

K_FOLDS        = 5
RANDOM_SEED    = 42
TEST_FRACTION  = 0.20   # patient-level 80/20 split

BATCH_SIZE     = 8
NUM_EPOCHS     = 220
LEARNING_RATE  = 0.001
LR_DECAY_STEP  = 80
LR_DECAY_GAMMA = 0.7
DROPOUT_RATE   = 0.35

# Determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# Patient-based split helpers
# ══════════════════════════════════════════════════════════════════════════════

def patient_id_from_name(folder_name: str) -> str:
    """
    Extract patient ID (first token before first underscore).
    e.g.  'F001_NE00WH_F3D_THIBAULT'    -> 'F001'
          'F001_NE00WH_F3D_THIBAULT_HA' -> 'F001'
    """
    return folder_name.split("_")[0]


def patient_based_split(names, test_fraction=0.20, seed=42):
    """
    Split sample indices so that all scans of a patient go to the
    same partition.  Returns (train_indices, test_indices).
    """
    # Map patient -> list of sample indices
    patient_to_idx = defaultdict(list)
    for i, name in enumerate(names):
        pid = patient_id_from_name(name)
        patient_to_idx[pid].append(i)

    patients = sorted(patient_to_idx.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n_test_patients = max(1, int(len(patients) * test_fraction))
    test_patients  = set(patients[:n_test_patients])
    train_patients = set(patients[n_test_patients:])

    train_idx, test_idx = [], []
    for pid in patients:
        if pid in test_patients:
            test_idx.extend(patient_to_idx[pid])
        else:
            train_idx.extend(patient_to_idx[pid])

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def patient_kfold(names_subset, indices_subset, k=5, seed=42):
    """
    K-Fold on the *patient* level over a subset of samples.
    Yields (fold_train_indices, fold_val_indices) as positions into indices_subset.
    """
    patient_to_local = defaultdict(list)
    for local_i, global_i in enumerate(indices_subset):
        pid = patient_id_from_name(names_subset[global_i])
        patient_to_local[pid].append(local_i)

    patients = sorted(patient_to_local.keys())
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for tr_pats, val_pats in kf.split(patients):
        tr_local, val_local = [], []
        for p_i in tr_pats:
            tr_local.extend(patient_to_local[patients[p_i]])
        for p_i in val_pats:
            val_local.extend(patient_to_local[patients[p_i]])
        yield np.array(tr_local), np.array(val_local)


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation (vectorised, on CPU tensors)
# Input : batch_pc  (B, 3, N), batch_lbl (B, 9, 3)
# Output: same shapes
# ══════════════════════════════════════════════════════════════════════════════

def augment_batch(batch_pc, batch_lbl):
    B, _, N = batch_pc.shape
    dev = batch_pc.device

    # Random Z-axis rotation ±15°
    theta = torch.rand(B, device=dev) * (2 * np.pi / 12) - (np.pi / 12)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.zeros(B, 3, 3, device=dev)
    rot[:, 0, 0] = cos_t
    rot[:, 0, 1] = -sin_t
    rot[:, 1, 0] = sin_t
    rot[:, 1, 1] = cos_t
    rot[:, 2, 2] = 1.0

    pc_t  = batch_pc.transpose(1, 2)            # (B, N, 3)
    pc_r  = torch.bmm(pc_t, rot)                # (B, N, 3)
    lbl_r = torch.bmm(batch_lbl, rot)           # (B, 9, 3)

    # Random uniform scale 0.95–1.05
    scale = torch.rand(B, 1, 1, device=dev) * 0.10 + 0.95
    pc_r  = pc_r  * scale
    lbl_r = lbl_r * scale

    # Random translation ±0.02 (normalised units)
    shift = (torch.rand(B, 1, 3, device=dev) * 0.04) - 0.02
    pc_r  = pc_r  + shift
    lbl_r = lbl_r + shift

    # Gaussian jitter on points
    pc_r  = pc_r  + torch.randn(B, N, 3, device=dev) * 0.005

    return pc_r.transpose(1, 2), lbl_r          # (B,3,N), (B,9,3)


# ── Left-right flip augmentation (optional, prob=0.2) ─────────────────────
def maybe_flip(batch_pc, batch_lbl, prob=0.2):
    """Randomly flip along X axis; used as additional augment."""
    B = batch_pc.size(0)
    mask = (torch.rand(B) < prob).to(batch_pc.device)
    for b in range(B):
        if mask[b]:
            batch_pc[b, 0, :] *= -1
            batch_lbl[b, :, 0] *= -1
    return batch_pc, batch_lbl


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_mm(model, X_np, Y_np, scales_np):
    """
    Run inference on X_np (N, 3, P), un-normalise with scales_np (N,),
    return per-landmark L2 error in mm: shape (N, 9).
    """
    model.eval()
    with torch.no_grad():
        X_t   = torch.from_numpy(X_np).float().to(device)
        pred  = model(X_t).cpu().numpy()          # (N, 27)
    pred_lm = pred.reshape(-1, NUM_LANDMARKS, 3)  # (N, 9, 3)
    errs = np.linalg.norm(pred_lm - Y_np, axis=2) * scales_np[:, None]  # (N, 9) mm
    return errs


def evaluate_mm_with_snap_mesh(model, X_np, Y_np, scales_np, full_clouds, triangles):
    """
    Return tuple:
      - errs_raw  : (N, 9) mm
      - errs_snap : (N, 9) mm  (triangle-mesh projection; fallback to nearest point)
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_np).float().to(device)
        pred = model(X_t).cpu().numpy().reshape(-1, NUM_LANDMARKS, 3)  # (N,9,3)

    errs_raw = np.linalg.norm(pred - Y_np, axis=2) * scales_np[:, None]

    snapped = np.empty_like(pred, dtype=np.float32)
    for i in range(pred.shape[0]):
        pc_full = full_clouds[i]
        tri = triangles[i] if triangles is not None else None
        for k in range(NUM_LANDMARKS):
            snapped[i, k] = base.snap_to_mesh(pred[i, k], pc_full, tri)

    errs_snap = np.linalg.norm(snapped - Y_np, axis=2) * scales_np[:, None]
    return errs_raw, errs_snap


def print_eval(tag, errs):
    """Print per-landmark and overall statistics."""
    print(f"\n[{tag}]  N={errs.shape[0]}")
    print(f"  {'landmark':<12} {'mean':>7} {'median':>7} {'P90':>7}")
    print("  " + "-" * 38)
    for k, lm in enumerate(LANDMARK_NAMES):
        e = errs[:, k]
        print(f"  {lm:<12} {e.mean():>7.3f} {np.median(e):>7.3f} {np.percentile(e,90):>7.3f}")
    flat = errs.flatten()
    print(f"  {'OVERALL':<12} {flat.mean():>7.3f} {np.median(flat):>7.3f} {np.percentile(flat,90):>7.3f}  (mm)")


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_kfold():
    # ── 0. Print configuration ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PointNet++ Regression Baseline — Configuration")
    print("=" * 60)
    print(f"  {'Model':<22}: PointNet2RegMSG  output_dim={OUTPUT_DIM}")
    print(f"  {'Landmarks':<22}: {NUM_LANDMARKS}  {LANDMARK_NAMES}")
    print(f"  {'Max points':<22}: {base.MAX_POINTS}  (mesh-uniform; fallback FPS)")
    print(f"  {'Normalisation':<22}: PC centroid + std")
    print(f"  {'Sampling':<22}: mesh-uniform (triangles.npy) / FPS")
    print(f"  {'Device':<22}: {device}")
    print()
    print(f"  {'K-Folds':<22}: {K_FOLDS}")
    print(f"  {'Random seed':<22}: {RANDOM_SEED}")
    print(f"  {'Test fraction':<22}: {TEST_FRACTION:.0%}  (patient-based split)")
    print()
    print(f"  {'Batch size':<22}: {BATCH_SIZE}")
    print(f"  {'Epochs':<22}: {NUM_EPOCHS}")
    print(f"  {'Learning rate':<22}: {LEARNING_RATE}")
    print(f"  {'LR decay step':<22}: {LR_DECAY_STEP}  gamma={LR_DECAY_GAMMA}")
    print(f"  {'Dropout':<22}: {DROPOUT_RATE}")
    print(f"  {'Loss':<22}: SmoothL1")
    print(f"  {'Print interval':<22}: every 20 epochs")
    print("=" * 60 + "\n")

    # ── 1. Load data (same pipeline as all recent experiments) ────────────────
    X, Y, scales, names, full_clouds, triangles = base.load_data()
    # X: (N, 3, 16384)  Y: (N, 9, 3) normalised  scales: (N,) mm
    print(f"\nTotal samples loaded: {len(X)}")

    # ── 2. Patient-based 80/20 split ─────────────────────────────────────────
    train_idx, test_idx = patient_based_split(names, test_fraction=TEST_FRACTION, seed=RANDOM_SEED)
    print(f"Train scans: {len(train_idx)}, Test scans: {len(test_idx)}")

    # Print patient counts for verification
    train_patients = set(patient_id_from_name(names[i]) for i in train_idx)
    test_patients  = set(patient_id_from_name(names[i]) for i in test_idx)
    assert train_patients.isdisjoint(test_patients), "DATA LEAKAGE: patient in both splits!"
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")

    n_inner = max(2, min(K_FOLDS, len(train_patients)))
    if n_inner != K_FOLDS:
        print(f"[warn] K_FOLDS={K_FOLDS} > train_patients={len(train_patients)}; using n_splits={n_inner}")

    Xtr_full   = X[train_idx]
    Ytr_full   = Y[train_idx]
    Str_full   = scales[train_idx]
    names_arr  = np.array(names)

    X_test     = X[test_idx]
    Y_test     = Y[test_idx]
    S_test     = scales[test_idx]
    full_clouds_test = [full_clouds[i] for i in test_idx]
    triangles_test   = [triangles[i] for i in test_idx]

    # ── 3. 5-Fold cross-validation (patient-level) ────────────────────────────
    criterion        = nn.SmoothL1Loss()
    fold_results     = []
    all_histories    = []
    best_overall_mm  = float("inf")
    best_fold_idx    = -1

    # Live history persistence (crash-safe)
    result_dir = os.path.join(ROOT_DIR, "results", "training_histories")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "training_history_pointnet2_baseline_kfold.json")

    run_started_at = datetime.now().isoformat(timespec="seconds")

    def save_progress(status, current_fold=0, current_epoch=0, test_results=None):
        val_mms = [r["best_val_mm"] for r in fold_results] if fold_results else []
        payload = {
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_started_at": run_started_at,
            "current_fold": int(current_fold),
            "current_epoch": int(current_epoch),
            "model": "PointNet2RegMSG (baseline)",
            "seed": RANDOM_SEED,
            "split": f"patient-based {int((1-TEST_FRACTION)*100)}/{int(TEST_FRACTION*100)}",
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "lr_decay_step": LR_DECAY_STEP,
                "lr_decay_gamma": LR_DECAY_GAMMA,
                "batch_size": BATCH_SIZE,
                "dropout": DROPOUT_RATE,
                "num_epochs": NUM_EPOCHS,
                "loss": "SmoothL1",
                "max_points": base.MAX_POINTS,
                "sampling": "mesh-uniform (fallback: FPS)",
            },
            "k_folds": K_FOLDS,
            "k_folds_effective": int(n_inner),
            "fold_results": fold_results,
            "cv_summary": {
                "mean_val_mm": float(np.mean(val_mms)) if val_mms else None,
                "std_val_mm": float(np.std(val_mms)) if val_mms else None,
                "best_fold": int(best_fold_idx) if best_fold_idx > 0 else None,
                "best_val_mm": float(best_overall_mm) if best_fold_idx > 0 else None,
            },
            "test_results": test_results,
            "training_histories": all_histories,
        }
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, result_path)

    # create file immediately so progress exists even before first fold ends
    save_progress(status="running", current_fold=0, current_epoch=0)

    for fold_idx, (tr_loc, val_loc) in enumerate(
        patient_kfold(names_arr, train_idx, k=n_inner, seed=RANDOM_SEED), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{n_inner}  |  train={len(tr_loc)}  val={len(val_loc)}")
        print(f"{'='*60}")

        X_tr, Y_tr = Xtr_full[tr_loc],  Ytr_full[tr_loc]
        X_va, Y_va = Xtr_full[val_loc], Ytr_full[val_loc]
        S_va       = Str_full[val_loc]

        model = PointNet2RegMSG(output_dim=OUTPUT_DIM,
                                normal_channel=False,
                                dropout=DROPOUT_RATE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

        # Pre-move validation data to GPU
        X_va_t    = torch.from_numpy(X_va).float().to(device)

        # Training DataLoader (CPU tensors; augment on-the-fly)
        train_ds  = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(Y_tr).float()      # (N, 9, 3)
        )
        train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        best_val_mm  = float("inf")
        best_state   = None
        history      = {"epoch": [], "train_loss": [], "val_mm": [], "lr": []}

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss_acc, n_batches = 0.0, 0

            for batch_pc, batch_lbl in train_dl:
                # batch_lbl: (B, 9, 3)
                batch_pc, batch_lbl = augment_batch(batch_pc, batch_lbl)
                batch_pc, batch_lbl = maybe_flip(batch_pc, batch_lbl, prob=0.2)
                batch_pc  = batch_pc.to(device)
                batch_lbl_flat = batch_lbl.reshape(batch_lbl.size(0), -1).to(device)

                optimizer.zero_grad()
                pred = model(batch_pc)                  # (B, 27)
                loss = criterion(pred, batch_lbl_flat)
                loss.backward()
                optimizer.step()

                train_loss_acc += loss.item()
                n_batches      += 1

            scheduler.step()
            avg_train = train_loss_acc / max(n_batches, 1)

            # Validation: L2 mm error
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va_t).cpu().numpy()    # (N, 27)
            val_pred_lm = val_pred.reshape(-1, NUM_LANDMARKS, 3)
            val_errs    = np.linalg.norm(val_pred_lm - Y_va, axis=2) * S_va[:, None]
            val_mm      = float(val_errs.mean())

            history["epoch"].append(epoch)
            history["train_loss"].append(float(avg_train))
            history["val_mm"].append(val_mm)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            if val_mm < best_val_mm:
                best_val_mm = val_mm
                best_state  = copy.deepcopy(model.state_dict())

            if epoch % 20 == 0 or epoch == NUM_EPOCHS:
                print(f"  Epoch {epoch:>3}/{NUM_EPOCHS}  "
                      f"train_loss={avg_train:.5f}  val_mm={val_mm:.3f}  best={best_val_mm:.3f}")
                save_progress(status="running", current_fold=fold_idx, current_epoch=epoch)

        # Save per-fold best
        fold_path = os.path.join(MODELS_DIR, f"pointnet2_baseline_kfold_fold{fold_idx}_best.pth")
        torch.save(best_state, fold_path)
        print(f"  -> Fold {fold_idx} best val L2 = {best_val_mm:.3f} mm  saved: {fold_path}")

        # Quick per-landmark breakdown for this fold
        model.load_state_dict(best_state)
        val_errs_lm = evaluate_mm(model, X_va, Y_va, S_va)
        print_eval(f"Fold {fold_idx} val", val_errs_lm)

        fold_results.append({
            "fold": fold_idx,
            "best_val_mm": float(best_val_mm),
            "train_scans": int(len(tr_loc)),
            "val_scans":   int(len(val_loc)),
            "per_landmark_val_mm": val_errs_lm.mean(axis=0).tolist(),
        })
        all_histories.append(history)

        if best_val_mm < best_overall_mm:
            best_overall_mm  = best_val_mm
            best_fold_idx    = fold_idx

        # persist at fold boundary as well
        save_progress(status="running", current_fold=fold_idx, current_epoch=NUM_EPOCHS)

    # ── 4. Copy best fold model ───────────────────────────────────────────────
    best_fold_path    = os.path.join(MODELS_DIR, f"pointnet2_baseline_kfold_fold{best_fold_idx}_best.pth")
    best_overall_path = os.path.join(MODELS_DIR, "pointnet2_baseline_kfold_best.pth")
    shutil.copy(best_fold_path, best_overall_path)
    print(f"\nBest fold: {best_fold_idx}  ({best_overall_mm:.3f} mm) -> {best_overall_path}")

    # ── 5. Evaluate best model on held-out test set ───────────────────────────
    best_model = PointNet2RegMSG(output_dim=OUTPUT_DIM,
                                 normal_channel=False,
                                 dropout=DROPOUT_RATE).to(device)
    best_model.load_state_dict(torch.load(best_overall_path, map_location=device))
    test_errs_raw, test_errs_snap = evaluate_mm_with_snap_mesh(
        best_model,
        X_test,
        Y_test,
        S_test,
        full_clouds_test,
        triangles_test,
    )
    print_eval("TEST SET (held-out, raw)", test_errs_raw)
    print_eval("TEST SET (held-out, snap-mesh)", test_errs_snap)

    # ── 6. Save final JSON state ──────────────────────────────────────────────
    final_test_results = {
        "primary_metric": "snap_mesh",
        "n_scans": int(len(X_test)),
        "raw": {
            "overall_mean": float(test_errs_raw.mean()),
            "overall_median": float(np.median(test_errs_raw)),
            "overall_p90": float(np.percentile(test_errs_raw, 90)),
            "per_landmark": {
                lm: float(test_errs_raw[:, k].mean())
                for k, lm in enumerate(LANDMARK_NAMES)
            },
        },
        "snap_mesh": {
            "overall_mean": float(test_errs_snap.mean()),
            "overall_median": float(np.median(test_errs_snap)),
            "overall_p90": float(np.percentile(test_errs_snap, 90)),
            "per_landmark": {
                lm: float(test_errs_snap[:, k].mean())
                for k, lm in enumerate(LANDMARK_NAMES)
            },
        },
    }
    save_progress(
        status="completed",
        current_fold=n_inner,
        current_epoch=NUM_EPOCHS,
        test_results=final_test_results,
    )

    # ── 7. Final summary ──────────────────────────────────────────────────────
    val_mms = [r["best_val_mm"] for r in fold_results]
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — PointNet++ Baseline")
    print("=" * 60)
    print(f"  CV mean val : {np.mean(val_mms):.3f} ± {np.std(val_mms):.3f} mm")
    print(f"  Test raw     : {test_errs_raw.mean():.3f} mm  (median {np.median(test_errs_raw):.3f}, P90 {np.percentile(test_errs_raw,90):.3f})")
    print(f"  Test snapmesh: {test_errs_snap.mean():.3f} mm  (median {np.median(test_errs_snap):.3f}, P90 {np.percentile(test_errs_snap,90):.3f})")
    print(f"  Best model  : {best_overall_path}")
    print(f"  Results JSON: {result_path}")


if __name__ == "__main__":
    train_kfold()
