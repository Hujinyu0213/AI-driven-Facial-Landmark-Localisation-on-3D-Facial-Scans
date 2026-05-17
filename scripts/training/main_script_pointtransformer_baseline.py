"""
PointTransformer Regression Baseline — K-Fold + Patient-Based Split
===================================================================
Single-stage PointTransformer baseline for 9-landmark regression.

Design goals:
- Keep baseline simple (no hybrid patch refiner / no cascade)
- Keep split leakage-safe (patient-based holdout + patient-level CV)
- Keep training stable (grad clip, full-epoch schedule)
- Keep outputs reproducible and resumable (live JSON snapshots)

Default baseline profile (tuned for your project):
- epochs=200
- lr=3e-4
- dims=128,256,512  (PT v2-style capacity used in your prior PT references)
- n_pts=2048,512,128; k=16; pre_fps_n=4096
"""

import os
import sys
import json
import copy
import shutil
import csv
import random
from datetime import datetime
from collections import defaultdict
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
for p in (ROOT_DIR, UTILS_DIR, MODELS_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import scripts.training.train_script_pointnet2_c2f as base
from models.point_transformer_reg import PointTransformerReg

NUM_LANDMARKS = 9
OUTPUT_DIM = NUM_LANDMARKS * 3
LANDMARK_NAMES = base.LANDMARK_NAMES
ALARE_R_IDX, ALARE_L_IDX = 5, 6
ZYGION_R_IDX, ZYGION_L_IDX = 7, 8

# Determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_model(args):
    return PointTransformerReg(
        output_dim=OUTPUT_DIM,
        dropout=args.dropout,
        dims=args.pt_dims,
        n_pts=args.pt_npts,
        k=args.pt_k,
        pre_fps_n=args.pt_pre_fps_n,
        normal_channel=False,
    ).to(device)


def patient_id_from_name(folder_name: str) -> str:
    return folder_name.split("_")[0]


def patient_based_split(names, test_fraction=0.20, seed=42):
    patient_to_idx = defaultdict(list)
    for i, name in enumerate(names):
        patient_to_idx[patient_id_from_name(name)].append(i)

    patients = sorted(patient_to_idx.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n_test_patients = max(1, int(len(patients) * test_fraction))
    test_patients = set(patients[:n_test_patients])

    train_idx, test_idx = [], []
    for pid in patients:
        if pid in test_patients:
            test_idx.extend(patient_to_idx[pid])
        else:
            train_idx.extend(patient_to_idx[pid])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def patient_kfold(names_subset, indices_subset, k=5, seed=42):
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


def augment_batch(batch_pc, batch_lbl):
    B, _, N = batch_pc.shape
    dev = batch_pc.device

    theta = torch.rand(B, device=dev) * (2 * np.pi / 12) - (np.pi / 12)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.zeros(B, 3, 3, device=dev)
    rot[:, 0, 0] = cos_t
    rot[:, 0, 1] = -sin_t
    rot[:, 1, 0] = sin_t
    rot[:, 1, 1] = cos_t
    rot[:, 2, 2] = 1.0

    pc_t = batch_pc.transpose(1, 2)
    pc_r = torch.bmm(pc_t, rot)
    lbl_r = torch.bmm(batch_lbl, rot)

    scale = torch.rand(B, 1, 1, device=dev) * 0.10 + 0.95
    pc_r = pc_r * scale
    lbl_r = lbl_r * scale

    shift = (torch.rand(B, 1, 3, device=dev) * 0.04) - 0.02
    pc_r = pc_r + shift
    lbl_r = lbl_r + shift

    pc_r = pc_r + torch.randn(B, N, 3, device=dev) * 0.005
    return pc_r.transpose(1, 2), lbl_r


def maybe_flip(batch_pc, batch_lbl, prob=0.2):
    B = batch_pc.size(0)
    mask = (torch.rand(B, device=batch_pc.device) < prob)
    for b in range(B):
        if bool(mask[b]):
            batch_pc[b, 0, :] *= -1
            batch_lbl[b, :, 0] *= -1
            saved_alare_r = batch_lbl[b, ALARE_R_IDX].clone()
            batch_lbl[b, ALARE_R_IDX] = batch_lbl[b, ALARE_L_IDX]
            batch_lbl[b, ALARE_L_IDX] = saved_alare_r
            saved_zygion_r = batch_lbl[b, ZYGION_R_IDX].clone()
            batch_lbl[b, ZYGION_R_IDX] = batch_lbl[b, ZYGION_L_IDX]
            batch_lbl[b, ZYGION_L_IDX] = saved_zygion_r
    return batch_pc, batch_lbl


def evaluate_mm(model, X_np, Y_np, scales_np, batch_size=8):
    pred = predict_in_batches(model, X_np, batch_size=batch_size)
    pred_lm = pred.reshape(-1, NUM_LANDMARKS, 3)
    return np.linalg.norm(pred_lm - Y_np, axis=2) * scales_np[:, None]


def evaluate_mm_with_snap_mesh(model, X_np, Y_np, scales_np, full_clouds, triangles, batch_size=8):
    pred = predict_in_batches(model, X_np, batch_size=batch_size).reshape(-1, NUM_LANDMARKS, 3)

    errs_raw = np.linalg.norm(pred - Y_np, axis=2) * scales_np[:, None]

    snapped = np.empty_like(pred, dtype=np.float32)
    for i in range(pred.shape[0]):
        pc_full = full_clouds[i]
        tri = triangles[i] if triangles is not None else None
        for k in range(NUM_LANDMARKS):
            snapped[i, k] = base.snap_to_mesh(pred[i, k], pc_full, tri)

    errs_snap = np.linalg.norm(snapped - Y_np, axis=2) * scales_np[:, None]
    return errs_raw, errs_snap


def predict_in_batches(model, X_np, batch_size=8):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[start:start + batch_size]).float().to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def print_eval(tag, errs):
    print(f"\n[{tag}]  N={errs.shape[0]}")
    print(f"  {'landmark':<12} {'mean':>7} {'median':>7} {'P90':>7}")
    print("  " + "-" * 38)
    for k, lm in enumerate(LANDMARK_NAMES):
        e = errs[:, k]
        print(f"  {lm:<12} {e.mean():>7.3f} {np.median(e):>7.3f} {np.percentile(e,90):>7.3f}")
    flat = errs.flatten()
    print(f"  {'OVERALL':<12} {flat.mean():>7.3f} {np.median(flat):>7.3f} {np.percentile(flat,90):>7.3f}  (mm)")


def train_kfold(args):
    set_global_seed(int(args.seed))

    print("\n" + "=" * 68)
    print("  PointTransformer Baseline — Configuration")
    print("=" * 68)
    print(f"  {'Model':<26}: PointTransformerReg output_dim={OUTPUT_DIM}")
    print(f"  {'PT dims':<26}: {args.pt_dims}")
    print(f"  {'PT n_pts':<26}: {args.pt_npts}")
    print(f"  {'PT k / pre_fps_n':<26}: {args.pt_k} / {args.pt_pre_fps_n}")
    print(f"  {'Device':<26}: {device}")
    print(f"  {'K-Folds':<26}: {args.k_folds}")
    print(f"  {'Seed':<26}: {args.seed}")
    print(f"  {'Split':<26}: patient-based {int((1-args.test_fraction)*100)}/{int(args.test_fraction*100)}")
    print(f"  {'Batch / Epochs':<26}: {args.batch_size} / {args.epochs}")
    print(f"  {'LR / step / gamma':<26}: {args.lr} / {args.lr_decay_step} / {args.lr_decay_gamma}")
    print(f"  {'Grad clip norm':<26}: {args.grad_clip_norm}")
    print(f"  {'Early stop':<26}: disabled (full epochs)")
    print(f"  {'Loss':<26}: SmoothL1")
    print(f"  {'Print interval':<26}: every {args.print_interval} epochs")
    print(f"  {'Run tag':<26}: {args.run_tag if args.run_tag else '(none)'}")
    print("=" * 68 + "\n")

    X, Y, scales, names, full_clouds, triangles = base.load_data()
    print(f"Total samples loaded: {len(X)}")

    train_idx, test_idx = patient_based_split(names, test_fraction=args.test_fraction, seed=args.seed)
    print(f"Train scans: {len(train_idx)}, Test scans: {len(test_idx)}")

    train_patients = set(patient_id_from_name(names[i]) for i in train_idx)
    test_patients = set(patient_id_from_name(names[i]) for i in test_idx)
    assert train_patients.isdisjoint(test_patients), "DATA LEAKAGE: patient in both splits!"
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")

    n_inner = max(2, min(args.k_folds, len(train_patients)))
    if n_inner != args.k_folds:
        print(f"[warn] k_folds={args.k_folds} > train_patients={len(train_patients)}; using n_splits={n_inner}")

    Xtr_full = X[train_idx]
    Ytr_full = Y[train_idx]
    Str_full = scales[train_idx]
    names_arr = np.array(names)

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    S_test = scales[test_idx]
    full_clouds_test = [full_clouds[i] for i in test_idx]
    triangles_test = [triangles[i] for i in test_idx] if triangles is not None else None

    criterion = nn.SmoothL1Loss()
    fold_results = []
    all_histories = []
    best_overall_mm = float("inf")
    best_fold_idx = -1

    split_tag = f"patient{int((1-args.test_fraction)*100)}_{int(args.test_fraction*100)}"
    exp_name = args.exp_name or f"{args.exp_id}_pt_single_{split_tag}_mesh16384"
    seed_tag = f"seed{args.seed}"
    run_name = f"{exp_name}_{seed_tag}"

    run_dir = os.path.join(ROOT_DIR, "results", "paper", args.exp_id, run_name)
    os.makedirs(run_dir, exist_ok=True)

    history_path = os.path.join(run_dir, "training_history.json")
    result_path = os.path.join(run_dir, "holdout_summary.json")
    innercv_csv_path = os.path.join(run_dir, "innercv_summary.csv")
    holdout_lm_csv_path = os.path.join(run_dir, "holdout_per_landmark.csv")
    holdout_sample_csv_path = os.path.join(run_dir, "holdout_per_sample.csv")

    run_started_at = datetime.now().isoformat(timespec="seconds")

    def save_progress(status, current_fold=0, current_epoch=0, test_results=None):
        val_mms = [r["best_val_mm"] for r in fold_results] if fold_results else []
        payload = {
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_started_at": run_started_at,
            "current_fold": int(current_fold),
            "current_epoch": int(current_epoch),
            "experiment_id": args.exp_id,
            "experiment_name": exp_name,
            "run_name": run_name,
            "model": "PointTransformerReg (baseline)",
            "seed": args.seed,
            "split": f"patient-based {int((1-args.test_fraction)*100)}/{int(args.test_fraction*100)}",
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "run_tag": args.run_tag,
            "output_dir": run_dir,
            "hyperparameters": {
                "learning_rate": args.lr,
                "lr_decay_step": args.lr_decay_step,
                "lr_decay_gamma": args.lr_decay_gamma,
                "batch_size": args.batch_size,
                "dropout": args.dropout,
                "num_epochs": args.epochs,
                "loss": "SmoothL1",
                "grad_clip_norm": args.grad_clip_norm,
                "early_stopping": False,
                "max_points": base.MAX_POINTS,
                "sampling": "mesh-uniform (fallback: FPS)",
                "pt_dims": list(args.pt_dims),
                "pt_n_pts": list(args.pt_npts),
                "pt_k": args.pt_k,
                "pt_pre_fps_n": args.pt_pre_fps_n,
                "eval_batch_size": args.eval_batch_size,
                "retrain_use_cv_epoch": bool(args.retrain_use_cv_epoch),
            },
            "k_folds": args.k_folds,
            "k_folds_effective": int(n_inner),
            "fold_results": fold_results,
            "cv_summary": {
                "mean_val_mm": float(np.mean(val_mms)) if val_mms else None,
                "std_val_mm": float(np.std(val_mms)) if val_mms else None,
                "best_fold": int(best_fold_idx) if best_fold_idx != -1 else None,
                "best_val_mm": float(best_overall_mm) if best_fold_idx != -1 else None,
            },
            "test_results": test_results,
            "training_histories": all_histories,
        }
        tmp_path = history_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, history_path)

    save_progress(status="running", current_fold=0, current_epoch=0)

    for fold_idx, (tr_loc, val_loc) in enumerate(
        patient_kfold(names_arr, train_idx, k=n_inner, seed=args.seed), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{n_inner}  |  train={len(tr_loc)}  val={len(val_loc)}")
        print(f"{'='*60}")

        X_tr, Y_tr = Xtr_full[tr_loc], Ytr_full[tr_loc]
        X_va, Y_va = Xtr_full[val_loc], Ytr_full[val_loc]
        S_va = Str_full[val_loc]

        model = build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.lr_decay_step),
            gamma=args.lr_decay_gamma,
        )

        train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).float())
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

        best_val_mm = float("inf")
        best_state = None
        best_epoch = -1
        history = {"epoch": [], "train_loss": [], "val_mm": [], "lr": []}

        for epoch in range(1, args.epochs + 1):
            model.train()
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss_acc, n_batches = 0.0, 0

            for batch_pc, batch_lbl in train_dl:
                batch_pc, batch_lbl = augment_batch(batch_pc, batch_lbl)
                batch_pc, batch_lbl = maybe_flip(batch_pc, batch_lbl, prob=args.flip_prob)
                batch_pc = batch_pc.to(device)
                batch_lbl_flat = batch_lbl.reshape(batch_lbl.size(0), -1).to(device)

                optimizer.zero_grad()
                pred = model(batch_pc)
                loss = criterion(pred, batch_lbl_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()

                train_loss_acc += loss.item()
                n_batches += 1

            avg_train = train_loss_acc / max(n_batches, 1)
            val_errs = evaluate_mm(model, X_va, Y_va, S_va, batch_size=args.eval_batch_size)
            val_mm = float(val_errs.mean())

            history["epoch"].append(epoch)
            history["train_loss"].append(float(avg_train))
            history["val_mm"].append(val_mm)
            history["lr"].append(current_lr)

            scheduler.step()

            if val_mm < best_val_mm:
                best_val_mm = val_mm
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if epoch % args.print_interval == 0 or epoch == args.epochs:
                print(
                    f"  Epoch {epoch:>3}/{args.epochs}  "
                    f"train_loss={avg_train:.5f}  val_mm={val_mm:.3f}  best={best_val_mm:.3f}"
                )
                save_progress(status="running", current_fold=fold_idx, current_epoch=epoch)

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = history["epoch"][-1] if history["epoch"] else 0

        model_prefix = run_name
        fold_path = os.path.join(run_dir, f"{model_prefix}_fold{fold_idx}_best.pth")
        torch.save(best_state, fold_path)
        print(f"  -> Fold {fold_idx} best val L2 = {best_val_mm:.3f} mm  saved: {fold_path}")

        model.load_state_dict(best_state)
        val_errs_lm = evaluate_mm(model, X_va, Y_va, S_va, batch_size=args.eval_batch_size)
        print_eval(f"Fold {fold_idx} val", val_errs_lm)

        fold_results.append(
            {
                "fold": fold_idx,
                "best_val_mm": float(best_val_mm),
                "best_epoch": int(best_epoch),
                "train_scans": int(len(tr_loc)),
                "val_scans": int(len(val_loc)),
                "per_landmark_val_mm": val_errs_lm.mean(axis=0).tolist(),
            }
        )
        all_histories.append(history)

        if best_val_mm < best_overall_mm:
            best_overall_mm = best_val_mm
            best_fold_idx = fold_idx

        save_progress(status="running", current_fold=fold_idx, current_epoch=args.epochs)

    model_prefix = run_name
    best_fold_path = os.path.join(run_dir, f"{model_prefix}_fold{best_fold_idx}_best.pth")
    cv_selected_model_path = os.path.join(run_dir, f"{model_prefix}_best_from_cv.pth")
    shutil.copy(best_fold_path, cv_selected_model_path)
    print(f"\nBest fold: {best_fold_idx}  ({best_overall_mm:.3f} mm) -> {cv_selected_model_path}")

    print("\n" + "=" * 60)
    print("  POST-CV RETRAIN ON FULL TRAIN SPLIT")
    print("=" * 60)
    cv_best_epochs = [int(r["best_epoch"]) for r in fold_results if int(r["best_epoch"]) > 0]
    if args.retrain_use_cv_epoch and len(cv_best_epochs) > 0:
        retrain_epochs = int(np.clip(int(round(float(np.mean(cv_best_epochs)))), 1, args.epochs))
    else:
        retrain_epochs = int(args.epochs)
    print(f"  Retrain epochs: {retrain_epochs} (cv_best_epochs={cv_best_epochs})")

    full_model = build_model(args)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, args.lr_decay_step),
        gamma=args.lr_decay_gamma,
    )
    train_full_ds = TensorDataset(torch.from_numpy(Xtr_full).float(), torch.from_numpy(Ytr_full).float())
    train_full_dl = DataLoader(train_full_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    retrain_history = {"phase": "retrain_full_train", "epoch": [], "train_loss": [], "lr": []}

    for epoch in range(1, retrain_epochs + 1):
        full_model.train()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss_acc, n_batches = 0.0, 0

        for batch_pc, batch_lbl in train_full_dl:
            batch_pc, batch_lbl = augment_batch(batch_pc, batch_lbl)
            batch_pc, batch_lbl = maybe_flip(batch_pc, batch_lbl, prob=args.flip_prob)
            batch_pc = batch_pc.to(device)
            batch_lbl_flat = batch_lbl.reshape(batch_lbl.size(0), -1).to(device)

            optimizer.zero_grad()
            pred = full_model(batch_pc)
            loss = criterion(pred, batch_lbl_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()

            train_loss_acc += loss.item()
            n_batches += 1

        avg_train = train_loss_acc / max(n_batches, 1)
        retrain_history["epoch"].append(epoch)
        retrain_history["train_loss"].append(float(avg_train))
        retrain_history["lr"].append(current_lr)

        scheduler.step()

        if epoch % args.print_interval == 0 or epoch == retrain_epochs:
            print(f"  [retrain] Epoch {epoch:>3}/{retrain_epochs}  train_loss={avg_train:.5f}")
            save_progress(status="running", current_fold=n_inner + 1, current_epoch=epoch)

    all_histories.append(retrain_history)

    retrain_final_path = os.path.join(run_dir, f"{model_prefix}_retrain_fulltrain_final.pth")
    torch.save(full_model.state_dict(), retrain_final_path)
    print(f"Retrained full-train model saved: {retrain_final_path}")

    best_model = build_model(args)
    try:
        state_dict = torch.load(retrain_final_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(retrain_final_path, map_location=device)
    best_model.load_state_dict(state_dict)

    test_errs_raw, test_errs_snap = evaluate_mm_with_snap_mesh(
        best_model,
        X_test,
        Y_test,
        S_test,
        full_clouds_test,
        triangles_test,
        batch_size=args.eval_batch_size,
    )

    print_eval("TEST SET (held-out, raw)", test_errs_raw)
    print_eval("TEST SET (held-out, snap-mesh)", test_errs_snap)

    final_test_results = {
        "primary_metric": "snap_mesh",
        "n_scans": int(len(X_test)),
        "raw": {
            "overall_mean": float(test_errs_raw.mean()),
            "overall_median": float(np.median(test_errs_raw)),
            "overall_p90": float(np.percentile(test_errs_raw, 90)),
            "per_landmark": {lm: float(test_errs_raw[:, k].mean()) for k, lm in enumerate(LANDMARK_NAMES)},
        },
        "snap_mesh": {
            "overall_mean": float(test_errs_snap.mean()),
            "overall_median": float(np.median(test_errs_snap)),
            "overall_p90": float(np.percentile(test_errs_snap, 90)),
            "per_landmark": {lm: float(test_errs_snap[:, k].mean()) for k, lm in enumerate(LANDMARK_NAMES)},
        },
    }
    final_test_results["overall_mean"] = final_test_results["snap_mesh"]["overall_mean"]
    final_test_results["overall_median"] = final_test_results["snap_mesh"]["overall_median"]
    final_test_results["overall_p90"] = final_test_results["snap_mesh"]["overall_p90"]
    final_test_results["per_landmark"] = final_test_results["snap_mesh"]["per_landmark"]

    with open(innercv_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_epoch", "best_val_mm", "train_scans", "val_scans"])
        for fr in fold_results:
            writer.writerow([fr["fold"], fr["best_epoch"], fr["best_val_mm"], fr["train_scans"], fr["val_scans"]])

    with open(holdout_lm_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark", "raw_mean_mm", "snap_mesh_mean_mm"])
        for k, lm in enumerate(LANDMARK_NAMES):
            writer.writerow([lm, float(test_errs_raw[:, k].mean()), float(test_errs_snap[:, k].mean())])

    with open(holdout_sample_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "raw_mean_mm", "snap_mesh_mean_mm"])
        for i in range(test_errs_raw.shape[0]):
            writer.writerow([i, float(test_errs_raw[i].mean()), float(test_errs_snap[i].mean())])

    holdout_payload = {
        "experiment_id": args.exp_id,
        "experiment_name": exp_name,
        "run_name": run_name,
        "seed": int(args.seed),
        "split": f"patient-based {int((1-args.test_fraction)*100)}/{int(args.test_fraction*100)}",
        "k_folds": int(args.k_folds),
        "k_folds_effective": int(n_inner),
        "best_fold": int(best_fold_idx),
        "retrain_epochs": int(retrain_epochs),
        "evaluation_model": "retrain_full_train",
        "cv_selected_model_path": cv_selected_model_path,
        "final_retrained_model_path": retrain_final_path,
        "test_results": final_test_results,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(holdout_payload, f, indent=2, ensure_ascii=False)

    save_progress(
        status="completed",
        current_fold=n_inner,
        current_epoch=retrain_epochs,
        test_results=final_test_results,
    )

    val_mms = [r["best_val_mm"] for r in fold_results]
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — PointTransformer Baseline")
    print("=" * 60)
    print(f"  CV mean val : {np.mean(val_mms):.3f} ± {np.std(val_mms):.3f} mm")
    print(
        f"  Test raw     : {test_errs_raw.mean():.3f} mm  "
        f"(median {np.median(test_errs_raw):.3f}, P90 {np.percentile(test_errs_raw,90):.3f})"
    )
    print(
        f"  Test snapmesh: {test_errs_snap.mean():.3f} mm  "
        f"(median {np.median(test_errs_snap):.3f}, P90 {np.percentile(test_errs_snap,90):.3f})"
    )
    print(f"  Best model  : {retrain_final_path}")
    print(f"  Run dir     : {run_dir}")
    print(f"  History JSON: {history_path}")
    print(f"  Results JSON: {result_path}")


def get_config():
    cfg = SimpleNamespace(
        seed=42,
        seeds=(17, 42, 123),
        exp_id="F02",
        exp_name="F02_pt_single_patient80_20_mesh16384",
        test_fraction=0.20,
        k_folds=5,
        epochs=200,
        batch_size=8,
        lr=3e-4,
        lr_decay_step=100,
        lr_decay_gamma=0.7,
        dropout=0.3,
        grad_clip_norm=1.0,
        flip_prob=0.2,
        eval_batch_size=8,
        retrain_use_cv_epoch=True,
        print_interval=20,
        pt_dims=(128, 256, 512),
        pt_npts=(2048, 512, 128),
        pt_k=16,
        pt_pre_fps_n=4096,
        run_tag="",
    )
    cfg.seeds = tuple(int(s) for s in cfg.seeds)
    if len(cfg.seeds) == 0:
        cfg.seeds = (int(cfg.seed),)
    cfg.seed = int(cfg.seeds[0])
    cfg.print_interval = max(1, int(cfg.print_interval))
    cfg.k_folds = max(2, int(cfg.k_folds))
    cfg.epochs = max(1, int(cfg.epochs))
    cfg.batch_size = max(1, int(cfg.batch_size))
    cfg.eval_batch_size = max(1, int(cfg.eval_batch_size))
    return cfg


def run_seed_ablation(cfg):
    seeds = tuple(int(s) for s in cfg.seeds)
    print("\n" + "#" * 68)
    print(f"  Running seed ablation: {list(seeds)}")
    print("#" * 68)

    for i, seed in enumerate(seeds, start=1):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = int(seed)
        if cfg.run_tag:
            run_cfg.run_tag = f"{cfg.run_tag}_{cfg.exp_id}_seed{seed}"
        else:
            run_cfg.run_tag = f"{cfg.exp_id}_seed{seed}"

        print("\n" + "*" * 68)
        print(f"  Seed run {i}/{len(seeds)}: seed={seed}, run_tag='{run_cfg.run_tag}'")
        print("*" * 68)
        train_kfold(run_cfg)


if __name__ == "__main__":
    run_seed_ablation(get_config())
