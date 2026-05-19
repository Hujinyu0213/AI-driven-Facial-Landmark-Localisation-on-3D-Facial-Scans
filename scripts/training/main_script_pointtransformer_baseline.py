import os
import sys
import gc
import json
import copy
import random
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

for path in (ROOT_DIR, UTILS_DIR, MODELS_DIR, BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import scripts.training.train_script_pointnet2_c2f as base
from models.point_transformer_reg import PointTransformerReg


NUM_LANDMARKS = 9
OUTPUT_DIM = NUM_LANDMARKS * 3
LANDMARK_NAMES = base.LANDMARK_NAMES

ALARE_R_IDX, ALARE_L_IDX = 5, 6
ZYGION_R_IDX, ZYGION_L_IDX = 7, 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_id(name):
    return name.split("_")[0]


def patient_based_split(names, test_fraction=0.20, seed=42):
    patient_to_idx = defaultdict(list)
    for i, name in enumerate(names):
        patient_to_idx[patient_id(name)].append(i)

    patients = sorted(patient_to_idx)
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n_test = max(1, int(len(patients) * test_fraction))
    test_patients = set(patients[:n_test])

    train_idx, test_idx = [], []
    for pid in patients:
        if pid in test_patients:
            test_idx.extend(patient_to_idx[pid])
        else:
            train_idx.extend(patient_to_idx[pid])

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def patient_kfold(names, indices, k=5, seed=42):
    patient_to_local = defaultdict(list)
    for local_i, global_i in enumerate(indices):
        patient_to_local[patient_id(names[global_i])].append(local_i)

    patients = sorted(patient_to_local)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for train_patient_idx, val_patient_idx in kf.split(patients):
        train_local, val_local = [], []

        for i in train_patient_idx:
            train_local.extend(patient_to_local[patients[i]])
        for i in val_patient_idx:
            val_local.extend(patient_to_local[patients[i]])

        yield np.array(train_local), np.array(val_local)


def make_model(cfg, device):
    return PointTransformerReg(
        output_dim=OUTPUT_DIM,
        dropout=cfg.dropout,
        dims=cfg.pt_dims,
        n_pts=cfg.pt_npts,
        k=cfg.pt_k,
        pre_fps_n=cfg.pt_pre_fps_n,
        normal_channel=False,
    ).to(device)


def make_loader(X, Y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def augment_batch(points, labels, flip_prob=0.2):
    points = points.float()
    labels = labels.float()

    batch_size, _, n_points = points.shape
    device = points.device

    theta = torch.rand(batch_size, device=device) * (np.pi / 6) - (np.pi / 12)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rot = torch.zeros(batch_size, 3, 3, device=device)
    rot[:, 0, 0] = cos_t
    rot[:, 0, 1] = -sin_t
    rot[:, 1, 0] = sin_t
    rot[:, 1, 1] = cos_t
    rot[:, 2, 2] = 1.0

    points = torch.bmm(points.transpose(1, 2), rot)
    labels = torch.bmm(labels, rot)

    scale = torch.rand(batch_size, 1, 1, device=device) * 0.10 + 0.95
    shift = torch.rand(batch_size, 1, 3, device=device) * 0.04 - 0.02

    points = points * scale + shift
    labels = labels * scale + shift
    points = points + torch.randn(batch_size, n_points, 3, device=device) * 0.005

    flip_idx = (torch.rand(batch_size, device=device) < float(flip_prob)).nonzero(as_tuple=True)[0]
    if len(flip_idx) > 0:
        points[flip_idx, :, 0] = -points[flip_idx, :, 0]
        labels[flip_idx, :, 0] = -labels[flip_idx, :, 0]

        tmp = labels[flip_idx, ALARE_R_IDX].clone()
        labels[flip_idx, ALARE_R_IDX] = labels[flip_idx, ALARE_L_IDX]
        labels[flip_idx, ALARE_L_IDX] = tmp

        tmp = labels[flip_idx, ZYGION_R_IDX].clone()
        labels[flip_idx, ZYGION_R_IDX] = labels[flip_idx, ZYGION_L_IDX]
        labels[flip_idx, ZYGION_L_IDX] = tmp

    return points.transpose(1, 2), labels


def predict(model, X, batch_size, device):
    model.eval()
    preds = []

    loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for points in loader:
            preds.append(model(points.to(device)).cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1, NUM_LANDMARKS, 3)


def evaluate_mm(model, X, Y, scales, batch_size, device):
    pred = predict(model, X, batch_size, device)
    return np.linalg.norm(pred - Y, axis=2) * scales[:, None]


def evaluate_mm_with_snap_mesh(model, X, Y, scales, full_clouds, triangles, batch_size, device):
    pred = predict(model, X, batch_size, device)

    raw_errs = np.linalg.norm(pred - Y, axis=2) * scales[:, None]

    snapped = np.empty_like(pred, dtype=np.float32)
    for i in range(pred.shape[0]):
        tri = triangles[i] if triangles is not None else None
        for k in range(NUM_LANDMARKS):
            snapped[i, k] = base.snap_to_mesh(pred[i, k], full_clouds[i], tri)

    snap_errs = np.linalg.norm(snapped - Y, axis=2) * scales[:, None]
    return raw_errs, snap_errs


def print_eval(title, errs):
    print(f"\n[{title}] N={errs.shape[0]}")

    for i, name in enumerate(LANDMARK_NAMES):
        e = errs[:, i]
        print(
            f"  {name:<12} "
            f"mean={e.mean():.3f}  "
            f"median={np.median(e):.3f}  "
            f"p90={np.percentile(e, 90):.3f}"
        )

    flat = errs.ravel()
    print(
        f"  {'OVERALL':<12} "
        f"mean={flat.mean():.3f}  "
        f"median={np.median(flat):.3f}  "
        f"p90={np.percentile(flat, 90):.3f}"
    )


def train_one_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss, n_batches = 0.0, 0

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)

        points, labels = augment_batch(points, labels, flip_prob=cfg.flip_prob)
        target = labels.reshape(labels.size(0), -1)

        optimizer.zero_grad()
        pred = model(points)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_kfold(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, scales, names, full_clouds, triangles = base.load_data()
    train_idx, test_idx = patient_based_split(names, cfg.test_fraction, cfg.seed)

    train_patients = {patient_id(names[i]) for i in train_idx}
    test_patients = {patient_id(names[i]) for i in test_idx}
    assert train_patients.isdisjoint(test_patients), "DATA LEAKAGE: patient appears in both train and test split."

    n_folds = max(2, min(cfg.k_folds, len(train_patients)))
    criterion = nn.SmoothL1Loss()
    fold_results = []

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    scales_train = scales[train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    scales_test = scales[test_idx]
    full_clouds_test = [full_clouds[i] for i in test_idx]
    triangles_test = [triangles[i] for i in test_idx] if triangles is not None else None

    print(f"Device: {device}")
    print(f"Samples: total={len(X)}, train={len(train_idx)}, test={len(test_idx)}")
    print(f"Patients: train={len(train_patients)}, test={len(test_patients)}, folds={n_folds}")

    for fold, (tr_local, val_local) in enumerate(patient_kfold(np.array(names), train_idx, n_folds, cfg.seed), start=1):
        X_tr = X_train[tr_local]
        Y_tr = Y_train[tr_local]
        X_val = X_train[val_local]
        Y_val = Y_train[val_local]
        scales_val = scales_train[val_local]

        model = make_model(cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.lr_decay_step),
            gamma=cfg.lr_decay_gamma,
        )
        loader = make_loader(X_tr, Y_tr, cfg.batch_size, shuffle=True)

        best_state = None
        best_epoch = 0
        best_val_mm = float("inf")

        print(f"\nFold {fold}/{n_folds}: train={len(X_tr)}, val={len(X_val)}")

        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, loader, optimizer, criterion, cfg, device)
            val_errs = evaluate_mm(model, X_val, Y_val, scales_val, cfg.eval_batch_size, device)
            val_mm = float(val_errs.mean())

            scheduler.step()

            if val_mm < best_val_mm:
                best_val_mm = val_mm
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            if epoch % cfg.print_interval == 0 or epoch == cfg.epochs:
                print(
                    f"  epoch {epoch:>3}/{cfg.epochs}  "
                    f"loss={train_loss:.5f}  "
                    f"val_mm={val_mm:.3f}  "
                    f"best={best_val_mm:.3f}"
                )

        model.load_state_dict(best_state)
        val_errs = evaluate_mm(model, X_val, Y_val, scales_val, cfg.eval_batch_size, device)
        print_eval(f"Fold {fold} val", val_errs)

        fold_results.append(
            {
                "fold": int(fold),
                "best_epoch": int(best_epoch),
                "best_val_mm": float(best_val_mm),
                "train_scans": int(len(X_tr)),
                "val_scans": int(len(X_val)),
                "per_landmark_val_mm": val_errs.mean(axis=0).tolist(),
            }
        )

        del model, optimizer, scheduler, loader, best_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_epochs = [r["best_epoch"] for r in fold_results if r["best_epoch"] > 0]
    retrain_epochs = (
        int(np.clip(round(float(np.mean(best_epochs))), 1, cfg.epochs))
        if cfg.retrain_use_cv_epoch and best_epochs
        else int(cfg.epochs)
    )

    print(f"\nRetraining on full train split for {retrain_epochs} epochs")

    model = make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, cfg.lr_decay_step),
        gamma=cfg.lr_decay_gamma,
    )
    loader = make_loader(X_train, Y_train, cfg.batch_size, shuffle=True)

    for epoch in range(1, retrain_epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, criterion, cfg, device)
        scheduler.step()

        if epoch % cfg.print_interval == 0 or epoch == retrain_epochs:
            print(f"  retrain epoch {epoch:>3}/{retrain_epochs}  loss={train_loss:.5f}")

    test_raw, test_snap = evaluate_mm_with_snap_mesh(
        model,
        X_test,
        Y_test,
        scales_test,
        full_clouds_test,
        triangles_test,
        cfg.eval_batch_size,
        device,
    )

    print_eval("TEST raw", test_raw)
    print_eval("TEST snap-mesh", test_snap)

    run_dir = os.path.join(ROOT_DIR, "results", "paper", cfg.exp_id, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "point_transformer_retrain_fulltrain_final.pth")
    summary_path = os.path.join(run_dir, "holdout_summary.json")

    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(cfg),
            "retrain_epochs": int(retrain_epochs),
        },
        model_path,
    )

    val_mms = [r["best_val_mm"] for r in fold_results]
    summary = {
        "model_path": model_path,
        "fold_results": fold_results,
        "cv_mean_val_mm": float(np.mean(val_mms)),
        "cv_std_val_mm": float(np.std(val_mms)),
        "retrain_epochs": int(retrain_epochs),
        "test_raw_mean_mm": float(test_raw.mean()),
        "test_raw_median_mm": float(np.median(test_raw)),
        "test_raw_p90_mm": float(np.percentile(test_raw, 90)),
        "test_snap_mesh_mean_mm": float(test_snap.mean()),
        "test_snap_mesh_median_mm": float(np.median(test_snap)),
        "test_snap_mesh_p90_mm": float(np.percentile(test_snap, 90)),
        "test_snap_mesh_per_landmark_mm": {
            name: float(test_snap[:, i].mean()) for i, name in enumerate(LANDMARK_NAMES)
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining complete")
    print(f"CV mean val: {summary['cv_mean_val_mm']:.3f} ± {summary['cv_std_val_mm']:.3f} mm")
    print(f"Test raw: {summary['test_raw_mean_mm']:.3f} mm")
    print(f"Test snap-mesh: {summary['test_snap_mesh_mean_mm']:.3f} mm")
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


def get_config():
    cfg = SimpleNamespace(
        seed=42,
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
    )

    cfg.k_folds = max(2, int(cfg.k_folds))
    cfg.epochs = max(1, int(cfg.epochs))
    cfg.batch_size = max(1, int(cfg.batch_size))
    cfg.eval_batch_size = max(1, int(cfg.eval_batch_size))
    cfg.print_interval = max(1, int(cfg.print_interval))
    return cfg


if __name__ == "__main__":
    train_kfold(get_config())