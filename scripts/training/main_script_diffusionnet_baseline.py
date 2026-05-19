import os
import sys
import gc
import json
import random
from collections import defaultdict
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DIFFNET_SRC = os.path.join(ROOT_DIR, "diffusion-net-repo", "src")
OP_CACHE_DIR = os.path.join(ROOT_DIR, "data", "diffnet_op_cache")

for path in (ROOT_DIR, UTILS_DIR, MODELS_DIR, BASE_DIR, DIFFNET_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

import diffusion_net
import train_heatmap_joint_flip_v3 as base


NUM_LANDMARKS = 9
OUTPUT_DIM = NUM_LANDMARKS * 3
LANDMARK_NAMES = base.LANDMARK_NAMES

ALARE_R_IDX, ALARE_L_IDX = 5, 6
ZYGION_R_IDX, ZYGION_L_IDX = 7, 8

DIFFNET_C_WIDTH = 128
DIFFNET_N_BLOCK = 4
DIFFNET_K_EIG = 64
DIFFNET_INPUT_FEATURES = "xyz"
DIFFNET_HEAD_DROPOUT = 0.3

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)


def resolve_device(mode="auto"):
    mode = str(mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    for p in patients:
        (test_idx if p in test_patients else train_idx).extend(patient_to_idx[p])

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def patient_kfold(names, indices, k=5, seed=42):
    patient_to_local = defaultdict(list)
    for local_i, global_i in enumerate(indices):
        patient_to_local[patient_id(names[global_i])].append(local_i)

    patients = sorted(patient_to_local)
    for tr_pat_idx, val_pat_idx in KFold(n_splits=k, shuffle=True, random_state=seed).split(patients):
        tr_local, val_local = [], []
        for i in tr_pat_idx:
            tr_local.extend(patient_to_local[patients[i]])
        for i in val_pat_idx:
            val_local.extend(patient_to_local[patients[i]])
        yield np.array(tr_local), np.array(val_local)


def compute_operators(verts_list, faces_list, k_eig=DIFFNET_K_EIG, cache_dir=OP_CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    ops = []

    for i, (verts, faces) in enumerate(zip(verts_list, faces_list)):
        try:
            op = diffusion_net.geometry.get_operators(
                verts,
                faces,
                k_eig=k_eig,
                op_cache_dir=cache_dir,
            )
        except Exception as exc:
            fallback_k = min(int(k_eig), 32)
            print(f"[DiffNet] sample {i}: k_eig={k_eig} failed ({exc}); retrying with {fallback_k}")
            op = diffusion_net.geometry.get_operators(
                verts,
                faces,
                k_eig=fallback_k,
                op_cache_dir=cache_dir,
            )
        ops.append(op)

    return ops


def load_data(k_eig=DIFFNET_K_EIG, input_features=DIFFNET_INPUT_FEATURES):
    _, Y, scales, names, full_clouds, triangles = base.load_data()

    verts_list = [torch.from_numpy(pc).float() for pc in full_clouds]
    faces_list = [
        torch.from_numpy(tri).long() if tri is not None and len(tri) > 0 else torch.zeros((0, 3), dtype=torch.long)
        for tri in triangles
    ]

    diffnet_data = []
    for verts, op in zip(verts_list, compute_operators(verts_list, faces_list, k_eig)):
        frames, mass, L, evals, evecs, gradX, gradY = op
        features = (
            diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            if input_features == "hks"
            else verts
        )
        diffnet_data.append(
            {
                "features": features,
                "mass": mass,
                "L": L,
                "evals": evals,
                "evecs": evecs,
                "gradX": gradX,
                "gradY": gradY,
            }
        )

    return Y, scales, names, full_clouds, triangles, diffnet_data


class DiffusionNetRegressor(nn.Module):
    def __init__(self, c_in=3, output_dim=OUTPUT_DIM, c_width=128, n_block=4, global_dim=256, dropout=0.3):
        super().__init__()
        self.backbone = diffusion_net.layers.DiffusionNet(
            C_in=c_in,
            C_out=global_dim,
            C_width=c_width,
            N_block=n_block,
            last_activation=None,
            outputs_at="global_mean",
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
            diffusion_method="spectral",
        )
        self.head = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, features, mass, L, evals, evecs, gradX, gradY):
        x = self.backbone(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        return self.head(x)


class DiffNetDataset(Dataset):
    def __init__(self, Y, indices, diffnet_data):
        self.Y = Y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.diffnet_data = diffnet_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return self.Y[idx], self.diffnet_data[idx], idx


def collate_fn(batch):
    y, dd, idx = zip(*batch)
    return torch.from_numpy(np.stack(y)).float(), list(dd), torch.tensor(idx, dtype=torch.long)


def make_loader(Y, indices, diffnet_data, batch_size, shuffle):
    return DataLoader(
        DiffNetDataset(Y, indices, diffnet_data),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )


def augment_labels_and_features(labels, dd_list, device, flip_prob=0.2):
    labels = labels.to(device)
    batch_size = labels.size(0)

    theta = torch.rand(batch_size, 1, 1, device=device) * (np.pi / 6) - (np.pi / 12)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    rot = torch.zeros(batch_size, 3, 3, device=device)
    rot[:, 0, 0] = cos_t.flatten()
    rot[:, 0, 1] = -sin_t.flatten()
    rot[:, 1, 0] = sin_t.flatten()
    rot[:, 1, 1] = cos_t.flatten()
    rot[:, 2, 2] = 1.0

    labels = torch.bmm(
        labels.view(batch_size * NUM_LANDMARKS, 1, 3),
        rot.repeat_interleave(NUM_LANDMARKS, dim=0),
    ).view(batch_size, NUM_LANDMARKS, 3)

    scale = torch.rand(batch_size, 1, 1, device=device) * 0.10 + 0.95
    shift = torch.rand(batch_size, 1, 3, device=device) * 0.04 - 0.02

    labels = labels * scale + shift

    flip_idx = (torch.rand(batch_size, device=device) < float(flip_prob)).nonzero(as_tuple=True)[0]
    flip_set = set(int(i) for i in flip_idx.cpu().tolist())

    if len(flip_idx) > 0:
        labels[flip_idx, :, 0] = -labels[flip_idx, :, 0]

        tmp = labels[flip_idx, ALARE_R_IDX].clone()
        labels[flip_idx, ALARE_R_IDX] = labels[flip_idx, ALARE_L_IDX]
        labels[flip_idx, ALARE_L_IDX] = tmp

        tmp = labels[flip_idx, ZYGION_R_IDX].clone()
        labels[flip_idx, ZYGION_R_IDX] = labels[flip_idx, ZYGION_L_IDX]
        labels[flip_idx, ZYGION_L_IDX] = tmp

    augmented = []
    for i, dd in enumerate(dd_list):
        dd_aug = {key: dd[key] for key in ("mass", "L", "evals", "evecs", "gradX", "gradY")}
        features = dd["features"]

        if features.shape[-1] == 3:
            feat = torch.mm(features.to(device), rot[i])
            feat = feat * scale[i, 0, 0] + shift[i, 0, :]
            if i in flip_set:
                feat[:, 0] = -feat[:, 0]
            dd_aug["features"] = feat.cpu()
        else:
            dd_aug["features"] = features

        augmented.append(dd_aug)

    return labels, augmented


def build_model(cfg, device):
    c_in = 3 if cfg.input_features == "xyz" else 16
    return DiffusionNetRegressor(
        c_in=c_in,
        output_dim=OUTPUT_DIM,
        c_width=cfg.diffnet_c_width,
        n_block=cfg.diffnet_n_block,
        global_dim=cfg.diffnet_global_dim,
        dropout=cfg.dropout,
    ).to(device)


def run_batch(model, dd_list, device):
    preds = []

    for dd in dd_list:
        pred = model(
            dd["features"].to(device),
            dd["mass"].to(device),
            dd["L"].to(device),
            dd["evals"].to(device),
            dd["evecs"].to(device),
            dd["gradX"].to(device),
            dd["gradY"].to(device),
        )
        preds.append(pred.unsqueeze(0) if pred.dim() == 1 else pred)

    return torch.cat(preds, dim=0)


def predict(model, Y, indices, diffnet_data, batch_size, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for _, dd_list, _ in make_loader(Y, indices, diffnet_data, batch_size=batch_size, shuffle=False):
            preds.append(run_batch(model, dd_list, device).cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1, NUM_LANDMARKS, 3)


def evaluate_mm(model, Y, scales, indices, diffnet_data, batch_size, device):
    pred = predict(model, Y, indices, diffnet_data, batch_size, device)
    gt = Y[indices]
    return np.linalg.norm(pred - gt, axis=2) * scales[indices, None]


def evaluate_mm_with_snap_mesh(model, Y, scales, indices, diffnet_data, full_clouds, triangles, batch_size, device):
    pred = predict(model, Y, indices, diffnet_data, batch_size, device)
    gt = Y[indices]

    raw = np.linalg.norm(pred - gt, axis=2) * scales[indices, None]

    snapped = np.empty_like(pred, dtype=np.float32)
    for row_i, sample_i in enumerate(indices):
        pc_full = full_clouds[int(sample_i)]
        tri = triangles[int(sample_i)] if triangles is not None else None
        for landmark_i in range(NUM_LANDMARKS):
            snapped[row_i, landmark_i] = base.snap_to_mesh(pred[row_i, landmark_i], pc_full, tri)

    snapped_err = np.linalg.norm(snapped - gt, axis=2) * scales[indices, None]
    return raw, snapped_err


def print_eval(title, errs):
    print(f"\n[{title}] N={errs.shape[0]}")
    for i, name in enumerate(LANDMARK_NAMES):
        e = errs[:, i]
        print(f"  {name:<12} mean={e.mean():.3f}  median={np.median(e):.3f}  p90={np.percentile(e, 90):.3f}")
    flat = errs.ravel()
    print(f"  {'OVERALL':<12} mean={flat.mean():.3f}  median={np.median(flat):.3f}  p90={np.percentile(flat, 90):.3f}")


def train_one_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss, n_batches = 0.0, 0

    for labels, dd_list, _ in loader:
        labels, dd_aug = augment_labels_and_features(labels, dd_list, device, flip_prob=cfg.flip_prob)
        target = labels.reshape(labels.size(0), -1)

        optimizer.zero_grad()
        pred = run_batch(model, dd_aug, device)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_kfold(cfg):
    set_seed(cfg.seed)
    device = resolve_device(cfg.diffnet_device)

    Y, scales, names, full_clouds, triangles, diffnet_data = load_data(cfg.k_eig, cfg.input_features)
    train_idx, test_idx = patient_based_split(names, cfg.test_fraction, cfg.seed)

    train_patients = {patient_id(names[i]) for i in train_idx}
    test_patients = {patient_id(names[i]) for i in test_idx}
    assert train_patients.isdisjoint(test_patients), "DATA LEAKAGE: patient appears in both train and test split."

    n_folds = max(2, min(cfg.k_folds, len(train_patients)))
    criterion = nn.SmoothL1Loss()
    fold_results = []

    print(f"Device: {device}")
    print(f"Samples: total={len(names)}, train={len(train_idx)}, test={len(test_idx)}")
    print(f"Patients: train={len(train_patients)}, test={len(test_patients)}, folds={n_folds}")

    for fold, (tr_local, val_local) in enumerate(patient_kfold(np.array(names), train_idx, n_folds, cfg.seed), start=1):
        tr_abs = train_idx[tr_local]
        val_abs = train_idx[val_local]

        model = build_model(cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.lr_decay_step), gamma=cfg.lr_decay_gamma)
        loader = make_loader(Y, tr_abs, diffnet_data, cfg.batch_size, shuffle=True)

        best_val, best_epoch = float("inf"), 0

        print(f"\nFold {fold}/{n_folds}: train={len(tr_abs)}, val={len(val_abs)}")
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, loader, optimizer, criterion, cfg, device)
            val_mm = float(evaluate_mm(model, Y, scales, val_abs, diffnet_data, cfg.eval_batch_size, device).mean())
            scheduler.step()

            if val_mm < best_val:
                best_val, best_epoch = val_mm, epoch

            if epoch % cfg.print_interval == 0 or epoch == cfg.epochs:
                print(f"  epoch {epoch:>3}/{cfg.epochs}  loss={train_loss:.5f}  val_mm={val_mm:.3f}  best={best_val:.3f}")

        fold_results.append({"fold": fold, "best_epoch": best_epoch, "best_val_mm": best_val})

        del model, optimizer, scheduler, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_epochs = [r["best_epoch"] for r in fold_results if r["best_epoch"] > 0]
    retrain_epochs = (
        int(np.clip(round(float(np.mean(best_epochs))), 1, cfg.epochs))
        if cfg.retrain_use_cv_epoch and best_epochs
        else cfg.epochs
    )

    print(f"\nRetraining on full train split for {retrain_epochs} epochs")

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.lr_decay_step), gamma=cfg.lr_decay_gamma)
    loader = make_loader(Y, train_idx, diffnet_data, cfg.batch_size, shuffle=True)

    for epoch in range(1, retrain_epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, criterion, cfg, device)
        scheduler.step()

        if epoch % cfg.print_interval == 0 or epoch == retrain_epochs:
            print(f"  retrain epoch {epoch:>3}/{retrain_epochs}  loss={train_loss:.5f}")

    raw_errs, snap_errs = evaluate_mm_with_snap_mesh(
        model,
        Y,
        scales,
        test_idx,
        diffnet_data,
        full_clouds,
        triangles,
        cfg.eval_batch_size,
        device,
    )

    print_eval("TEST raw", raw_errs)
    print_eval("TEST snap-mesh", snap_errs)

    run_dir = os.path.join(ROOT_DIR, "results", "paper", cfg.exp_id, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "diffnet_retrain_fulltrain_final.pth")
    summary_path = os.path.join(run_dir, "holdout_summary.json")

    torch.save({"model": model.state_dict(), "config": vars(cfg), "retrain_epochs": retrain_epochs}, model_path)

    summary = {
        "model_path": model_path,
        "fold_results": fold_results,
        "cv_mean_val_mm": float(np.mean([r["best_val_mm"] for r in fold_results])),
        "cv_std_val_mm": float(np.std([r["best_val_mm"] for r in fold_results])),
        "test_raw_mean_mm": float(raw_errs.mean()),
        "test_snap_mesh_mean_mm": float(snap_errs.mean()),
        "test_snap_mesh_median_mm": float(np.median(snap_errs)),
        "test_snap_mesh_p90_mm": float(np.percentile(snap_errs, 90)),
        "test_snap_mesh_per_landmark_mm": {
            name: float(snap_errs[:, i].mean()) for i, name in enumerate(LANDMARK_NAMES)
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved model: {model_path}")
    print(f"Saved summary: {summary_path}")


def get_config():
    return SimpleNamespace(
        seed=42,
        exp_id="D01",
        exp_name="D01_diffnet_single_patient80_20_mesh16384",
        test_fraction=0.20,
        k_folds=5,
        epochs=200,
        batch_size=2,
        lr=2e-4,
        lr_decay_step=100,
        lr_decay_gamma=0.7,
        dropout=DIFFNET_HEAD_DROPOUT,
        grad_clip_norm=1.0,
        flip_prob=0.2,
        eval_batch_size=4,
        retrain_use_cv_epoch=True,
        print_interval=20,
        diffnet_c_width=DIFFNET_C_WIDTH,
        diffnet_n_block=DIFFNET_N_BLOCK,
        diffnet_global_dim=256,
        k_eig=DIFFNET_K_EIG,
        input_features=DIFFNET_INPUT_FEATURES,
        diffnet_device="auto",
    )


if __name__ == "__main__":
    train_kfold(get_config())