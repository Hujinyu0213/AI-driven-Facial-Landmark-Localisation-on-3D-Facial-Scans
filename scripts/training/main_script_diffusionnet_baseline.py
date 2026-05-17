# pyright: reportMissingImports=false

"""
DiffusionNet Regression Baseline — K-Fold + Patient-Based Split
================================================================
Single-stage DiffusionNet baseline for 9-landmark regression.

Policy source:
- mirrors scripts/training/main_script_pointtransformer_kfold.py
  - patient-based 80/20 holdout
  - patient-level inner K-fold on the train split
  - full-epoch training (no early stopping)
  - live JSON progress snapshots
  - post-CV retrain on the full train split

DiffusionNet safety handling source:
- refers to scripts/training/train_heatmap_joint_hybrid_diffnet_floor_v1_kfold.py
- refers to scripts/training/train_heatmap_joint_hybrid_diffnet_floor_v1_phase1_search.py

Safety design goals:
- Prefer CUDA DiffusionNet runtime when available (`diffnet_device="auto"`)
- Keep `k_eig=64` by default to avoid heavy eigendecomposition stalls
- Use cached operators in data/diffnet_op_cache
- Never add fake batch dimensions to sparse operators
- Use `num_workers=0` and modest batch sizes
"""

import os
import sys
import gc
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

for p in (ROOT_DIR, UTILS_DIR, MODELS_DIR, BASE_DIR, DIFFNET_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

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


def resolve_diffnet_device(mode: str) -> torch.device:
    mode = str(mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--diffnet-device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def patient_kfold(names, indices_subset, k=5, seed=42):
    patient_to_local = defaultdict(list)
    for local_i, global_i in enumerate(indices_subset):
        pid = patient_id_from_name(names[global_i])
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


def precompute_diffnet_operators(verts_list, faces_list, k_eig=DIFFNET_K_EIG, op_cache_dir=None):
    if op_cache_dir is not None:
        os.makedirs(op_cache_dir, exist_ok=True)

    print(f"[DiffNet] Precomputing operators for {len(verts_list)} samples (k_eig={k_eig}) ...")
    ops_list = []
    from tqdm import tqdm

    for i in tqdm(range(len(verts_list)), desc="DiffNet operators", ascii=True, file=sys.stdout):
        verts = verts_list[i]
        faces = faces_list[i]
        try:
            frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
                verts,
                faces,
                k_eig=k_eig,
                op_cache_dir=op_cache_dir,
            )
        except Exception as e:
            fallback_k = min(int(k_eig), 32)
            print(
                f"\n[DiffNet] WARNING: operator computation failed for sample {i} "
                f"(V={verts.shape[0]}): {e}"
            )
            print(f"[DiffNet] Retrying with k_eig={fallback_k} ...")
            frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
                verts,
                faces,
                k_eig=fallback_k,
                op_cache_dir=op_cache_dir,
            )
        ops_list.append((frames, mass, L, evals, evecs, gradX, gradY))

    print(f"[DiffNet] Operators ready for {len(ops_list)} samples.")
    return ops_list


def compute_hks_features(evals, evecs, num_features=16):
    return diffusion_net.geometry.compute_hks_autoscale(evals, evecs, num_features)


def load_data_with_diffnet_ops(k_eig=DIFFNET_K_EIG, input_features=DIFFNET_INPUT_FEATURES, op_cache_dir=None):
    X, Y, scales, names, full_clouds, triangles = base.load_data()

    verts_list = []
    faces_list = []
    for i in range(len(names)):
        verts_t = torch.from_numpy(full_clouds[i]).float()
        tri = triangles[i]
        if tri is not None and len(tri) > 0:
            faces_t = torch.from_numpy(tri).long()
        else:
            faces_t = torch.zeros((0, 3), dtype=torch.long)
        verts_list.append(verts_t)
        faces_list.append(faces_t)

    ops_list = precompute_diffnet_operators(
        verts_list,
        faces_list,
        k_eig=k_eig,
        op_cache_dir=op_cache_dir,
    )

    diffnet_data = []
    for i in range(len(names)):
        frames, mass, L, evals, evecs, gradX, gradY = ops_list[i]
        verts = verts_list[i]
        if input_features == "hks":
            features = compute_hks_features(evals, evecs, num_features=16)
        else:
            features = verts

        diffnet_data.append(
            {
                "verts": verts,
                "mass": mass,
                "L": L,
                "evals": evals,
                "evecs": evecs,
                "gradX": gradX,
                "gradY": gradY,
                "features": features,
            }
        )

    return X, Y, scales, names, full_clouds, triangles, diffnet_data


class DiffusionNetRegressor(nn.Module):
    def __init__(
        self,
        output_dim=OUTPUT_DIM,
        c_in=3,
        c_width=DIFFNET_C_WIDTH,
        n_block=DIFFNET_N_BLOCK,
        global_dim=256,
        head_dropout=DIFFNET_HEAD_DROPOUT,
        diffnet_dropout=True,
    ):
        super().__init__()
        self.global_dim = int(global_dim)
        self.backbone = diffusion_net.layers.DiffusionNet(
            C_in=c_in,
            C_out=self.global_dim,
            C_width=c_width,
            N_block=n_block,
            last_activation=None,
            outputs_at="global_mean",
            dropout=diffnet_dropout,
            with_gradient_features=True,
            with_gradient_rotations=True,
            diffusion_method="spectral",
        )
        self.head = nn.Sequential(
            nn.Linear(self.global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, features, mass, L, evals, evecs, gradX, gradY):
        global_feat = self.backbone(
            features,
            mass,
            L=L,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
        )
        pred = self.head(global_feat)
        return pred


class DiffNetRegressionDataset(Dataset):
    def __init__(self, X, Y, indices, diffnet_data):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.diffnet_data = diffnet_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return self.X[idx], self.Y[idx], self.diffnet_data[idx], idx


def diffnet_collate_fn(batch):
    pcs, gts, dd_list, idxs = zip(*batch)
    pc_tensor = torch.from_numpy(np.stack(pcs)).float()
    gt_tensor = torch.from_numpy(np.stack(gts)).float()
    idx_tensor = torch.tensor(idxs, dtype=torch.long)
    return pc_tensor, gt_tensor, list(dd_list), idx_tensor


def make_diffnet_loader(X, Y, indices, diffnet_data, batch_size, shuffle):
    ds = DiffNetRegressionDataset(X, Y, indices, diffnet_data)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=diffnet_collate_fn,
        num_workers=0,
        drop_last=False,
    )


def augment_diffnet_batch(pc, lbl, dd_list, dev, flip_prob=0.2):
    pc = pc.to(dev)
    lbl = lbl.to(dev)
    B, _, N = pc.shape

    th = torch.rand(B, 1, 1, device=dev) * (2 * np.pi / 12) - np.pi / 12
    c, s = torch.cos(th), torch.sin(th)
    rot = torch.zeros(B, 3, 3, device=dev)
    rot[:, 0, 0] = c.flatten()
    rot[:, 0, 1] = -s.flatten()
    rot[:, 1, 0] = s.flatten()
    rot[:, 1, 1] = c.flatten()
    rot[:, 2, 2] = 1.0

    pc_r = torch.bmm(pc.transpose(1, 2), rot)
    lbl_r = torch.bmm(
        lbl.view(B * NUM_LANDMARKS, 1, 3),
        rot.repeat_interleave(NUM_LANDMARKS, dim=0),
    ).view(B, NUM_LANDMARKS, 3)

    sc = torch.rand(B, 1, 1, device=dev) * 0.10 + 0.95
    pc_r = pc_r * sc
    lbl_r = lbl_r * sc

    sh = torch.rand(B, 1, 3, device=dev) * 0.04 - 0.02
    pc_r = pc_r + sh
    lbl_r = lbl_r + sh

    pc_r = pc_r + torch.randn(B, N, 3, device=dev) * 0.005

    flip_idx = (torch.rand(B, device=dev) < float(flip_prob)).nonzero(as_tuple=True)[0]
    flip_set = set(int(x) for x in flip_idx.cpu().tolist())
    if len(flip_set) > 0:
        pc_r[flip_idx, :, 0] = -pc_r[flip_idx, :, 0]
        lbl_r[flip_idx, :, 0] = -lbl_r[flip_idx, :, 0]

        saved_alare_r = lbl_r[flip_idx, ALARE_R_IDX].clone()
        lbl_r[flip_idx, ALARE_R_IDX] = lbl_r[flip_idx, ALARE_L_IDX]
        lbl_r[flip_idx, ALARE_L_IDX] = saved_alare_r

        saved_zygion_r = lbl_r[flip_idx, ZYGION_R_IDX].clone()
        lbl_r[flip_idx, ZYGION_R_IDX] = lbl_r[flip_idx, ZYGION_L_IDX]
        lbl_r[flip_idx, ZYGION_L_IDX] = saved_zygion_r

    augmented_dd_list = []
    for b_idx in range(B):
        dd = dd_list[b_idx]
        dd_aug = {}
        for key in ["mass", "L", "evals", "evecs", "gradX", "gradY"]:
            dd_aug[key] = dd[key]

        features = dd["features"]
        if features.shape[-1] == 3:
            feat_dev = features.to(dev)
            feat_r = torch.mm(feat_dev, rot[b_idx])
            feat_r = feat_r * sc[b_idx, 0, 0]
            feat_r = feat_r + sh[b_idx, 0, :]
            if b_idx in flip_set:
                feat_r[:, 0] = -feat_r[:, 0]
            dd_aug["features"] = feat_r.cpu()
        else:
            dd_aug["features"] = features
        augmented_dd_list.append(dd_aug)

    return pc_r.transpose(1, 2), lbl_r, augmented_dd_list


def build_model(args, runtime_device):
    c_in = 3 if args.input_features == "xyz" else 16
    model = DiffusionNetRegressor(
        output_dim=OUTPUT_DIM,
        c_in=c_in,
        c_width=args.diffnet_c_width,
        n_block=args.diffnet_n_block,
        global_dim=args.diffnet_global_dim,
        head_dropout=args.dropout,
        diffnet_dropout=True,
    ).to(runtime_device)
    return model


def run_diffusion_batch(model, dd_list, runtime_device):
    preds = []
    for dd in dd_list:
        features = dd["features"].to(runtime_device)
        mass = dd["mass"].to(runtime_device)
        L = dd["L"].to(runtime_device)
        evals = dd["evals"].to(runtime_device)
        evecs = dd["evecs"].to(runtime_device)
        gradX = dd["gradX"].to(runtime_device)
        gradY = dd["gradY"].to(runtime_device)

        pred = model(features, mass, L, evals, evecs, gradX, gradY)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        preds.append(pred)

        del features, mass, L, evals, evecs, gradX, gradY

    out = torch.cat(preds, dim=0)
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    return out


def predict_in_batches(model, X_np, indices, diffnet_data, batch_size=4, runtime_device=torch.device("cpu")):
    model.eval()
    loader = make_diffnet_loader(
        X_np,
        np.zeros((len(X_np), NUM_LANDMARKS, 3), dtype=np.float32),
        indices,
        diffnet_data,
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for _, _, dd_list, _ in loader:
            pred = run_diffusion_batch(model, dd_list, runtime_device)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_mm(model, X_np, Y_np, scales_np, indices, diffnet_data, batch_size=4, runtime_device=torch.device("cpu")):
    pred = predict_in_batches(model, X_np, indices, diffnet_data, batch_size=batch_size, runtime_device=runtime_device)
    pred_lm = pred.reshape(-1, NUM_LANDMARKS, 3)
    gt = Y_np[indices]
    sc = scales_np[indices]
    return np.linalg.norm(pred_lm - gt, axis=2) * sc[:, None]


def evaluate_mm_with_snap_mesh(
    model,
    X_np,
    Y_np,
    scales_np,
    indices,
    diffnet_data,
    full_clouds,
    triangles,
    batch_size=4,
    runtime_device=torch.device("cpu"),
):
    pred = predict_in_batches(model, X_np, indices, diffnet_data, batch_size=batch_size, runtime_device=runtime_device)
    pred = pred.reshape(-1, NUM_LANDMARKS, 3)
    gt = Y_np[indices]
    sc = scales_np[indices]
    errs_raw = np.linalg.norm(pred - gt, axis=2) * sc[:, None]

    snapped = np.empty_like(pred, dtype=np.float32)
    for i, abs_idx in enumerate(indices):
        pc_full = full_clouds[int(abs_idx)]
        tri = triangles[int(abs_idx)] if triangles is not None else None
        for k in range(NUM_LANDMARKS):
            snapped[i, k] = base.snap_to_mesh(pred[i, k], pc_full, tri)

    errs_snap = np.linalg.norm(snapped - gt, axis=2) * sc[:, None]
    return errs_raw, errs_snap


def print_eval(tag, errs):
    print(f"\n[{tag}]  N={errs.shape[0]}")
    print(f"  {'landmark':<12} {'mean':>7} {'median':>7} {'P90':>7}")
    print("  " + "-" * 38)
    for k, lm in enumerate(LANDMARK_NAMES):
        e = errs[:, k]
        print(f"  {lm:<12} {e.mean():>7.3f} {np.median(e):>7.3f} {np.percentile(e,90):>7.3f}")
    flat = errs.flatten()
    print(f"  {'OVERALL':<12} {flat.mean():>7.3f} {np.median(flat):>7.3f} {np.percentile(flat,90):>7.3f}  (mm)")


def _atomic_json_dump(payload, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def train_kfold(args):
    set_global_seed(int(args.seed))
    runtime_device = resolve_diffnet_device(args.diffnet_device)

    print("\n" + "=" * 68)
    print("  DiffusionNet Baseline — Configuration")
    print("=" * 68)
    print(f"  {'Model':<26}: DiffusionNetRegressor output_dim={OUTPUT_DIM}")
    print(f"  {'DiffNet c_width':<26}: {args.diffnet_c_width}")
    print(f"  {'DiffNet n_block':<26}: {args.diffnet_n_block}")
    print(f"  {'DiffNet global_dim':<26}: {args.diffnet_global_dim}")
    print(f"  {'DiffNet k_eig':<26}: {args.k_eig}")
    print(f"  {'Input features':<26}: {args.input_features}")
    print(f"  {'Runtime device':<26}: {runtime_device}")
    print(f"  {'K-Folds':<26}: {args.k_folds}")
    print(f"  {'Seed':<26}: {args.seed}")
    print(f"  {'Split':<26}: patient-based {int((1-args.test_fraction)*100)}/{int(args.test_fraction*100)}")
    print(f"  {'Batch / Epochs':<26}: {args.batch_size} / {args.epochs}")
    print(f"  {'LR / step / gamma':<26}: {args.lr} / {args.lr_decay_step} / {args.lr_decay_gamma}")
    print(f"  {'Grad clip norm':<26}: {args.grad_clip_norm}")
    print(f"  {'Early stop':<26}: disabled (full epochs)")
    print(f"  {'Loss':<26}: SmoothL1")
    print(f"  {'Op cache':<26}: {OP_CACHE_DIR}")
    print(f"  {'Run tag':<26}: {args.run_tag if args.run_tag else '(none)'}")
    print("=" * 68 + "\n")

    X, Y, scales, names, full_clouds, triangles, diffnet_data = load_data_with_diffnet_ops(
        k_eig=args.k_eig,
        input_features=args.input_features,
        op_cache_dir=OP_CACHE_DIR,
    )
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

    criterion = nn.SmoothL1Loss()
    fold_results = []
    all_histories = []
    best_overall_mm = float("inf")
    best_fold_idx = -1

    split_tag = f"patient{int((1-args.test_fraction)*100)}_{int(args.test_fraction*100)}"
    exp_name = args.exp_name or f"{args.exp_id}_diffnet_single_{split_tag}_mesh16384"
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
            "model": "DiffusionNetRegressor (baseline)",
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
                "diffnet_c_width": args.diffnet_c_width,
                "diffnet_n_block": args.diffnet_n_block,
                "diffnet_global_dim": args.diffnet_global_dim,
                "k_eig": args.k_eig,
                "input_features": args.input_features,
                "diffnet_device": str(runtime_device),
                "eval_batch_size": args.eval_batch_size,
                "retrain_use_cv_epoch": bool(args.retrain_use_cv_epoch),
                "flip_prob": args.flip_prob,
                "operator_cache": OP_CACHE_DIR,
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
        _atomic_json_dump(payload, history_path)

    save_progress(status="running", current_fold=0, current_epoch=0)

    names_arr = np.array(names)
    for fold_idx, (tr_loc, val_loc) in enumerate(
        patient_kfold(names_arr, train_idx, k=n_inner, seed=args.seed), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{n_inner}  |  train={len(tr_loc)}  val={len(val_loc)}")
        print(f"{'='*60}")

        tr_abs = train_idx[tr_loc]
        val_abs = train_idx[val_loc]

        model = build_model(args, runtime_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.lr_decay_step),
            gamma=args.lr_decay_gamma,
        )
        train_dl = make_diffnet_loader(X, Y, tr_abs, diffnet_data, batch_size=args.batch_size, shuffle=True)

        best_val_mm = float("inf")
        best_state = None
        best_epoch = -1
        history = {"epoch": [], "train_loss": [], "val_mm": [], "lr": []}

        for epoch in range(1, args.epochs + 1):
            model.train()
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss_acc, n_batches = 0.0, 0

            for batch_pc, batch_lbl, dd_list, _ in train_dl:
                batch_pc, batch_lbl, dd_list_aug = augment_diffnet_batch(
                    batch_pc,
                    batch_lbl,
                    dd_list,
                    runtime_device,
                    flip_prob=args.flip_prob,
                )
                batch_lbl_flat = batch_lbl.reshape(batch_lbl.size(0), -1)

                optimizer.zero_grad()
                pred = run_diffusion_batch(model, dd_list_aug, runtime_device)
                loss = criterion(pred, batch_lbl_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()

                train_loss_acc += loss.item()
                n_batches += 1

            avg_train = train_loss_acc / max(n_batches, 1)
            val_errs = evaluate_mm(
                model,
                X,
                Y,
                scales,
                val_abs,
                diffnet_data,
                batch_size=args.eval_batch_size,
                runtime_device=runtime_device,
            )
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
        torch.save(
            {
                "model": best_state,
                "config": vars(args),
                "runtime_device": str(runtime_device),
                "best_epoch": int(best_epoch),
                "best_val_mm": float(best_val_mm),
            },
            fold_path,
        )
        print(f"  -> Fold {fold_idx} best val L2 = {best_val_mm:.3f} mm  saved: {fold_path}")

        model.load_state_dict(best_state)
        val_errs_lm = evaluate_mm(
            model,
            X,
            Y,
            scales,
            val_abs,
            diffnet_data,
            batch_size=args.eval_batch_size,
            runtime_device=runtime_device,
        )
        print_eval(f"Fold {fold_idx} val", val_errs_lm)

        fold_results.append(
            {
                "fold": fold_idx,
                "best_val_mm": float(best_val_mm),
                "best_epoch": int(best_epoch),
                "train_scans": int(len(tr_abs)),
                "val_scans": int(len(val_abs)),
                "per_landmark_val_mm": val_errs_lm.mean(axis=0).tolist(),
            }
        )
        all_histories.append(history)

        if best_val_mm < best_overall_mm:
            best_overall_mm = best_val_mm
            best_fold_idx = fold_idx

        save_progress(status="running", current_fold=fold_idx, current_epoch=args.epochs)

        del model, optimizer, scheduler, train_dl, best_state
        gc.collect()
        if runtime_device.type == "cuda":
            torch.cuda.empty_cache()

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

    full_model = build_model(args, runtime_device)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, args.lr_decay_step),
        gamma=args.lr_decay_gamma,
    )
    train_full_dl = make_diffnet_loader(X, Y, train_idx, diffnet_data, batch_size=args.batch_size, shuffle=True)
    retrain_history = {"phase": "retrain_full_train", "epoch": [], "train_loss": [], "lr": []}

    for epoch in range(1, retrain_epochs + 1):
        full_model.train()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss_acc, n_batches = 0.0, 0

        for batch_pc, batch_lbl, dd_list, _ in train_full_dl:
            batch_pc, batch_lbl, dd_list_aug = augment_diffnet_batch(
                batch_pc,
                batch_lbl,
                dd_list,
                runtime_device,
                flip_prob=args.flip_prob,
            )
            batch_lbl_flat = batch_lbl.reshape(batch_lbl.size(0), -1)

            optimizer.zero_grad()
            pred = run_diffusion_batch(full_model, dd_list_aug, runtime_device)
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
    torch.save(
        {
            "model": full_model.state_dict(),
            "config": vars(args),
            "runtime_device": str(runtime_device),
            "retrain_epochs": int(retrain_epochs),
        },
        retrain_final_path,
    )
    print(f"Retrained full-train model saved: {retrain_final_path}")

    best_model = build_model(args, runtime_device)
    ckpt = torch.load(retrain_final_path, map_location=runtime_device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    best_model.load_state_dict(state_dict)

    test_errs_raw, test_errs_snap = evaluate_mm_with_snap_mesh(
        best_model,
        X,
        Y,
        scales,
        test_idx,
        diffnet_data,
        full_clouds,
        triangles,
        batch_size=args.eval_batch_size,
        runtime_device=runtime_device,
    )

    print_eval("TEST SET (held-out, raw)", test_errs_raw)
    print_eval("TEST SET (held-out, snap-mesh)", test_errs_snap)

    final_test_results = {
        "primary_metric": "snap_mesh",
        "n_scans": int(len(test_idx)),
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
            writer.writerow([int(test_idx[i]), float(test_errs_raw[i].mean()), float(test_errs_snap[i].mean())])

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
        "runtime_device": str(runtime_device),
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
    print("  TRAINING COMPLETE — DiffusionNet Baseline")
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
        exp_id="D01",
        exp_name="D01_diffnet_single_patient80_20_mesh16384",
        test_fraction=0.20,
        k_folds=5,
        epochs=200,
        batch_size=2,
        lr=2e-4,
        lr_decay_step=100,
        lr_decay_gamma=0.7,
        dropout=0.3,
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
        diffnet_device="cuda",
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