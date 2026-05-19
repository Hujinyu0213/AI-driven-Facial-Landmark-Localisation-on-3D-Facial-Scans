import copy
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
EXPORT_ROOT = os.path.join(ROOT_DIR, "data", "pointcloud")
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")

for path in (ROOT_DIR, MODELS_DIR, UTILS_DIR, BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from pointnet2_reg import PointNet2RegMSG
from point_transformer_reg import PointTransformerReg

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as _fps_cuda

    def furthest_point_sample(xyz_tensor, npoint):
        return _fps_cuda(xyz_tensor, npoint)

except ImportError:

    def furthest_point_sample(xyz_tensor, npoint):
        batch_size, n_points, _ = xyz_tensor.shape
        device = xyz_tensor.device

        idx = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
        dist = torch.full((batch_size, n_points), 1e10, device=device)
        farthest = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(npoint):
            idx[:, i] = farthest
            selected = xyz_tensor[torch.arange(batch_size), farthest].unsqueeze(1)
            d = ((xyz_tensor - selected) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            farthest = dist.argmax(-1)

        return idx


try:
    from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG as PointNetSetAbstractionMsg
except ImportError:
    from pointnet2_utils import PointNetSetAbstractionMsg as _LegacySAMsg

    class PointNetSetAbstractionMsg(nn.Module):
        def __init__(self, npoint, radii, nsamples, mlps, use_xyz=True):
            super().__init__()
            in_channel = mlps[0][0]
            mlp_list = [m[1:] for m in mlps]
            self.impl = _LegacySAMsg(npoint, radii, nsamples, in_channel, mlp_list)

        def forward(self, xyz, features):
            xyz_t = xyz.transpose(1, 2).contiguous()
            new_xyz, new_points = self.impl(xyz_t, features)
            return new_xyz.transpose(1, 2).contiguous(), new_points


LANDMARK_NAMES = [
    "glabella",
    "nasion",
    "rhinion",
    "nasal_tip",
    "subnasale",
    "alare_r",
    "alare_l",
    "zygion_r",
    "zygion_l",
]

N_LANDMARKS = len(LANDMARK_NAMES)
OUTPUT_DIM = N_LANDMARKS * 3

ALARE_R_IDX, ALARE_L_IDX = 5, 6
ZYGION_R_IDX, ZYGION_L_IDX = 7, 8

MAX_POINTS = 16384
PATCH_POINTS = 512
SNAP_KNN = 10

SIGMA_DEFAULT = 0.04
LAMBDA_H = 0.5
CENTER_JITTER = 0.05
MIX_PROB = 0.5

RADIUS_MIN_DEFAULT = 0.25
RADIUS_MIN_BY_LM = {
    "rhinion": 0.35,
    "nasal_tip": 0.55,
    "zygion_r": 0.57,
    "zygion_l": 0.51,
}
RADIUS_MAX = 0.80

SA1_RADII_S1 = [0.1, 0.2, 0.4]
SA2_RADII_S1 = [0.2, 0.4, 0.8]

SA1_RADII_S2 = [0.05, 0.1, 0.2]
SA2_RADII_S2 = [0.1, 0.2, 0.4]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def stable_seed_from_name(name):
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


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


def patient_kfold_indices(names, k=5, seed=42):
    patient_to_idx = defaultdict(list)

    for i, name in enumerate(names):
        patient_to_idx[patient_id(name)].append(i)

    patients = sorted(patient_to_idx)
    n_folds = max(2, min(k, len(patients)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_patient_idx, val_patient_idx in kf.split(patients):
        train_idx, val_idx = [], []

        for i in train_patient_idx:
            train_idx.extend(patient_to_idx[patients[i]])
        for i in val_patient_idx:
            val_idx.extend(patient_to_idx[patients[i]])

        yield np.array(train_idx), np.array(val_idx)


def build_s1_model(cfg):
    if cfg.backbone == "pt":
        return PointTransformerReg(
            output_dim=OUTPUT_DIM,
            dropout=cfg.dropout_s1,
            dims=(128, 256, 512),
            n_pts=(2048, 512, 128),
            k=16,
        )

    return PointNet2RegMSG(
        output_dim=OUTPUT_DIM,
        normal_channel=False,
        dropout=cfg.dropout_s1,
        sa1_radii=SA1_RADII_S1,
        sa2_radii=SA2_RADII_S1,
    )


class JointHeatmapS2(nn.Module):
    def __init__(self, n_landmarks=9, dropout=0.3, sa1_radii=None):
        super().__init__()
        sa1_radii = sa1_radii or SA1_RADII_S2

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radii=sa1_radii,
            nsamples=[16, 32, 64],
            mlps=[[0, 32, 32, 64], [0, 64, 64, 128], [0, 64, 96, 128]],
            use_xyz=True,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radii=SA2_RADII_S2,
            nsamples=[32, 64, 128],
            mlps=[[320, 64, 64, 128], [320, 128, 128, 256], [320, 128, 128, 256]],
            use_xyz=True,
        )
        self.global_mlp = nn.Sequential(
            nn.Conv1d(640, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(320 + 128, 128, 1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(128, 64, 1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, 1, 1),
                )
                for _ in range(n_landmarks)
            ]
        )
        self.n_landmarks = int(n_landmarks)

    def forward(self, patches_list):
        batch_size = patches_list[0].shape[0]

        all_patches = torch.cat(patches_list, dim=0)
        xyz_t = all_patches.transpose(1, 2).contiguous()

        l1_xyz, l1_features = self.sa1(xyz_t, None)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        global_features = self.global_mlp(l2_features).max(dim=2)[0]

        preds, scores_list, l1_xyz_list = [], [], []

        for k in range(self.n_landmarks):
            local_features = l1_features[k * batch_size : (k + 1) * batch_size]
            global_k = global_features[k * batch_size : (k + 1) * batch_size]
            xyz_k = l1_xyz[k * batch_size : (k + 1) * batch_size]

            global_expanded = global_k.unsqueeze(2).expand(-1, -1, PATCH_POINTS)
            features = torch.cat([local_features, global_expanded], dim=1)

            scores = self.heads[k](features).squeeze(1)
            weights = torch.softmax(scores, dim=1)
            pred = (weights.unsqueeze(-1) * xyz_k).sum(dim=1)

            preds.append(pred)
            scores_list.append(scores)
            l1_xyz_list.append(xyz_k)

        return preds, scores_list, l1_xyz_list


def coord_loss(pred, target):
    return F.smooth_l1_loss(pred, target)


def gaussian_heatmap_kl(scores, l1_xyz, gt_residual, sigma):
    gt = gt_residual.unsqueeze(1).float()
    dist = torch.norm(l1_xyz.float() - gt, dim=2)

    target = torch.exp(-(dist**2) / (2 * sigma**2))
    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)

    log_prob = F.log_softmax(scores.float(), dim=1)
    return (target * (torch.log(target + 1e-8) - log_prob)).sum(dim=1).mean()


def joint_loss(preds, scores_list, l1_xyz_list, residuals_batch, sigma):
    loss = 0.0

    for k in range(N_LANDMARKS):
        residual = residuals_batch[:, k, :]
        loss = loss + coord_loss(preds[k], residual)
        loss = loss + LAMBDA_H * gaussian_heatmap_kl(scores_list[k], l1_xyz_list[k], residual, sigma)

    return loss / N_LANDMARKS


def fps(points, n_points, seed=None):
    if DEVICE.type == "cuda" and points.shape[0] >= n_points:
        try:
            t = torch.from_numpy(points).float().unsqueeze(0).to(DEVICE)
            idx = furthest_point_sample(t, n_points)
            return points[idx[0].cpu().numpy()]
        except Exception:
            pass

    rng = np.random.RandomState(seed if seed is not None else 0)
    idx = rng.choice(points.shape[0], n_points, replace=points.shape[0] < n_points)
    return points[idx]


def sample_mesh_uniform(vertices, triangles, n_points, seed=None):
    rng = np.random.RandomState(seed if seed is not None else 0)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.maximum(areas, 1e-12)
    probs = areas / areas.sum()

    chosen = rng.choice(len(triangles), size=n_points, p=probs)

    r1 = rng.rand(n_points, 1).astype(np.float32)
    r2 = rng.rand(n_points, 1).astype(np.float32)

    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]

    points = (1.0 - r1 - r2) * v0[chosen] + r1 * v1[chosen] + r2 * v2[chosen]
    return points.astype(np.float32)


def augment_batch(points, labels, device, flip_prob=0.2):
    points = points.to(device)
    labels = labels.to(device)

    batch_size, _, n_points = points.shape

    theta = torch.rand(batch_size, 1, 1, device=device) * (np.pi / 6) - (np.pi / 12)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rot = torch.zeros(batch_size, 3, 3, device=device)
    rot[:, 0, 0] = cos_t.flatten()
    rot[:, 0, 1] = -sin_t.flatten()
    rot[:, 1, 0] = sin_t.flatten()
    rot[:, 1, 1] = cos_t.flatten()
    rot[:, 2, 2] = 1.0

    points_aug = torch.bmm(points.transpose(1, 2), rot)

    if labels.dim() == 3:
        labels_aug = torch.bmm(
            labels.view(batch_size * N_LANDMARKS, 1, 3),
            rot.repeat_interleave(N_LANDMARKS, dim=0),
        ).view(batch_size, N_LANDMARKS, 3)
    else:
        labels_aug = torch.bmm(labels.view(batch_size, 1, 3), rot).view(batch_size, 3)

    scale = torch.rand(batch_size, 1, 1, device=device) * 0.10 + 0.95
    shift = torch.rand(batch_size, 1, 3, device=device) * 0.04 - 0.02

    points_aug = points_aug * scale + shift
    points_aug = points_aug + torch.randn(batch_size, n_points, 3, device=device) * 0.005

    if labels.dim() == 3:
        labels_aug = labels_aug * scale + shift
    else:
        labels_aug = labels_aug * scale.squeeze(-1) + shift.squeeze(1)

    flip_idx = (torch.rand(batch_size, device=device) < float(flip_prob)).nonzero(as_tuple=True)[0]

    if len(flip_idx) > 0:
        points_aug[flip_idx, :, 0] = -points_aug[flip_idx, :, 0]

        if labels.dim() == 3:
            labels_aug[flip_idx, :, 0] = -labels_aug[flip_idx, :, 0]

            tmp = labels_aug[flip_idx, ALARE_R_IDX].clone()
            labels_aug[flip_idx, ALARE_R_IDX] = labels_aug[flip_idx, ALARE_L_IDX]
            labels_aug[flip_idx, ALARE_L_IDX] = tmp

            tmp = labels_aug[flip_idx, ZYGION_R_IDX].clone()
            labels_aug[flip_idx, ZYGION_R_IDX] = labels_aug[flip_idx, ZYGION_L_IDX]
            labels_aug[flip_idx, ZYGION_L_IDX] = tmp
        else:
            labels_aug[flip_idx, 0] = -labels_aug[flip_idx, 0]

    return points_aug.transpose(1, 2), labels_aug


def snap_to_nearest(pred, pc_full):
    idx = np.argmin(np.linalg.norm(pc_full - pred, axis=1))
    return pc_full[idx]


def closest_on_triangles_batch(point, v0, v1, v2):
    point = point.astype(np.float64)
    v0 = v0.astype(np.float64)
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)

    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0

    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)

    bp = point - v1
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)

    cp = point - v2
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc

    result = np.empty_like(v0, dtype=np.float64)

    mask_a = (d1 <= 0) & (d2 <= 0)
    result[mask_a] = v0[mask_a]

    mask_b = (~mask_a) & (d3 >= 0) & (d4 <= d3)
    result[mask_b] = v1[mask_b]

    mask_c = (~mask_a) & (~mask_b) & (d6 >= 0) & (d5 <= d6)
    result[mask_c] = v2[mask_c]

    done = mask_a | mask_b | mask_c

    mask_ab = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    if mask_ab.any():
        s_den = d1[mask_ab] - d3[mask_ab]
        s = np.where(np.abs(s_den) > 1e-30, np.clip(d1[mask_ab] / s_den, 0, 1), 0.0)
        result[mask_ab] = v0[mask_ab] + s[:, None] * ab[mask_ab]

    done = done | mask_ab

    mask_ac = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    if mask_ac.any():
        w_den = d2[mask_ac] - d6[mask_ac]
        w = np.where(np.abs(w_den) > 1e-30, np.clip(d2[mask_ac] / w_den, 0, 1), 0.0)
        result[mask_ac] = v0[mask_ac] + w[:, None] * ac[mask_ac]

    done = done | mask_ac

    mask_bc = (~done) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    if mask_bc.any():
        numerator = d4[mask_bc] - d3[mask_bc]
        denominator = numerator + (d5[mask_bc] - d6[mask_bc])
        w = np.where(np.abs(denominator) > 1e-30, np.clip(numerator / denominator, 0, 1), 0.0)
        result[mask_bc] = v1[mask_bc] + w[:, None] * (v2[mask_bc] - v1[mask_bc])

    done = done | mask_bc
    mask_inside = ~done

    if mask_inside.any():
        d = denom[mask_inside]
        safe = np.abs(d) > 1e-30
        v = np.where(safe, vb[mask_inside] / np.where(safe, d, 1.0), 1.0 / 3.0)
        w = np.where(safe, vc[mask_inside] / np.where(safe, d, 1.0), 1.0 / 3.0)
        result[mask_inside] = v0[mask_inside] + v[:, None] * ab[mask_inside] + w[:, None] * ac[mask_inside]

    return result


def snap_to_mesh(pred, pc_full, triangles):
    if triangles is None or len(triangles) == 0:
        return snap_to_nearest(pred, pc_full)

    v0 = pc_full[triangles[:, 0]]
    v1 = pc_full[triangles[:, 1]]
    v2 = pc_full[triangles[:, 2]]

    closest = closest_on_triangles_batch(pred, v0, v1, v2)
    dists = np.linalg.norm(closest - pred.astype(np.float64), axis=1)
    return closest[np.argmin(dists)].astype(np.float32)


def percentile_stats(errors):
    errors = np.asarray(errors)
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p75": float(np.percentile(errors, 75)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
    }


def load_data():
    sample_names = sorted(
        d for d in os.listdir(EXPORT_ROOT) if os.path.isdir(os.path.join(EXPORT_ROOT, d))
    )

    X_list, Y_list, scale_list, names, full_clouds, triangles_list = [], [], [], [], [], []
    n_with_mesh = 0

    for name in sample_names:
        pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        lm_path = os.path.join(EXPORT_ROOT, name, "nose_landmarks.npy")
        tri_path = os.path.join(EXPORT_ROOT, name, "triangles.npy")

        if not (os.path.exists(pc_path) and os.path.exists(lm_path)):
            continue

        pc = np.load(pc_path).astype(np.float32)
        lm = np.load(lm_path).astype(np.float32)

        if pc.shape[0] == 0:
            continue
        if lm.ndim == 1:
            lm = lm.reshape(-1, 3)
        if lm.shape[0] < 9:
            continue

        lm9 = lm[-9:]

        centroid = pc.mean(0)
        pc_centered = pc - centroid

        scale = float(np.std(pc_centered))
        scale = scale if scale > 1e-6 else 1.0

        pc_norm = pc_centered / scale
        lm_norm = (lm9 - centroid) / scale

        seed = stable_seed_from_name(name)

        if os.path.exists(tri_path):
            triangles = np.load(tri_path)
            sampled = sample_mesh_uniform(pc_norm, triangles, MAX_POINTS, seed=seed)
            n_with_mesh += 1
        else:
            triangles = None
            sampled = fps(pc_norm, MAX_POINTS, seed=seed)

        X_list.append(sampled.T.astype(np.float32))
        Y_list.append(lm_norm.astype(np.float32))
        scale_list.append(np.float32(scale))
        names.append(name)
        full_clouds.append(pc_norm.astype(np.float32))
        triangles_list.append(triangles)

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    scales = np.array(scale_list, dtype=np.float32)

    print(f"[data] samples={len(X)}, with_mesh={n_with_mesh}")
    return X, Y, scales, names, full_clouds, triangles_list


def estimate_stage2_patch_params(coarse, gt):
    err = np.linalg.norm(coarse - gt, axis=2)

    p80 = np.percentile(err, 80, axis=0)
    p95 = np.percentile(err, 95, axis=0)

    jitter = np.maximum(CENTER_JITTER, p80).astype(np.float32)

    radii = []
    for k, name in enumerate(LANDMARK_NAMES):
        base_radius = max(RADIUS_MIN_DEFAULT, float(p95[k]) * 1.2)
        floor_radius = RADIUS_MIN_BY_LM.get(name, RADIUS_MIN_DEFAULT)
        radius = min(max(base_radius, floor_radius), RADIUS_MAX)
        radii.append(float(radius))

    print("\n[Stage2 patch params]")
    for k, name in enumerate(LANDMARK_NAMES):
        print(
            f"  {name:<12} "
            f"p80={p80[k]:.4f}  "
            f"p95={p95[k]:.4f}  "
            f"jitter={jitter[k]:.4f}  "
            f"radius={radii[k]:.4f}"
        )

    return radii, jitter.tolist()


class JointPatchDataset(Dataset):
    def __init__(self, X, Y, coarse, radii, jitter, mode="train", mix_prob=0.0):
        self.X = X
        self.Y = Y
        self.coarse = coarse
        self.radii = radii
        self.jitter = jitter
        self.mode = mode
        self.mix_prob = float(mix_prob)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        pc = self.X[i].T
        gt_all = self.Y[i]
        ctr_all = self.coarse[i]

        patches, residuals = [], []

        for k in range(N_LANDMARKS):
            gt = gt_all[k]
            center = ctr_all[k]

            if self.mode == "train" and np.random.rand() < self.mix_prob:
                center = gt

            jitter_k = float(self.jitter[k]) if isinstance(self.jitter, (list, tuple, np.ndarray)) else float(self.jitter)
            if self.mode == "train" and jitter_k > 0:
                center = center + np.random.normal(0, jitter_k, 3).astype(np.float32)

            dists = np.linalg.norm(pc - center, axis=1)
            inside = np.where(dists < self.radii[k])[0]

            if inside.size == 0:
                inside = np.argsort(dists)[:PATCH_POINTS]

            if inside.size < PATCH_POINTS:
                extra = np.random.choice(inside, PATCH_POINTS - inside.size, replace=True)
                inside = np.concatenate([inside, extra])
            else:
                inside = np.random.choice(inside, PATCH_POINTS, replace=False)

            patch = (pc[inside] - center).T.astype(np.float32)
            residual = (gt - center).astype(np.float32)

            patches.append(patch)
            residuals.append(residual)

        return patches, residuals


def joint_collate(batch):
    patches_by_landmark = [[] for _ in range(N_LANDMARKS)]
    residuals = []

    for patches, residual in batch:
        for k in range(N_LANDMARKS):
            patches_by_landmark[k].append(patches[k])
        residuals.append(np.stack(residual))

    patches_tensor = [
        torch.from_numpy(np.stack(patches_by_landmark[k])).float()
        for k in range(N_LANDMARKS)
    ]
    residuals_tensor = torch.from_numpy(np.stack(residuals)).float()

    return patches_tensor, residuals_tensor


def predict_stage1(model, X, batch_size=16):
    model.eval()
    preds = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).float().to(DEVICE)
            pred = model(batch).cpu().numpy().reshape(-1, N_LANDMARKS, 3)
            preds.append(pred)

    return np.concatenate(preds, axis=0)


def train_stage1_kfold(X_train, Y_train, scales_train, names_train, cfg):
    print(f"\n[Stage1 K-fold] backbone={cfg.backbone}, epochs={cfg.epochs_s1}")

    best_epochs = []

    for fold, (tr_idx, val_idx) in enumerate(
        patient_kfold_indices(names_train, k=cfg.k_folds, seed=cfg.seed),
        start=1,
    ):
        X_tr = X_train[tr_idx]
        Y_tr = Y_train[tr_idx]
        X_val = X_train[val_idx]
        Y_val = Y_train[val_idx]
        scales_val = scales_train[val_idx]

        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_tr).float(),
                torch.from_numpy(Y_tr.reshape(len(Y_tr), -1)).float(),
            ),
            batch_size=cfg.batch_size_s1,
            shuffle=True,
            drop_last=False,
        )

        model = build_s1_model(cfg).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_s1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs_s1,
            eta_min=5e-6,
        )

        best_val = float("inf")
        best_epoch = cfg.epochs_s1

        print(f"\n  Fold {fold}: train={len(X_tr)}, val={len(X_val)}")

        for epoch in range(1, cfg.epochs_s1 + 1):
            model.train()

            for points, labels in loader:
                labels = labels.view(-1, N_LANDMARKS, 3)
                points_aug, labels_aug = augment_batch(points, labels, DEVICE, flip_prob=cfg.flip_prob)

                optimizer.zero_grad()
                pred = model(points_aug)
                loss = F.smooth_l1_loss(pred, labels_aug.reshape(labels_aug.size(0), -1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            scheduler.step()

            if epoch % cfg.print_interval_s1 == 0 or epoch == cfg.epochs_s1:
                pred_val = predict_stage1(model, X_val)
                val_err = np.linalg.norm(pred_val - Y_val, axis=2) * scales_val[:, None]
                val_mm = float(val_err.mean())

                if val_mm < best_val:
                    best_val = val_mm
                    best_epoch = epoch

                print(f"    epoch {epoch:>3}/{cfg.epochs_s1}  val={val_mm:.3f}  best={best_val:.3f}")

        best_epochs.append(best_epoch)

    selected_epoch = int(np.median(best_epochs))
    print(f"\n[Stage1] selected epoch={selected_epoch}, fold_epochs={best_epochs}")

    return selected_epoch


def train_stage1_full(X_train, Y_train, best_epoch, cfg):
    print(f"\n[Stage1 full retrain] epochs={best_epoch}, samples={len(X_train)}")

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train.reshape(len(Y_train), -1)).float(),
        ),
        batch_size=cfg.batch_size_s1,
        shuffle=True,
        drop_last=False,
    )

    model = build_s1_model(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_s1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=best_epoch,
        eta_min=5e-6,
    )

    for epoch in range(1, best_epoch + 1):
        model.train()
        total_loss = 0.0

        for points, labels in loader:
            labels = labels.view(-1, N_LANDMARKS, 3)
            points_aug, labels_aug = augment_batch(points, labels, DEVICE, flip_prob=cfg.flip_prob)

            optimizer.zero_grad()
            pred = model(points_aug)
            loss = F.smooth_l1_loss(pred, labels_aug.reshape(labels_aug.size(0), -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % cfg.print_interval_s1 == 0 or epoch == best_epoch:
            print(f"  epoch {epoch:>3}/{best_epoch}  loss={total_loss / max(1, len(loader)):.5f}")

    return model


def predict_stage2(model, X, coarse, radii, eval_seed=42):
    model.eval()
    rng = np.random.RandomState(eval_seed)

    refined_all = np.empty_like(coarse, dtype=np.float32)

    for i in range(len(X)):
        pc = X[i].T
        patches = []

        for k in range(N_LANDMARKS):
            center = coarse[i, k]
            dists = np.linalg.norm(pc - center, axis=1)
            inside = np.where(dists < radii[k])[0]

            if inside.size == 0:
                inside = np.argsort(dists)[:PATCH_POINTS]

            if inside.size < PATCH_POINTS:
                extra = rng.choice(inside, PATCH_POINTS - inside.size, replace=True)
                inside = np.concatenate([inside, extra])
            else:
                inside = rng.choice(inside, PATCH_POINTS, replace=False)

            patch = (pc[inside] - center).T.astype(np.float32)
            patches.append(torch.from_numpy(patch[None]).float().to(DEVICE))

        with torch.no_grad():
            residuals, _, _ = model(patches)

        for k in range(N_LANDMARKS):
            refined_all[i, k] = coarse[i, k] + residuals[k].cpu().numpy().reshape(3)

    return refined_all


def train_stage2_kfold(X_train, Y_train, scales_train, names_train, coarse_train, radii, jitter, cfg):
    print(f"\n[Stage2 K-fold] sigma={cfg.sigma}, epochs={cfg.epochs_s2}")

    best_epochs = []

    for fold, (tr_idx, val_idx) in enumerate(
        patient_kfold_indices(names_train, k=cfg.k_folds, seed=cfg.seed),
        start=1,
    ):
        X_tr = X_train[tr_idx]
        Y_tr = Y_train[tr_idx]
        coarse_tr = coarse_train[tr_idx]

        X_val = X_train[val_idx]
        Y_val = Y_train[val_idx]
        scales_val = scales_train[val_idx]
        coarse_val = coarse_train[val_idx]

        dataset = JointPatchDataset(
            X_tr,
            Y_tr,
            coarse_tr,
            radii,
            jitter,
            mode="train",
            mix_prob=cfg.mix_prob,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size_s2,
            shuffle=True,
            collate_fn=joint_collate,
            drop_last=False,
        )

        model = JointHeatmapS2(
            n_landmarks=N_LANDMARKS,
            dropout=cfg.dropout_s2,
            sa1_radii=SA1_RADII_S2,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_s2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_decay_step_s2,
            gamma=cfg.lr_decay_gamma_s2,
        )

        best_val = float("inf")
        best_epoch = cfg.epochs_s2

        print(f"\n  Fold {fold}: train={len(X_tr)}, val={len(X_val)}")

        for epoch in range(1, cfg.epochs_s2 + 1):
            model.train()

            for patches_list, residuals_batch in loader:
                patches_list = [p.to(DEVICE) for p in patches_list]
                residuals_batch = residuals_batch.to(DEVICE)

                optimizer.zero_grad()
                preds, scores_list, l1_xyz_list = model(patches_list)
                loss = joint_loss(preds, scores_list, l1_xyz_list, residuals_batch, cfg.sigma)
                loss.backward()
                optimizer.step()

            scheduler.step()

            if epoch % cfg.print_interval_s2 == 0 or epoch == cfg.epochs_s2:
                refined_val = predict_stage2(model, X_val, coarse_val, radii, eval_seed=cfg.seed)
                val_err = np.linalg.norm(refined_val - Y_val, axis=2) * scales_val[:, None]
                val_mm = float(val_err.mean())

                if val_mm < best_val:
                    best_val = val_mm
                    best_epoch = epoch

                print(f"    epoch {epoch:>3}/{cfg.epochs_s2}  val={val_mm:.3f}  best={best_val:.3f}")

        best_epochs.append(best_epoch)

    selected_epoch = int(np.median(best_epochs))
    print(f"\n[Stage2] selected epoch={selected_epoch}, fold_epochs={best_epochs}")

    return selected_epoch


def train_stage2_full(X_train, Y_train, coarse_train, radii, jitter, best_epoch, cfg):
    print(f"\n[Stage2 full retrain] epochs={best_epoch}, samples={len(X_train)}")

    dataset = JointPatchDataset(
        X_train,
        Y_train,
        coarse_train,
        radii,
        jitter,
        mode="train",
        mix_prob=cfg.mix_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_s2,
        shuffle=True,
        collate_fn=joint_collate,
        drop_last=False,
    )

    model = JointHeatmapS2(
        n_landmarks=N_LANDMARKS,
        dropout=cfg.dropout_s2,
        sa1_radii=SA1_RADII_S2,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_s2)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_decay_step_s2,
        gamma=cfg.lr_decay_gamma_s2,
    )

    for epoch in range(1, best_epoch + 1):
        model.train()
        total_loss = 0.0

        for patches_list, residuals_batch in loader:
            patches_list = [p.to(DEVICE) for p in patches_list]
            residuals_batch = residuals_batch.to(DEVICE)

            optimizer.zero_grad()
            preds, scores_list, l1_xyz_list = model(patches_list)
            loss = joint_loss(preds, scores_list, l1_xyz_list, residuals_batch, cfg.sigma)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % cfg.print_interval_s2 == 0 or epoch == best_epoch:
            print(f"  epoch {epoch:>3}/{best_epoch}  loss={total_loss / max(1, len(loader)):.5f}")

    return model


def evaluate(model_s1, model_s2, X, Y, scales, full_clouds, triangles, radii, cfg):
    coarse = predict_stage1(model_s1, X)
    refined = predict_stage2(model_s2, X, coarse, radii, eval_seed=cfg.seed)

    coarse_err = np.linalg.norm(coarse - Y, axis=2) * scales[:, None]
    refined_err = np.linalg.norm(refined - Y, axis=2) * scales[:, None]

    snap_nearest = np.empty_like(refined, dtype=np.float32)
    snap_mesh = np.empty_like(refined, dtype=np.float32)

    for i in range(len(X)):
        tri = triangles[i] if triangles is not None else None

        for k in range(N_LANDMARKS):
            snap_nearest[i, k] = snap_to_nearest(refined[i, k], full_clouds[i])
            snap_mesh[i, k] = snap_to_mesh(refined[i, k], full_clouds[i], tri)

    snap_nearest_err = np.linalg.norm(snap_nearest - Y, axis=2) * scales[:, None]
    snap_mesh_err = np.linalg.norm(snap_mesh - Y, axis=2) * scales[:, None]

    print("\n[TEST]")
    print(f"  {'landmark':<12} {'coarse':>8} {'soft':>8} {'snap-pt':>9} {'snap-mesh':>10}")

    per_landmark = {}

    for k, name in enumerate(LANDMARK_NAMES):
        print(
            f"  {name:<12} "
            f"{coarse_err[:, k].mean():>8.3f} "
            f"{refined_err[:, k].mean():>8.3f} "
            f"{snap_nearest_err[:, k].mean():>9.3f} "
            f"{snap_mesh_err[:, k].mean():>10.3f}"
        )

        per_landmark[name] = {
            "coarse": percentile_stats(coarse_err[:, k]),
            "soft_argmax": percentile_stats(refined_err[:, k]),
            "snap_nearest": percentile_stats(snap_nearest_err[:, k]),
            "snap_mesh": percentile_stats(snap_mesh_err[:, k]),
        }

    print(
        f"  {'OVERALL':<12} "
        f"{coarse_err.mean():>8.3f} "
        f"{refined_err.mean():>8.3f} "
        f"{snap_nearest_err.mean():>9.3f} "
        f"{snap_mesh_err.mean():>10.3f}  mm"
    )

    return {
        "per_landmark": per_landmark,
        "overall": {
            "coarse": percentile_stats(coarse_err.ravel()),
            "soft_argmax": percentile_stats(refined_err.ravel()),
            "snap_nearest": percentile_stats(snap_nearest_err.ravel()),
            "snap_mesh": percentile_stats(snap_mesh_err.ravel()),
        },
    }


def train(cfg):
    set_seed(cfg.seed)

    print(f"Device: {DEVICE}")
    print(f"Backbone: {cfg.backbone}")
    print(f"Seed: {cfg.seed}, sigma: {cfg.sigma}")

    X, Y, scales, names, full_clouds, triangles = load_data()

    train_idx, test_idx = patient_based_split(
        names,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    scales_train = scales[train_idx]
    names_train = [names[i] for i in train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    scales_test = scales[test_idx]
    full_clouds_test = [full_clouds[i] for i in test_idx]
    triangles_test = [triangles[i] for i in test_idx]

    train_patients = {patient_id(names[i]) for i in train_idx}
    test_patients = {patient_id(names[i]) for i in test_idx}
    assert train_patients.isdisjoint(test_patients), "DATA LEAKAGE: patient appears in both train and test."

    print(f"Samples: total={len(X)}, train={len(X_train)}, test={len(X_test)}")
    print(f"Patients: train={len(train_patients)}, test={len(test_patients)}")

    best_epoch_s1 = train_stage1_kfold(X_train, Y_train, scales_train, names_train, cfg)
    model_s1 = train_stage1_full(X_train, Y_train, best_epoch_s1, cfg)

    coarse_train = predict_stage1(model_s1, X_train)
    radii, jitter = estimate_stage2_patch_params(coarse_train, Y_train)

    best_epoch_s2 = train_stage2_kfold(
        X_train,
        Y_train,
        scales_train,
        names_train,
        coarse_train,
        radii,
        jitter,
        cfg,
    )
    model_s2 = train_stage2_full(
        X_train,
        Y_train,
        coarse_train,
        radii,
        jitter,
        best_epoch_s2,
        cfg,
    )

    results = evaluate(
        model_s1,
        model_s2,
        X_test,
        Y_test,
        scales_test,
        full_clouds_test,
        triangles_test,
        radii,
        cfg,
    )

    run_name = f"joint_heatmap_{cfg.backbone}_seed{cfg.seed}_sigma{cfg.sigma}"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    s1_path = os.path.join(MODELS_DIR, f"{run_name}_stage1.pth")
    s2_path = os.path.join(MODELS_DIR, f"{run_name}_stage2.pth")
    result_path = os.path.join(RESULTS_DIR, f"{run_name}_summary.json")

    torch.save(model_s1.state_dict(), s1_path)
    torch.save(model_s2.state_dict(), s2_path)

    summary = {
        "run_name": run_name,
        "stage1_model_path": s1_path,
        "stage2_model_path": s2_path,
        "seed": int(cfg.seed),
        "sigma": float(cfg.sigma),
        "backbone": cfg.backbone,
        "split": f"patient-based {int((1 - cfg.test_fraction) * 100)}/{int(cfg.test_fraction * 100)}",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "best_epoch_s1": int(best_epoch_s1),
        "best_epoch_s2": int(best_epoch_s2),
        "stage2_patch_params": {
            "radius": {name: float(radii[k]) for k, name in enumerate(LANDMARK_NAMES)},
            "jitter": {name: float(jitter[k]) for k, name in enumerate(LANDMARK_NAMES)},
        },
        "results": results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining complete")
    print(f"Stage1 saved: {s1_path}")
    print(f"Stage2 saved: {s2_path}")
    print(f"Summary saved: {result_path}")
    print(f"Test soft-argmax mean: {results['overall']['soft_argmax']['mean']:.3f} mm")
    print(f"Test snap-mesh mean: {results['overall']['snap_mesh']['mean']:.3f} mm")


def get_config():
    return SimpleNamespace(
        seed=42,
        test_fraction=0.20,
        k_folds=5,
        backbone="pn2",
        sigma=SIGMA_DEFAULT,
        batch_size_s1=8,
        epochs_s1=300,
        lr_s1=3e-4,
        dropout_s1=0.4,
        batch_size_s2=4,
        epochs_s2=300,
        lr_s2=1e-3,
        lr_decay_step_s2=80,
        lr_decay_gamma_s2=0.5,
        dropout_s2=0.3,
        mix_prob=MIX_PROB,
        flip_prob=0.2,
        grad_clip_norm=1.0,
        print_interval_s1=20,
        print_interval_s2=30,
    )


if __name__ == "__main__":
    train(get_config())