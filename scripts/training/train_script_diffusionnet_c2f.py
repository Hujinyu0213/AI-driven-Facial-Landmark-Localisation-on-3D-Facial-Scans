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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DIFFNET_SRC = os.path.join(ROOT_DIR, "diffusion-net-repo", "src")

for path in (ROOT_DIR, BASE_DIR, DIFFNET_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

import diffusion_net
import scripts.training.train_script_pointnet2_c2f as base


N_LANDMARKS = base.N_LANDMARKS
LANDMARK_NAMES = base.LANDMARK_NAMES
PATCH_POINTS_DEFAULT = base.PATCH_POINTS
DEVICE = getattr(base, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

DIFFNET_C_WIDTH = 128
DIFFNET_N_BLOCK = 4
DIFFNET_K_EIG = 64
DIFFNET_INPUT_FEATURES = "xyz"
DIFFNET_DROPOUT = True
OP_CACHE_DIR = os.path.join(ROOT_DIR, "data", "diffnet_op_cache")

COND_DIM = 128
COND_DROPOUT = 0.1
W_COARSE = 0.5
W_STAGE2 = 1.0
W_REFINED = 0.5

RADIUS_MIN_DEFAULT = 0.25
RADIUS_MIN_BY_LM = {
    "glabella": 0.38,
    "nasion": 0.30,
    "rhinion": 0.35,
    "nasal_tip": 0.55,
    "subnasale": 0.38,
    "alare_r": 0.37,
    "alare_l": 0.36,
    "zygion_r": 0.57,
    "zygion_l": 0.51,
}
RADIUS_MAX = 0.80

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


def resolve_diffnet_device(mode="auto"):
    mode = str(mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return torch.device("cuda")
    if os.name == "nt":
        return torch.device("cpu")
    return DEVICE


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

    for train_pat_idx, val_pat_idx in kf.split(patients):
        train_local, val_local = [], []
        for i in train_pat_idx:
            train_local.extend(patient_to_local[patients[i]])
        for i in val_pat_idx:
            val_local.extend(patient_to_local[patients[i]])
        yield np.array(train_local), np.array(val_local)


def precompute_diffnet_operators(verts_list, faces_list, k_eig=DIFFNET_K_EIG, op_cache_dir=OP_CACHE_DIR):
    if op_cache_dir is not None:
        os.makedirs(op_cache_dir, exist_ok=True)

    print(f"[DiffNet] computing operators: samples={len(verts_list)}, k_eig={k_eig}")
    ops = []
    for i, (verts, faces) in enumerate(zip(verts_list, faces_list), start=1):
        try:
            op = diffusion_net.geometry.get_operators(
                verts,
                faces,
                k_eig=k_eig,
                op_cache_dir=op_cache_dir,
            )
        except Exception as exc:
            fallback_k = min(int(k_eig), 32)
            print(f"[DiffNet] sample {i}: k_eig={k_eig} failed ({exc}); retrying with {fallback_k}")
            op = diffusion_net.geometry.get_operators(
                verts,
                faces,
                k_eig=fallback_k,
                op_cache_dir=op_cache_dir,
            )
        ops.append(op)

        if i % 10 == 0 or i == len(verts_list):
            print(f"  operators {i}/{len(verts_list)}")

    return ops


def load_data_with_diffnet_ops(k_eig=DIFFNET_K_EIG, input_features=DIFFNET_INPUT_FEATURES):
    X, Y, scales, names, full_clouds, triangles = base.load_data()

    verts_list = [torch.from_numpy(pc).float() for pc in full_clouds]
    faces_list = [
        torch.from_numpy(tri).long()
        if tri is not None and len(tri) > 0
        else torch.zeros((0, 3), dtype=torch.long)
        for tri in triangles
    ]

    ops = precompute_diffnet_operators(
        verts_list,
        faces_list,
        k_eig=k_eig,
        op_cache_dir=OP_CACHE_DIR,
    )

    diffnet_data = []
    for verts, op in zip(verts_list, ops):
        _, mass, L, evals, evecs, gradX, gradY = op
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

    return X, Y, scales, names, full_clouds, triangles, diffnet_data


class DiffNetCoarseWithGlobalFeature(nn.Module):
    def __init__(self, c_in=3, c_width=128, n_block=4, global_dim=256, n_landmarks=9, dropout=True):
        super().__init__()
        self.global_dim = int(global_dim)
        self.diffnet = diffusion_net.layers.DiffusionNet(
            C_in=c_in,
            C_out=global_dim,
            C_width=c_width,
            N_block=n_block,
            last_activation=None,
            outputs_at="global_mean",
            dropout=dropout,
            with_gradient_features=True,
            with_gradient_rotations=True,
            diffusion_method="spectral",
        )
        self.coarse_head = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_landmarks * 3),
        )

    def forward(self, features, mass, L, evals, evecs, gradX, gradY):
        feat = self.diffnet(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        return self.coarse_head(feat), feat


class JointHeatmapS2Conditioned(nn.Module):
    def __init__(
        self,
        n_landmarks=9,
        dropout=0.3,
        stage1_feat_dim=256,
        cond_dim=128,
        cond_dropout=0.1,
        use_conditioning=True,
    ):
        super().__init__()
        self.n_landmarks = int(n_landmarks)
        self.use_conditioning = bool(use_conditioning)
        self.cond_dim = int(cond_dim) if self.use_conditioning else 0

        self.sa1 = base.PointNetSetAbstractionMsg(
            npoint=512,
            radii=base.SA1_RADII_S2,
            nsamples=[16, 32, 64],
            mlps=[[0, 32, 32, 64], [0, 64, 64, 128], [0, 64, 96, 128]],
            use_xyz=True,
        )
        self.sa2 = base.PointNetSetAbstractionMsg(
            npoint=128,
            radii=base.SA2_RADII_S2,
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

        self.cond_proj = (
            nn.Sequential(
                nn.Linear(stage1_feat_dim, self.cond_dim),
                nn.LayerNorm(self.cond_dim),
                nn.ReLU(),
                nn.Dropout(cond_dropout),
            )
            if self.use_conditioning
            else nn.Identity()
        )

        head_in = 320 + 128 + self.cond_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(head_in, 128, 1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(128, 64, 1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, 1, 1),
                )
                for _ in range(self.n_landmarks)
            ]
        )

    def forward(self, patches_list, stage1_global_feat=None):
        if self.use_conditioning and stage1_global_feat is None:
            raise RuntimeError("stage1_global_feat is required when conditioning is enabled.")

        batch_size = patches_list[0].shape[0]
        all_patches = torch.cat(patches_list, dim=0)
        xyz_t = all_patches.transpose(1, 2).contiguous()

        l1_xyz, l1_f = self.sa1(xyz_t, None)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f)
        global_patch_feat = self.global_mlp(l2_f).max(dim=2)[0]

        cond = self.cond_proj(stage1_global_feat) if self.use_conditioning else None

        preds, scores_list, l1_xyz_list = [], [], []
        for k in range(self.n_landmarks):
            lf_k = l1_f[k * batch_size : (k + 1) * batch_size]
            g_k = global_patch_feat[k * batch_size : (k + 1) * batch_size]
            xyz_k = l1_xyz[k * batch_size : (k + 1) * batch_size]
            n_patch = xyz_k.shape[1]

            feat_parts = [lf_k, g_k.unsqueeze(2).expand(-1, -1, n_patch)]
            if cond is not None:
                feat_parts.append(cond.unsqueeze(2).expand(-1, -1, n_patch))

            scores = self.heads[k](torch.cat(feat_parts, dim=1)).squeeze(1)
            weights = torch.softmax(scores, dim=1)
            pred = (weights.unsqueeze(-1) * xyz_k).sum(dim=1)

            preds.append(pred)
            scores_list.append(scores)
            l1_xyz_list.append(xyz_k)

        return preds, scores_list, l1_xyz_list


class UnifiedCoarseFineNet(nn.Module):
    def __init__(
        self,
        radii,
        sigma,
        c_in=3,
        c_width=128,
        n_block=4,
        global_dim=256,
        patch_points=PATCH_POINTS_DEFAULT,
        cond_dim=128,
        cond_dropout=0.1,
        use_conditioning=True,
        diffnet_dropout=True,
    ):
        super().__init__()
        self.stage1 = DiffNetCoarseWithGlobalFeature(
            c_in=c_in,
            c_width=c_width,
            n_block=n_block,
            global_dim=global_dim,
            n_landmarks=N_LANDMARKS,
            dropout=diffnet_dropout,
        )
        self.stage2 = JointHeatmapS2Conditioned(
            n_landmarks=N_LANDMARKS,
            dropout=base.DROPOUT_S2,
            stage1_feat_dim=global_dim,
            cond_dim=cond_dim,
            cond_dropout=cond_dropout,
            use_conditioning=use_conditioning,
        )
        self.register_buffer("radii", torch.tensor(radii, dtype=torch.float32))
        self.sigma = float(sigma)
        self.patch_points = int(patch_points)
        self.n_landmarks = N_LANDMARKS
        self.stage1_runtime_device = DEVICE

    def _sample_patch_indices(self, dists, radius):
        inside = torch.nonzero(dists < radius, as_tuple=False).squeeze(1)
        if inside.numel() == 0:
            return torch.topk(dists, k=self.patch_points, largest=False).indices
        if inside.numel() < self.patch_points:
            deficit = self.patch_points - inside.numel()
            repeat = (
                inside[torch.randint(0, inside.numel(), (deficit,), device=dists.device)]
                if self.training
                else inside[:1].repeat(deficit)
            )
            return torch.cat([inside, repeat], dim=0)
        if self.training:
            return inside[torch.randperm(inside.numel(), device=dists.device)[: self.patch_points]]
        return inside[torch.argsort(dists[inside])[: self.patch_points]]

    def _prepare_centers(self, coarse, gt, mix_prob, jitter_std):
        centers = coarse
        if gt is not None and mix_prob > 0 and self.training:
            use_gt = (torch.rand(centers.shape[:2], device=centers.device) < float(mix_prob)).unsqueeze(-1)
            centers = torch.where(use_gt, gt, centers)
        if self.training and float(jitter_std) > 0:
            centers = centers + torch.randn_like(centers) * float(jitter_std)
        return centers

    def _crop_patches(self, x, centers, gt=None):
        pc = x.transpose(1, 2).contiguous()
        patches_by_lm = [[] for _ in range(self.n_landmarks)]
        residual_targets = []

        for b in range(pc.shape[0]):
            residual_b = []
            for k in range(self.n_landmarks):
                center = centers[b, k]
                dists = torch.norm(pc[b] - center.unsqueeze(0), dim=1)
                idx = self._sample_patch_indices(dists, float(self.radii[k].item()))
                patch = (pc[b, idx] - center.unsqueeze(0)).transpose(0, 1).contiguous()
                patches_by_lm[k].append(patch)
                if gt is not None:
                    residual_b.append(gt[b, k] - center)
            if gt is not None:
                residual_targets.append(torch.stack(residual_b, dim=0))

        patches = [torch.stack(patches_by_lm[k], dim=0) for k in range(self.n_landmarks)]
        residuals = torch.stack(residual_targets, dim=0) if gt is not None else None
        return patches, residuals


class DiffNetJointDataset(Dataset):
    def __init__(self, X, Y, indices, diffnet_data):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.diffnet_data = diffnet_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return self.X[idx], self.Y[idx], self.diffnet_data[idx]


def collate_fn(batch):
    pcs, gts, dd = zip(*batch)
    return torch.from_numpy(np.stack(pcs)).float(), torch.from_numpy(np.stack(gts)).float(), list(dd)


def make_loader(X, Y, indices, diffnet_data, batch_size, shuffle):
    return DataLoader(
        DiffNetJointDataset(X, Y, indices, diffnet_data),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )


def run_stage1_batch(stage1, dd_list, stage1_device, output_device):
    coarse_list, feat_list = [], []
    for dd in dd_list:
        coarse, feat = stage1(
            dd["features"].to(stage1_device),
            dd["mass"].to(stage1_device),
            dd["L"].to(stage1_device),
            dd["evals"].to(stage1_device),
            dd["evecs"].to(stage1_device),
            dd["gradX"].to(stage1_device),
            dd["gradY"].to(stage1_device),
        )
        coarse_list.append(coarse.unsqueeze(0).to(output_device) if coarse.dim() == 1 else coarse.to(output_device))
        feat_list.append(feat.unsqueeze(0).to(output_device) if feat.dim() == 1 else feat.to(output_device))
    return torch.cat(coarse_list, dim=0), torch.cat(feat_list, dim=0)


def augment_batch(pc, gt, dd_list, device, flip_prob=0.2):
    pc = pc.to(device)
    gt = gt.to(device)
    batch_size, _, n_points = pc.shape

    theta = torch.rand(batch_size, 1, 1, device=device) * (np.pi / 6) - (np.pi / 12)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    rot = torch.zeros(batch_size, 3, 3, device=device)
    rot[:, 0, 0] = cos_t.flatten()
    rot[:, 0, 1] = -sin_t.flatten()
    rot[:, 1, 0] = sin_t.flatten()
    rot[:, 1, 1] = cos_t.flatten()
    rot[:, 2, 2] = 1.0

    pc_aug = torch.bmm(pc.transpose(1, 2), rot)
    gt_aug = torch.bmm(
        gt.view(batch_size * N_LANDMARKS, 1, 3),
        rot.repeat_interleave(N_LANDMARKS, dim=0),
    ).view(batch_size, N_LANDMARKS, 3)

    scale = torch.rand(batch_size, 1, 1, device=device) * 0.10 + 0.95
    shift = torch.rand(batch_size, 1, 3, device=device) * 0.04 - 0.02

    pc_aug = pc_aug * scale + shift
    gt_aug = gt_aug * scale + shift
    pc_aug = pc_aug + torch.randn(batch_size, n_points, 3, device=device) * 0.005

    flip_idx = (torch.rand(batch_size, device=device) < float(flip_prob)).nonzero(as_tuple=True)[0]
    flip_set = set(int(i) for i in flip_idx.cpu().tolist())
    if len(flip_idx) > 0:
        pc_aug[flip_idx, :, 0] = -pc_aug[flip_idx, :, 0]
        gt_aug[flip_idx, :, 0] = -gt_aug[flip_idx, :, 0]

        tmp = gt_aug[flip_idx, ALARE_R_IDX].clone()
        gt_aug[flip_idx, ALARE_R_IDX] = gt_aug[flip_idx, ALARE_L_IDX]
        gt_aug[flip_idx, ALARE_L_IDX] = tmp

        tmp = gt_aug[flip_idx, ZYGION_R_IDX].clone()
        gt_aug[flip_idx, ZYGION_R_IDX] = gt_aug[flip_idx, ZYGION_L_IDX]
        gt_aug[flip_idx, ZYGION_L_IDX] = tmp

    dd_aug_list = []
    for b, dd in enumerate(dd_list):
        dd_aug = {key: dd[key] for key in ("mass", "L", "evals", "evecs", "gradX", "gradY")}
        features = dd["features"]
        if features.shape[-1] == 3:
            feat = torch.mm(features.to(device), rot[b])
            feat = feat * scale[b, 0, 0] + shift[b, 0]
            if b in flip_set:
                feat[:, 0] = -feat[:, 0]
            dd_aug["features"] = feat.cpu()
        else:
            dd_aug["features"] = features
        dd_aug_list.append(dd_aug)

    return pc_aug.transpose(1, 2), gt_aug, dd_aug_list


def forward_model(model, pc, gt, dd_list, mix_prob=0.0, jitter_std=0.0, detach_coarse=False, detach_feat=False):
    stage1_device = getattr(model, "stage1_runtime_device", pc.device)
    coarse_flat, stage1_feat = run_stage1_batch(model.stage1, dd_list, stage1_device, pc.device)

    coarse = coarse_flat.view(-1, model.n_landmarks, 3)
    crop_centers = coarse.detach() if detach_coarse else coarse
    centers = model._prepare_centers(crop_centers, gt, mix_prob=mix_prob, jitter_std=jitter_std)
    patches, residual_targets = model._crop_patches(pc, centers, gt)

    s2_feat = stage1_feat.detach() if detach_feat else stage1_feat
    preds, scores, l1_xyz = model.stage2(patches, stage1_global_feat=s2_feat)
    pred_residual = torch.stack(preds, dim=1)

    return {
        "coarse": coarse,
        "refined": centers + pred_residual,
        "preds_list": preds,
        "scores_list": scores,
        "l1_xyz_list": l1_xyz,
        "residual_targets": residual_targets,
    }


def compute_losses(out, gt, sigma):
    coarse_loss = F.smooth_l1_loss(out["coarse"], gt)
    refined_loss = F.smooth_l1_loss(out["refined"], gt)
    stage2_loss = base.joint_loss(
        out["preds_list"],
        out["scores_list"],
        out["l1_xyz_list"],
        out["residual_targets"],
        sigma=sigma,
    )
    return coarse_loss, stage2_loss, refined_loss


def lerp(a, b, t):
    return float(a + (b - a) * t)


def train_curriculum(model, train_idx, X, Y, diffnet_data, cfg, val_idx=None, scales=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs_total,
        eta_min=2e-6,
    )
    loader = make_loader(X, Y, train_idx, diffnet_data, cfg.batch_size, shuffle=True)

    base_radii = model.radii.detach().clone()
    best_train, best_train_epoch = float("inf"), 0
    best_val, best_val_epoch = float("inf"), 0

    for epoch in range(1, cfg.epochs_total + 1):
        t = 0.0 if cfg.epochs_total <= 1 else (epoch - 1) / float(cfg.epochs_total - 1)
        mix_prob = lerp(cfg.mix_prob_start, cfg.mix_prob_end, t)
        radius_scale = lerp(cfg.radius_scale_start, cfg.radius_scale_end, t)
        jitter_std = lerp(cfg.jitter_start, cfg.jitter_end, t)
        w_coarse = lerp(cfg.w_coarse_start, cfg.w_coarse, t)
        w_stage2 = lerp(cfg.w_stage2_start, cfg.w_stage2, t)
        w_refined = lerp(cfg.w_refined_start, cfg.w_refined, t)

        with torch.no_grad():
            model.radii.copy_(torch.clamp(base_radii * radius_scale, max=RADIUS_MAX))

        model.train()
        total_acc, coarse_acc, s2_acc, refined_acc = 0.0, 0.0, 0.0, 0.0

        for pc, gt, dd_list in loader:
            pc_aug, gt_aug, dd_aug = augment_batch(
                pc,
                gt,
                dd_list,
                DEVICE,
                flip_prob=cfg.flip_prob,
            )

            optimizer.zero_grad()
            out = forward_model(
                model,
                pc_aug,
                gt_aug,
                dd_aug,
                mix_prob=mix_prob,
                jitter_std=jitter_std,
            )
            coarse_loss, s2_loss, refined_loss = compute_losses(out, gt_aug, cfg.sigma)
            loss = w_coarse * coarse_loss + w_stage2 * s2_loss + w_refined * refined_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()

            total_acc += loss.item()
            coarse_acc += coarse_loss.item()
            s2_acc += s2_loss.item()
            refined_acc += refined_loss.item()

        scheduler.step()

        mean_train = total_acc / max(1, len(loader))
        if mean_train < best_train:
            best_train, best_train_epoch = mean_train, epoch

        val_mm = None
        if val_idx is not None and scales is not None and epoch % max(1, cfg.val_every) == 0:
            val_mm = evaluate_mm(
                model,
                X,
                Y,
                scales,
                val_idx,
                diffnet_data,
                cfg.eval_batch_size,
            )["refined_mean"]
            if val_mm < best_val:
                best_val, best_val_epoch = val_mm, epoch

        if epoch % cfg.print_interval == 0 or epoch == cfg.epochs_total:
            denom = max(1, len(loader))
            val_text = f" val={val_mm:.3f}" if val_mm is not None else ""
            print(
                f"  epoch {epoch:>3}/{cfg.epochs_total} "
                f"loss={mean_train:.5f} "
                f"coarse={coarse_acc / denom:.5f} "
                f"s2={s2_acc / denom:.5f} "
                f"refined={refined_acc / denom:.5f} "
                f"mix={mix_prob:.2f} "
                f"radius={radius_scale:.2f}"
                f"{val_text}"
            )

    with torch.no_grad():
        model.radii.copy_(base_radii)

    return {
        "best_train_epoch": int(best_train_epoch),
        "best_train_loss": float(best_train),
        "best_val_epoch": int(best_val_epoch) if best_val_epoch else None,
        "best_val_mm": float(best_val) if best_val_epoch else None,
    }


@torch.no_grad()
def predict(model, X, indices, diffnet_data, batch_size):
    model.eval()
    coarse_all, refined_all = [], []
    dummy_y = np.zeros((len(X), N_LANDMARKS, 3), dtype=np.float32)
    loader = make_loader(X, dummy_y, indices, diffnet_data, batch_size=batch_size, shuffle=False)

    for pc, _, dd_list in loader:
        pc = pc.to(DEVICE)
        out = forward_model(model, pc, None, dd_list, mix_prob=0.0, jitter_std=0.0)
        coarse_all.append(out["coarse"].cpu().numpy())
        refined_all.append(out["refined"].cpu().numpy())

    return np.concatenate(coarse_all, axis=0), np.concatenate(refined_all, axis=0)


def evaluate_mm(model, X, Y, scales, indices, diffnet_data, batch_size):
    coarse, refined = predict(model, X, indices, diffnet_data, batch_size=batch_size)
    gt = Y[indices]
    scale = scales[indices, None]

    coarse_err = np.linalg.norm(coarse - gt, axis=2) * scale
    refined_err = np.linalg.norm(refined - gt, axis=2) * scale

    return {
        "coarse": coarse_err,
        "refined": refined_err,
        "coarse_mean": float(coarse_err.mean()),
        "refined_mean": float(refined_err.mean()),
    }


def evaluate_with_snap(model, X, Y, scales, indices, diffnet_data, full_clouds, triangles, batch_size):
    coarse, refined = predict(model, X, indices, diffnet_data, batch_size=batch_size)

    gt = Y[indices]
    scale = scales[indices, None]

    coarse_err = np.linalg.norm(coarse - gt, axis=2) * scale
    refined_err = np.linalg.norm(refined - gt, axis=2) * scale

    snap_pt = np.empty_like(refined, dtype=np.float32)
    snap_mesh = np.empty_like(refined, dtype=np.float32)

    for row_i, sample_i in enumerate(indices):
        pc_full = full_clouds[int(sample_i)]
        tri = triangles[int(sample_i)] if triangles is not None else None

        for k in range(N_LANDMARKS):
            snap_pt[row_i, k] = base.snap_to_nearest(refined[row_i, k], pc_full)
            snap_mesh[row_i, k] = base.snap_to_mesh(refined[row_i, k], pc_full, tri)

    snap_pt_err = np.linalg.norm(snap_pt - gt, axis=2) * scale
    snap_mesh_err = np.linalg.norm(snap_mesh - gt, axis=2) * scale

    return coarse_err, refined_err, snap_pt_err, snap_mesh_err


def stats(errs):
    flat = errs.ravel()
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "p90": float(np.percentile(flat, 90)),
    }


def print_eval(title, coarse, refined, snap_pt=None, snap_mesh=None):
    print(f"\n[{title}] N={coarse.shape[0]}")

    header = f"  {'landmark':<12} {'coarse':>8} {'refined':>9}"
    if snap_pt is not None:
        header += f" {'snap-pt':>9}"
    if snap_mesh is not None:
        header += f" {'snap-mesh':>10}"
    print(header)

    for k, name in enumerate(LANDMARK_NAMES):
        line = f"  {name:<12} {coarse[:, k].mean():>8.3f} {refined[:, k].mean():>9.3f}"
        if snap_pt is not None:
            line += f" {snap_pt[:, k].mean():>9.3f}"
        if snap_mesh is not None:
            line += f" {snap_mesh[:, k].mean():>10.3f}"
        print(line)

    print(f"  {'OVERALL':<12} {coarse.mean():>8.3f} {refined.mean():>9.3f}", end="")
    if snap_pt is not None:
        print(f" {snap_pt.mean():>9.3f}", end="")
    if snap_mesh is not None:
        print(f" {snap_mesh.mean():>10.3f}", end="")
    print("  mm")


def build_model(cfg, diffnet_runtime_device):
    radii = [float(RADIUS_MIN_BY_LM.get(name, RADIUS_MIN_DEFAULT)) for name in LANDMARK_NAMES]
    c_in = 3 if cfg.input_features == "xyz" else 16

    model = UnifiedCoarseFineNet(
        radii=radii,
        sigma=cfg.sigma,
        c_in=c_in,
        c_width=cfg.diffnet_c_width,
        n_block=cfg.diffnet_n_block,
        global_dim=cfg.diffnet_global_dim,
        patch_points=cfg.patch_points,
        cond_dim=cfg.cond_dim,
        cond_dropout=cfg.cond_dropout,
        use_conditioning=cfg.use_conditioning,
        diffnet_dropout=DIFFNET_DROPOUT,
    ).to(DEVICE)

    model.stage1_runtime_device = diffnet_runtime_device
    model.stage1.to(diffnet_runtime_device)
    model.stage2.to(DEVICE)

    return model


def train_kfold(cfg):
    set_seed(cfg.seed)
    diffnet_runtime_device = resolve_diffnet_device(cfg.diffnet_device)

    print(f"Stage1 device: {diffnet_runtime_device}")
    print(f"Stage2 device: {DEVICE}")

    X, Y, scales, names, full_clouds, triangles, diffnet_data = load_data_with_diffnet_ops(
        k_eig=cfg.k_eig,
        input_features=cfg.input_features,
    )

    train_idx, test_idx = patient_based_split(names, test_fraction=cfg.test_fraction, seed=cfg.seed)

    train_patients = {patient_id(names[i]) for i in train_idx}
    test_patients = {patient_id(names[i]) for i in test_idx}
    assert train_patients.isdisjoint(test_patients), "DATA LEAKAGE: patient appears in both train and test split."

    n_folds = max(2, min(cfg.kfold_splits, len(train_patients)))
    fold_records = []

    print(f"Samples: total={len(X)}, train={len(train_idx)}, test={len(test_idx)}")
    print(f"Patients: train={len(train_patients)}, test={len(test_patients)}, folds={n_folds}")

    for fold, (tr_local, val_local) in enumerate(
        patient_kfold(np.array(names), train_idx, n_folds, cfg.seed),
        start=1,
    ):
        tr_abs = train_idx[tr_local]
        val_abs = train_idx[val_local]

        print(f"\nFold {fold}/{n_folds}: train={len(tr_abs)}, val={len(val_abs)}")

        model = build_model(cfg, diffnet_runtime_device)
        info = train_curriculum(
            model,
            tr_abs,
            X,
            Y,
            diffnet_data,
            cfg,
            val_idx=val_abs,
            scales=scales,
        )

        val_eval = evaluate_mm(
            model,
            X,
            Y,
            scales,
            val_abs,
            diffnet_data,
            cfg.eval_batch_size,
        )
        print_eval(f"Fold {fold} val", val_eval["coarse"], val_eval["refined"])

        fold_records.append(
            {
                "fold": int(fold),
                "train_scans": int(len(tr_abs)),
                "val_scans": int(len(val_abs)),
                "best_train_epoch": info["best_train_epoch"],
                "best_val_epoch": info["best_val_epoch"],
                "best_val_mm": float(val_eval["refined_mean"]),
                "per_landmark_refined_mm": val_eval["refined"].mean(axis=0).tolist(),
            }
        )

        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    epoch_candidates = [r["best_val_epoch"] for r in fold_records if r["best_val_epoch"]]
    if not epoch_candidates:
        epoch_candidates = [r["best_train_epoch"] for r in fold_records if r["best_train_epoch"]]

    final_epochs = (
        int(np.clip(round(float(np.median(epoch_candidates))), 1, cfg.epochs_total))
        if epoch_candidates
        else cfg.epochs_total
    )

    print(f"\nRetraining on full train split for {final_epochs} epochs")

    final_cfg = SimpleNamespace(**vars(cfg))
    final_cfg.epochs_total = final_epochs

    model = build_model(final_cfg, diffnet_runtime_device)
    final_info = train_curriculum(
        model,
        train_idx,
        X,
        Y,
        diffnet_data,
        final_cfg,
    )

    test_coarse, test_refined, test_snap_pt, test_snap_mesh = evaluate_with_snap(
        model,
        X,
        Y,
        scales,
        test_idx,
        diffnet_data,
        full_clouds,
        triangles,
        cfg.eval_batch_size,
    )

    print_eval("TEST", test_coarse, test_refined, test_snap_pt, test_snap_mesh)

    run_dir = os.path.join(ROOT_DIR, "results", "paper", cfg.exp_id, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "diffnet_patch_retrain_fulltrain_final.pth")
    summary_path = os.path.join(run_dir, "holdout_summary.json")

    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(cfg),
            "final_epochs": int(final_epochs),
            "final_train_info": final_info,
        },
        model_path,
    )

    val_mms = [r["best_val_mm"] for r in fold_records]
    summary = {
        "model_path": model_path,
        "fold_results": fold_records,
        "cv_refined_mean_mm": float(np.mean(val_mms)),
        "cv_refined_std_mm": float(np.std(val_mms)),
        "selected_final_epochs": int(final_epochs),
        "test_coarse": stats(test_coarse),
        "test_refined": stats(test_refined),
        "test_snap_nearest": stats(test_snap_pt),
        "test_snap_mesh": stats(test_snap_mesh),
        "test_snap_mesh_per_landmark_mm": {
            name: float(test_snap_mesh[:, i].mean())
            for i, name in enumerate(LANDMARK_NAMES)
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining complete")
    print(f"CV refined mean: {summary['cv_refined_mean_mm']:.3f} ± {summary['cv_refined_std_mm']:.3f} mm")
    print(f"Test refined: {summary['test_refined']['mean']:.3f} mm")
    print(f"Test snap-mesh: {summary['test_snap_mesh']['mean']:.3f} mm")
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


def get_config():
    return SimpleNamespace(
        seed=42,
        exp_id="H01",
        exp_name="H01_diffnet_patch_patient80_20_mesh16384",
        test_fraction=0.20,
        kfold_splits=5,
        batch_size=4,
        eval_batch_size=8,
        epochs_total=320,
        lr=2e-4,
        sigma=base.SIGMA_DEFAULT,
        patch_points=PATCH_POINTS_DEFAULT,
        flip_prob=0.2,
        grad_clip_norm=1.0,
        val_every=1,
        print_interval=20,
        diffnet_c_width=DIFFNET_C_WIDTH,
        diffnet_n_block=DIFFNET_N_BLOCK,
        diffnet_global_dim=256,
        k_eig=DIFFNET_K_EIG,
        input_features=DIFFNET_INPUT_FEATURES,
        diffnet_device="auto",
        cond_dim=COND_DIM,
        cond_dropout=COND_DROPOUT,
        use_conditioning=True,
        mix_prob_start=0.7,
        mix_prob_end=0.2,
        radius_scale_start=1.2,
        radius_scale_end=1.0,
        jitter_start=0.08,
        jitter_end=base.CENTER_JITTER,
        w_coarse_start=1.0,
        w_coarse=W_COARSE,
        w_stage2_start=0.5,
        w_stage2=W_STAGE2,
        w_refined_start=0.2,
        w_refined=W_REFINED,
    )


if __name__ == "__main__":
    train_kfold(get_config())