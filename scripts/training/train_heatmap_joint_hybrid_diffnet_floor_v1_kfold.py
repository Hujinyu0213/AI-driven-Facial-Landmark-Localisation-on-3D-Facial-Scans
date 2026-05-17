"""
Hybrid DiffusionNet+Patch coarse-to-fine training for 9 facial landmarks.

Combined method:
    - Stage1: DiffusionNet (Sharp et al. 2022) + CoarseHead  (replaces PointTransformer)
    - Stage2: explicit local patch heatmap refiner with conditioning (same as PT+Patch)

Radius policy:
    - Option A (fixed floor radii): landmark-specific radius floors, no adaptive estimation.

Training policy:
    - Single-phase end-to-end + curriculum (one optimizer, always joint backward)
    - Curriculum via mix_prob, loss weights, and radius scale scheduling.

DiffusionNet key properties:
    - Works natively on meshes (vertices + faces) or point clouds
    - Requires precomputed geometric operators (mass, Laplacian, eigenvalues/vectors, gradients)
    - Sampling-agnostic: invariant to different mesh discretizations
    - Use spectral diffusion method for efficiency
"""
import argparse
import copy
import json
import os
import sys
import time
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Add DiffusionNet source to path
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DIFFNET_SRC = os.path.join(ROOT_DIR, "diffusion-net-repo", "src")
if DIFFNET_SRC not in sys.path:
    sys.path.insert(0, DIFFNET_SRC)

import diffusion_net

import train_heatmap_joint_flip_v3 as base

N_LANDMARKS = base.N_LANDMARKS
LANDMARK_NAMES = base.LANDMARK_NAMES
PATCH_POINTS_DEFAULT = base.PATCH_POINTS
device = base.device


def resolve_diffnet_device(mode: str) -> torch.device:
    mode = str(mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--diffnet-device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if os.name == "nt":
        return torch.device("cpu")
    return device

# Global determinism flags
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
EPOCHS_TOTAL = 320
LR_TOTAL = 2e-4

W_COARSE = 0.5
W_STAGE2 = 1.0
W_REFINED = 0.5
COND_DIM = 128
COND_DROPOUT = 0.1

# DiffusionNet hyperparameters
DIFFNET_C_WIDTH = 128        # internal width of DiffusionNet blocks
DIFFNET_N_BLOCK = 4          # number of DiffusionNet blocks
DIFFNET_K_EIG = 64           # number of eigenvalues/vectors for spectral method
                             # (128 can freeze system during eigendecomposition;
                             #  64 is safer and usually sufficient)
DIFFNET_INPUT_FEATURES = 'xyz'  # 'xyz' or 'hks'
DIFFNET_DROPOUT = True

# DiffusionNet operator cache
OP_CACHE_DIR = os.path.join(ROOT_DIR, "data", "diffnet_op_cache")

# Stage2 radius floors (identical to PT+Patch version)
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


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


# ══════════════════════════════════════════════════════════════════════════════
# DiffusionNet operator precomputation
# ══════════════════════════════════════════════════════════════════════════════

def precompute_diffnet_operators(verts_list, faces_list, k_eig=DIFFNET_K_EIG, op_cache_dir=None):
    """
    Precompute DiffusionNet geometric operators for all samples.

    Args:
        verts_list: list of (V_i, 3) torch tensors  (normalized vertex positions)
        faces_list: list of (F_i, 3) torch tensors  (triangle faces, 0-indexed)
        k_eig: number of eigenvalues/eigenvectors
        op_cache_dir: directory for caching operators (will be created if needed)

    Returns:
        List of tuples: (frames, mass, L, evals, evecs, gradX, gradY)
    """
    if op_cache_dir is not None:
        os.makedirs(op_cache_dir, exist_ok=True)

    print(f"[DiffNet] Precomputing operators for {len(verts_list)} samples (k_eig={k_eig}) ...")

    ops_list = []
    from tqdm import tqdm
    for i in tqdm(range(len(verts_list)), desc="DiffNet operators", ascii=True, file=sys.stdout):
        verts = verts_list[i]
        faces = faces_list[i]

        try:
            frames, mass, L, evals, evecs, gradX, gradY = \
                diffusion_net.geometry.get_operators(
                    verts, faces,
                    k_eig=k_eig,
                    op_cache_dir=op_cache_dir,
                )
        except Exception as e:
            print(f"\n[DiffNet] WARNING: Operator computation failed for sample {i} "
                  f"(V={verts.shape[0]}): {e}")
            print(f"[DiffNet] Retrying with k_eig={min(k_eig, 32)} ...")
            frames, mass, L, evals, evecs, gradX, gradY = \
                diffusion_net.geometry.get_operators(
                    verts, faces,
                    k_eig=min(k_eig, 32),
                    op_cache_dir=op_cache_dir,
                )
        ops_list.append((frames, mass, L, evals, evecs, gradX, gradY))

    print(f"[DiffNet] Operators ready for {len(ops_list)} samples.")
    return ops_list


def compute_hks_features(evals, evecs, num_features=16):
    """Compute Heat Kernel Signature features (optional alternative to xyz)."""
    return diffusion_net.geometry.compute_hks_autoscale(evals, evecs, num_features)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (extended with DiffusionNet operators)
# ══════════════════════════════════════════════════════════════════════════════

def load_data_with_diffnet_ops(k_eig=DIFFNET_K_EIG, input_features='xyz', op_cache_dir=None):
    """
    Load data and precompute DiffusionNet operators.

    Returns:
        X          : (N, 3, MAX_POINTS) float32  — mesh-sampled normalized point cloud (for Stage2)
        Y          : (N, 9, 3)   float32  — normalized landmark coordinates
        Scales     : (N,)        float32  — std scale (mm)
        names      : list[str]
        full_clouds: list of (M_i, 3) float32
        triangles  : list of (T_i, 3) int32
        diffnet_data : list of dicts with DiffusionNet inputs per sample:
            {
                'verts':  (V_i, 3) torch tensor,
                'mass':   (V_i,)   torch tensor,
                'L':      (V_i, V_i) sparse tensor,
                'evals':  (K,) torch tensor,
                'evecs':  (V_i, K) torch tensor,
                'gradX':  (V_i, V_i) sparse tensor,
                'gradY':  (V_i, V_i) sparse tensor,
                'features': (V_i, C_in) torch tensor,  # xyz or hks
            }
    """
    # Use base loader for standard data
    X, Y, scales, names, full_clouds, triangles = base.load_data()

    # Prepare verts/faces lists for operator precomputation
    verts_list = []
    faces_list = []
    for i in range(len(names)):
        # full_clouds[i] is already centered and scaled (numpy)
        verts_t = torch.from_numpy(full_clouds[i]).float()
        tri = triangles[i]
        if tri is not None and len(tri) > 0:
            faces_t = torch.from_numpy(tri).long()
        else:
            # Point cloud mode: empty faces tensor
            faces_t = torch.zeros((0, 3), dtype=torch.long)
        verts_list.append(verts_t)
        faces_list.append(faces_t)

    # Precompute operators
    ops_list = precompute_diffnet_operators(
        verts_list, faces_list,
        k_eig=k_eig,
        op_cache_dir=op_cache_dir,
    )

    # Build per-sample DiffusionNet data dictionaries
    diffnet_data = []
    for i in range(len(names)):
        frames, mass, L, evals, evecs, gradX, gradY = ops_list[i]
        verts = verts_list[i]

        if input_features == 'hks':
            features = compute_hks_features(evals, evecs, num_features=16)  # (V, 16)
        else:  # 'xyz'
            features = verts  # (V, 3)

        diffnet_data.append({
            'verts': verts,
            'mass': mass,
            'L': L,
            'evals': evals,
            'evecs': evecs,
            'gradX': gradX,
            'gradY': gradY,
            'features': features,
        })

    return X, Y, scales, names, full_clouds, triangles, diffnet_data


# ══════════════════════════════════════════════════════════════════════════════
# DiffusionNet Stage1 model
# ══════════════════════════════════════════════════════════════════════════════

class DiffNetCoarseWithGlobalFeature(nn.Module):
    """
    Stage1 using DiffusionNet backbone.
    Returns:
      - coarse_flat: (B, L*3)   — coarse landmark predictions
      - global_feat: (B, D)     — global feature for Stage2 conditioning

    DiffusionNet with outputs_at='global_mean' produces a mass-weighted global
    feature vector (B, C_out). We set C_out = global_dim and use a CoarseHead
    to regress landmark coordinates.
    """

    def __init__(self, c_in=3, c_width=DIFFNET_C_WIDTH, n_block=DIFFNET_N_BLOCK,
                 global_dim=256, n_landmarks=9, dropout=True):
        super().__init__()
        self.c_in = c_in
        self.global_dim = global_dim

        # DiffusionNet as feature extractor with global mean pooling
        self.diffnet = diffusion_net.layers.DiffusionNet(
            C_in=c_in,
            C_out=global_dim,
            C_width=c_width,
            N_block=n_block,
            last_activation=None,
            outputs_at='global_mean',
            dropout=dropout,
            with_gradient_features=True,
            with_gradient_rotations=True,
            diffusion_method='spectral',
        )

        # Coarse regression head
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
        """
        Args:
            features: (B, V, C_in) or (V, C_in)   — per-vertex input features
            mass:     (B, V)       or (V,)         — mass vector
            L:        (B, V, V)    or (V, V)       — sparse Laplacian
            evals:    (B, K)       or (K,)         — eigenvalues
            evecs:    (B, V, K)    or (V, K)       — eigenvectors
            gradX:    (B, V, V)    or (V, V)       — sparse gradient
            gradY:    (B, V, V)    or (V, V)       — sparse gradient

        Returns:
            coarse_flat: (B, L*3)
            global_feat: (B, D)
        """
        global_feat = self.diffnet(features, mass, L=L, evals=evals, evecs=evecs,
                                   gradX=gradX, gradY=gradY)  # (B, global_dim)
        coarse = self.coarse_head(global_feat)  # (B, L*3)
        return coarse, global_feat


# ══════════════════════════════════════════════════════════════════════════════
# Stage2: JointHeatmapS2Conditioned  (identical to PT+Patch version)
# ══════════════════════════════════════════════════════════════════════════════

class JointHeatmapS2Conditioned(nn.Module):
    """
    Stage2 heatmap refiner with optional Stage1 global-feature conditioning.
    (Identical to the PT+Patch version — operates on local patches via PointNet++ MSG.)
    """
    def __init__(
        self,
        n_landmarks=9,
        dropout=0.3,
        sa1_radii=None,
        stage1_feat_dim=256,
        cond_dim=128,
        cond_dropout=0.1,
        use_conditioning=True,
    ):
        super().__init__()
        sa1_radii = sa1_radii or base.SA1_RADII_S2
        self.use_conditioning = bool(use_conditioning)
        self.cond_dim = int(cond_dim) if self.use_conditioning else 0

        self.sa1 = base.PointNetSetAbstractionMsg(
            npoint=512,
            radii=sa1_radii,
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

        if self.use_conditioning:
            self.cond_proj = nn.Sequential(
                nn.Linear(stage1_feat_dim, self.cond_dim),
                nn.LayerNorm(self.cond_dim),
                nn.ReLU(),
                nn.Dropout(cond_dropout),
            )
        else:
            self.cond_proj = nn.Identity()

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
                for _ in range(n_landmarks)
            ]
        )
        self.n_landmarks = n_landmarks

    def forward(self, patches_list, stage1_global_feat=None, detach_cond=False):
        if self.use_conditioning and stage1_global_feat is None:
            raise RuntimeError("stage1_global_feat is required when use_conditioning=True.")

        B = patches_list[0].shape[0]
        all_patches = torch.cat(patches_list, dim=0)
        xyz_t = all_patches.transpose(1, 2).contiguous()
        l1_xyz, l1_f = self.sa1(xyz_t, None)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f)
        g = self.global_mlp(l2_f).max(dim=2)[0]

        cond = None
        if self.use_conditioning:
            cond_in = stage1_global_feat.detach() if detach_cond else stage1_global_feat
            cond = self.cond_proj(cond_in)

        preds, scores_list, l1_xyz_list = [], [], []
        for k in range(self.n_landmarks):
            lf_k = l1_f[k * B : (k + 1) * B]
            g_k = g[k * B : (k + 1) * B]
            xyz_k = l1_xyz[k * B : (k + 1) * B]
            patch_points = xyz_k.shape[1]

            g_e = g_k.unsqueeze(2).expand(-1, -1, patch_points)
            if cond is not None:
                c_e = cond.unsqueeze(2).expand(-1, -1, patch_points)
                feat = torch.cat([lf_k, g_e, c_e], dim=1)
            else:
                feat = torch.cat([lf_k, g_e], dim=1)

            scores = self.heads[k](feat).squeeze(1)
            weights = torch.softmax(scores, dim=1)
            pred = (weights.unsqueeze(-1) * xyz_k).sum(dim=1)
            preds.append(pred)
            scores_list.append(scores)
            l1_xyz_list.append(xyz_k)
        return preds, scores_list, l1_xyz_list


# ══════════════════════════════════════════════════════════════════════════════
# Unified Coarse-Fine Network (DiffusionNet Stage1 + Patch Stage2)
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedCoarseFineNet(nn.Module):
    """
    Stage1: DiffNetCoarseWithGlobalFeature (DiffusionNet → global feature → CoarseHead)
    Stage2: JointHeatmapS2Conditioned       (local patch refinement, same as before)
    """
    def __init__(
        self,
        radii,
        sigma,
        c_in=3,
        c_width=DIFFNET_C_WIDTH,
        n_block=DIFFNET_N_BLOCK,
        global_dim=256,
        patch_points=PATCH_POINTS_DEFAULT,
        s2_dropout=base.DROPOUT_S2,
        cond_dim=COND_DIM,
        cond_dropout=COND_DROPOUT,
        use_conditioning=True,
        diffnet_dropout=DIFFNET_DROPOUT,
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
        stage1_feat_dim = int(self.stage1.global_dim)
        self.stage2 = JointHeatmapS2Conditioned(
            n_landmarks=N_LANDMARKS,
            dropout=s2_dropout,
            sa1_radii=base.SA1_RADII_S2,
            stage1_feat_dim=stage1_feat_dim,
            cond_dim=cond_dim,
            cond_dropout=cond_dropout,
            use_conditioning=use_conditioning,
        )
        self.register_buffer("radii", torch.tensor(radii, dtype=torch.float32))
        self.sigma = float(sigma)
        self.patch_points = int(patch_points)
        self.n_landmarks = N_LANDMARKS

    def _sample_patch_indices(self, dists, radius):
        inside = torch.nonzero(dists < radius, as_tuple=False).squeeze(1)
        if inside.numel() == 0:
            return torch.topk(dists, k=self.patch_points, largest=False).indices
        if inside.numel() < self.patch_points:
            deficit = self.patch_points - inside.numel()
            if self.training:
                rep = inside[torch.randint(0, inside.numel(), (deficit,), device=dists.device)]
            else:
                rep = inside[:1].repeat(deficit)
            return torch.cat([inside, rep], dim=0)
        if self.training:
            perm = torch.randperm(inside.numel(), device=dists.device)[:self.patch_points]
            return inside[perm]
        order = torch.argsort(dists[inside])
        return inside[order[:self.patch_points]]

    def _prepare_centers(self, coarse_centers, gt_centers, mix_prob, jitter_std):
        centers = coarse_centers
        if gt_centers is not None and mix_prob > 0 and self.training:
            use_gt = (
                torch.rand(centers.shape[0], centers.shape[1], device=centers.device) < mix_prob
            ).unsqueeze(-1)
            centers = torch.where(use_gt, gt_centers, centers)
        if self.training:
            if np.isscalar(jitter_std):
                jitter_value = float(jitter_std)
                if jitter_value > 0:
                    centers = centers + torch.randn_like(centers) * jitter_value
            else:
                jitter_vec = torch.as_tensor(jitter_std, dtype=centers.dtype, device=centers.device)
                if jitter_vec.ndim == 0:
                    jitter_value = float(jitter_vec.item())
                    if jitter_value > 0:
                        centers = centers + torch.randn_like(centers) * jitter_value
                else:
                    jitter_vec = jitter_vec.view(1, -1, 1)
                    centers = centers + torch.randn_like(centers) * jitter_vec
        return centers

    def _crop_patches(self, x, centers, gt_centers):
        """Crop local patches from point cloud for Stage2.

        x:       (B, 3, N) — the downsampled point cloud (same as before)
        centers: (B, L, 3)
        """
        pc = x.transpose(1, 2).contiguous()  # (B, N, 3)
        batch_size = pc.shape[0]
        patches_per_landmark = [[] for _ in range(self.n_landmarks)]
        residual_targets = []
        for b in range(batch_size):
            res_b = []
            pc_b = pc[b]
            for k in range(self.n_landmarks):
                ctr = centers[b, k]
                dists = torch.norm(pc_b - ctr.unsqueeze(0), dim=1)
                idx = self._sample_patch_indices(dists, float(self.radii[k].item()))
                patch = (pc_b[idx] - ctr.unsqueeze(0)).transpose(0, 1).contiguous()
                patches_per_landmark[k].append(patch)
                if gt_centers is not None:
                    res_b.append(gt_centers[b, k] - ctr)
            if gt_centers is not None:
                residual_targets.append(torch.stack(res_b, dim=0))
        patches_list = [torch.stack(patches_per_landmark[k], dim=0)
                        for k in range(self.n_landmarks)]
        residuals = torch.stack(residual_targets, dim=0) if gt_centers is not None else None
        return patches_list, residuals

    def forward(
        self,
        x,
        diffnet_inputs,
        gt_centers=None,
        mix_prob=0.0,
        jitter_std=0.0,
        detach_coarse_for_crop=False,
        detach_stage1_feat_for_s2=False,
    ):
        """
        Args:
            x:             (B, 3, N) downsampled point cloud for Stage2 patch cropping
            diffnet_inputs: dict with DiffusionNet operator tensors (already on device):
                            'features', 'mass', 'L', 'evals', 'evecs', 'gradX', 'gradY'
            gt_centers:    (B, L, 3) or None
        """
        # Stage1: DiffusionNet
        coarse_flat, stage1_global_feat = self.stage1(
            diffnet_inputs['features'],
            diffnet_inputs['mass'],
            diffnet_inputs['L'],
            diffnet_inputs['evals'],
            diffnet_inputs['evecs'],
            diffnet_inputs['gradX'],
            diffnet_inputs['gradY'],
        )
        coarse = coarse_flat.view(-1, self.n_landmarks, 3)
        coarse_for_crop = coarse.detach() if detach_coarse_for_crop else coarse

        centers = self._prepare_centers(
            coarse_for_crop, gt_centers, mix_prob=mix_prob, jitter_std=jitter_std
        )
        patches_list, residual_targets = self._crop_patches(x, centers, gt_centers)

        preds_list, scores_list, l1_xyz_list = self.stage2(
            patches_list,
            stage1_global_feat=stage1_global_feat,
            detach_cond=detach_stage1_feat_for_s2,
        )
        pred_residual = torch.stack(preds_list, dim=1)
        refined = centers + pred_residual

        return {
            "coarse": coarse,
            "stage1_global_feat": stage1_global_feat,
            "centers_used": centers,
            "preds_list": preds_list,
            "pred_residual": pred_residual,
            "scores_list": scores_list,
            "l1_xyz_list": l1_xyz_list,
            "residual_targets": residual_targets,
            "refined": refined,
        }


# ══════════════════════════════════════════════════════════════════════════════
# DiffusionNet-aware Dataset & DataLoader
# ══════════════════════════════════════════════════════════════════════════════

class DiffNetJointDataset(Dataset):
    """
    Dataset that yields (pc, gt, diffnet_sample_data) tuples.
    DiffusionNet operates on variable-sized meshes, so we use batch_size=1
    for DiffusionNet and handle the batching manually.

    Actually, DiffusionNet CAN handle batches if all samples have the same
    number of vertices, but our meshes have different vertex counts.
    So we process DiffusionNet per-sample and batch the outputs.
    """

    def __init__(self, X, Y, indices, diffnet_data):
        """
        X:            (N_total, 3, MAX_POINTS)
        Y:            (N_total, 9, 3)
        indices:      array of sample indices into X/Y/diffnet_data
        diffnet_data: list of dicts (one per sample in full dataset)
        """
        self.X = X
        self.Y = Y
        self.indices = indices
        self.diffnet_data = diffnet_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        pc = self.X[idx]                # (3, MAX_POINTS) numpy
        gt = self.Y[idx]                # (9, 3) numpy
        dd = self.diffnet_data[idx]     # dict with DiffusionNet operators
        return pc, gt, dd


def diffnet_collate_fn(batch):
    """
    Custom collate: since DiffusionNet operators have variable sizes per sample,
    we return them as lists rather than stacked tensors.
    """
    pcs, gts, dd_list = zip(*batch)
    pc_tensor = torch.from_numpy(np.stack(pcs)).float()     # (B, 3, N)
    gt_tensor = torch.from_numpy(np.stack(gts)).float()     # (B, 9, 3)
    return pc_tensor, gt_tensor, list(dd_list)


def run_stage1_batch(stage1_model, dd_list, stage1_dev, output_dev):
    """
    Run DiffusionNet Stage1 per-sample (since meshes have different vertex counts)
    and stack the results.

    Returns:
        coarse_flat:      (B, L*3)
        stage1_global_feat: (B, D)

    IMPORTANT: DiffusionNet's forward() handles unbatched inputs internally —
    it will unsqueeze the batch dimension itself. We must NOT add a batch
    dimension to sparse tensors (L, gradX, gradY) because that creates
    malformed 3D sparse tensors that can hang the GPU.
    """
    coarse_list = []
    feat_list = []
    for dd in dd_list:
        # Pass WITHOUT unsqueeze — DiffusionNet handles unbatched (2D) inputs
        # by adding the batch dimension internally for both dense and sparse tensors.
        features = dd['features'].to(stage1_dev)    # (V, C)  — no batch dim
        mass = dd['mass'].to(stage1_dev)             # (V,)    — no batch dim
        L = dd['L'].to(stage1_dev)                   # (V, V)  — sparse, no batch dim
        evals = dd['evals'].to(stage1_dev)           # (K,)    — no batch dim
        evecs = dd['evecs'].to(stage1_dev)           # (V, K)  — no batch dim
        gradX = dd['gradX'].to(stage1_dev)           # (V, V)  — sparse, no batch dim
        gradY = dd['gradY'].to(stage1_dev)           # (V, V)  — sparse, no batch dim

        coarse, feat = stage1_model(features, mass, L, evals, evecs, gradX, gradY)
        # DiffusionNet returns unbatched output when input is unbatched,
        # so we need to add batch dim for stacking.
        if coarse.dim() == 1:
            coarse = coarse.unsqueeze(0)  # (L*3,) -> (1, L*3)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)      # (D,) -> (1, D)
        coarse_list.append(coarse.to(output_dev))  # (1, L*3)
        feat_list.append(feat.to(output_dev))      # (1, D)

        # Free GPU memory for variable-size sparse ops between samples
        del features, mass, L, evals, evecs, gradX, gradY

    return torch.cat(coarse_list, dim=0), torch.cat(feat_list, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation for DiffusionNet inputs
# ══════════════════════════════════════════════════════════════════════════════

def augment_diffnet_batch(pc, lbl, dd_list, dev):
    """
    Apply identical augmentation to both the downsampled point cloud (for Stage2)
    AND the DiffusionNet per-vertex features (xyz coordinates).

    Augmentation: rotation, scale, shift, noise, mirror flip (same as base.augment_batch).

    Note: DiffusionNet operators (mass, L, evals, evecs, gradX, gradY) are INTRINSIC
    geometric properties and do NOT change under rigid transformations (rotation, translation,
    uniform scale). Only the xyz features need to be transformed.

    For HKS features, no transformation is needed since HKS is intrinsically defined.
    """
    pc = pc.to(dev)
    lbl = lbl.to(dev)
    B, _, N = pc.shape

    # Random Z-axis rotation (±15°)
    th = torch.rand(B, 1, 1, device=dev) * (2 * np.pi / 12) - np.pi / 12
    c, s = torch.cos(th), torch.sin(th)
    rot = torch.zeros(B, 3, 3, device=dev)
    rot[:, 0, 0] = c.flatten()
    rot[:, 0, 1] = -s.flatten()
    rot[:, 1, 0] = s.flatten()
    rot[:, 1, 1] = c.flatten()
    rot[:, 2, 2] = 1.0

    # Apply rotation to pc
    pc_r = torch.bmm(pc.transpose(1, 2), rot)  # (B, N, 3)

    # Apply rotation to labels
    if lbl.dim() == 3:
        lbl_r = torch.bmm(
            lbl.view(B * N_LANDMARKS, 1, 3),
            rot.repeat_interleave(N_LANDMARKS, dim=0)
        ).view(B, N_LANDMARKS, 3)
    else:
        lbl_r = torch.bmm(lbl.view(B, 1, 3), rot).view(B, 3)

    # Random scale
    sc = torch.rand(B, 1, 1, device=dev) * 0.1 + 0.95
    pc_r = pc_r * sc
    if lbl.dim() == 3:
        lbl_r = lbl_r * sc.squeeze(-1).unsqueeze(-1)
    else:
        lbl_r = lbl_r * sc.squeeze(-1)

    # Random shift
    sh = torch.rand(B, 1, 3, device=dev) * 0.04 - 0.02
    pc_r = pc_r + sh
    if lbl.dim() == 3:
        lbl_r = lbl_r + sh.squeeze(1).unsqueeze(1)
    else:
        lbl_r = lbl_r + sh.squeeze(1)

    # Add noise to pc
    pc_r = pc_r + torch.randn(B, N, 3, device=dev) * 0.005

    # Mirror flip (prob=0.2)
    flip_idx = (torch.rand(B, device=dev) < 0.2).nonzero(as_tuple=True)[0]
    if len(flip_idx) > 0:
        pc_r[flip_idx, :, 0] = -pc_r[flip_idx, :, 0]
        if lbl.dim() == 3:
            lbl_r[flip_idx, :, 0] = -lbl_r[flip_idx, :, 0]
            saved_5 = lbl_r[flip_idx, 5].clone()
            saved_7 = lbl_r[flip_idx, 7].clone()
            lbl_r[flip_idx, 5] = lbl_r[flip_idx, 6]
            lbl_r[flip_idx, 6] = saved_5
            lbl_r[flip_idx, 7] = lbl_r[flip_idx, 8]
            lbl_r[flip_idx, 8] = saved_7

    # Apply SAME augmentation to DiffusionNet xyz features
    augmented_dd_list = []
    for b_idx in range(B):
        dd = dd_list[b_idx]
        dd_aug = {}
        # Copy intrinsic operators (unchanged by rigid transforms)
        for key in ['mass', 'L', 'evals', 'evecs', 'gradX', 'gradY']:
            dd_aug[key] = dd[key]

        features = dd['features']  # (V, C_in) on CPU

        # Check if features are xyz (C_in=3) — apply transform
        if features.shape[-1] == 3:
            feat_dev = features.to(dev)     # (V, 3)
            # Rotation
            feat_r = torch.mm(feat_dev, rot[b_idx])  # (V, 3)
            # Scale
            feat_r = feat_r * sc[b_idx, 0, :]       # broadcast (V, 3) * (3,)
            # Shift
            feat_r = feat_r + sh[b_idx, 0, :]       # (V, 3)
            # Mirror flip
            if b_idx in flip_idx:
                feat_r[:, 0] = -feat_r[:, 0]
            dd_aug['features'] = feat_r.cpu()
        else:
            # HKS or other intrinsic features: no augmentation needed
            dd_aug['features'] = features

        augmented_dd_list.append(dd_aug)

    return pc_r.transpose(1, 2), lbl_r, augmented_dd_list


# ══════════════════════════════════════════════════════════════════════════════
# Loss computation (identical to PT+Patch version)
# ══════════════════════════════════════════════════════════════════════════════

def compute_losses(outputs, gt_centers, sigma):
    coarse_loss = F.smooth_l1_loss(outputs["coarse"], gt_centers)
    refined_loss = F.smooth_l1_loss(outputs["refined"], gt_centers)
    if outputs["residual_targets"] is None:
        raise RuntimeError("residual_targets is None. Pass gt_centers during training.")
    stage2_loss = base.joint_loss(
        outputs["preds_list"],
        outputs["scores_list"],
        outputs["l1_xyz_list"],
        outputs["residual_targets"],
        sigma=sigma,
    )
    return coarse_loss, stage2_loss, refined_loss


def _lerp(a, b, t):
    return float(a + (b - a) * t)


# ══════════════════════════════════════════════════════════════════════════════
# Training functions
# ══════════════════════════════════════════════════════════════════════════════

def make_diffnet_loader(X, Y, indices, diffnet_data, batch_size, shuffle):
    ds = DiffNetJointDataset(X, Y, indices, diffnet_data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=diffnet_collate_fn, num_workers=0)


def forward_model_with_diffnet(model, pc_a, gt_a, dd_list_aug, mix_prob, jitter_std,
                                detach_coarse, detach_feat):
    """
    Run the unified model:
      1) Stage1 per-sample (variable mesh sizes)
      2) Use Stage1 outputs for Stage2 (batched patches)
    """
    B = pc_a.shape[0]
    dev = pc_a.device

    # Run Stage1 per-sample
    stage1_dev = getattr(model, "stage1_runtime_device", dev)
    coarse_flat, stage1_global_feat = run_stage1_batch(
        model.stage1, dd_list_aug, stage1_dev=stage1_dev, output_dev=dev
    )
    coarse = coarse_flat.view(-1, model.n_landmarks, 3)
    coarse_for_crop = coarse.detach() if detach_coarse else coarse

    centers = model._prepare_centers(
        coarse_for_crop, gt_a, mix_prob=mix_prob, jitter_std=jitter_std
    )
    patches_list, residual_targets = model._crop_patches(pc_a, centers, gt_a)

    feat_for_s2 = stage1_global_feat.detach() if detach_feat else stage1_global_feat
    preds_list, scores_list, l1_xyz_list = model.stage2(
        patches_list,
        stage1_global_feat=feat_for_s2,
        detach_cond=False,  # already handled above
    )
    pred_residual = torch.stack(preds_list, dim=1)
    refined = centers + pred_residual

    return {
        "coarse": coarse,
        "stage1_global_feat": stage1_global_feat,
        "centers_used": centers,
        "preds_list": preds_list,
        "pred_residual": pred_residual,
        "scores_list": scores_list,
        "l1_xyz_list": l1_xyz_list,
        "residual_targets": residual_targets,
        "refined": refined,
    }


def train_single_phase_curriculum(
    model,
    train_indices,
    X, Y, diffnet_data,
    epochs,
    lr,
    sigma,
    batch_size,
    mix_prob_start, mix_prob_end,
    radius_scale_start, radius_scale_end,
    w_coarse_start, w_coarse_end,
    w_stage2_start, w_stage2_end,
    w_refined_start, w_refined_end,
    jitter_start, jitter_end,
    Xval=None, Yval=None, Sval=None,
    val_indices=None,
    val_batch_size=8,
    val_every=1,
    log_every_epochs=20,
    log_every_batches=0,
    log_first_batch=False,
):
    print(f"\n[SinglePhase-Curriculum] epochs={epochs}, lr={lr}, sigma={sigma}")
    set_requires_grad(model.stage1, True)
    set_requires_grad(model.stage2, True)

    base_radii = model.radii.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-6)

    best_total = float("inf")
    best_epoch = 0
    best_val_mm = float("inf")
    best_epoch_val = 0

    loader = make_diffnet_loader(X, Y, train_indices, diffnet_data,
                                 batch_size=batch_size, shuffle=True)

    for ep in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        t = 0.0 if epochs <= 1 else (ep - 1) / float(epochs - 1)
        mix_prob = _lerp(mix_prob_start, mix_prob_end, t)
        radius_scale = _lerp(radius_scale_start, radius_scale_end, t)
        w_coarse = _lerp(w_coarse_start, w_coarse_end, t)
        w_stage2 = _lerp(w_stage2_start, w_stage2_end, t)
        w_refined = _lerp(w_refined_start, w_refined_end, t)
        jitter_std = _lerp(jitter_start, jitter_end, t)

        with torch.no_grad():
            cur_r = torch.clamp(base_radii * radius_scale, max=RADIUS_MAX)
            model.radii.copy_(cur_r)

        model.train()
        acc_total, acc_c, acc_s2, acc_r = 0.0, 0.0, 0.0, 0.0

        for batch_idx, (pc, gt, dd_list) in enumerate(loader, start=1):
            debug_this_batch = bool(log_first_batch) and ep == 1 and batch_idx == 1
            if debug_this_batch:
                print("[debug] epoch=1 batch=1 fetched batch from loader", flush=True)

            pc_a, gt_a, dd_list_aug = augment_diffnet_batch(pc, gt, dd_list, device)
            if debug_this_batch:
                print("[debug] epoch=1 batch=1 augmentation done", flush=True)

            opt.zero_grad()
            out = forward_model_with_diffnet(
                model, pc_a, gt_a, dd_list_aug,
                mix_prob=mix_prob,
                jitter_std=jitter_std,
                detach_coarse=False,
                detach_feat=False,
            )
            if debug_this_batch:
                print("[debug] epoch=1 batch=1 forward done", flush=True)

            c_loss, s2_loss, r_loss = compute_losses(out, gt_a, sigma=sigma)
            total = w_coarse * c_loss + w_stage2 * s2_loss + w_refined * r_loss
            total.backward()
            if debug_this_batch:
                print("[debug] epoch=1 batch=1 backward done", flush=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if debug_this_batch:
                print("[debug] epoch=1 batch=1 optimizer step done", flush=True)

            acc_total += total.item()
            acc_c += c_loss.item()
            acc_s2 += s2_loss.item()
            acc_r += r_loss.item()

            if log_every_batches and (batch_idx % int(log_every_batches) == 0):
                print(
                    f"    [batch {batch_idx:>4d}/{len(loader):<4d}] "
                    f"total={total.item():.5f} coarse={c_loss.item():.5f} "
                    f"s2={s2_loss.item():.5f} refined={r_loss.item():.5f}",
                    flush=True,
                )

        sched.step()

        denom = max(1, len(loader))
        tr_total = acc_total / denom
        if tr_total < best_total:
            best_total = tr_total
            best_epoch = ep

        cur_val_mm = None
        if Xval is not None and Yval is not None and Sval is not None and val_indices is not None:
            if ep % max(1, int(val_every)) == 0:
                cur_val_mm = val_softargmax_mm(model, Xval, Yval, Sval, val_indices, diffnet_data,
                                               batch_size=val_batch_size)
                if cur_val_mm < best_val_mm:
                    best_val_mm = cur_val_mm
                    best_epoch_val = ep

        epoch_secs = time.perf_counter() - epoch_t0
        log_epoch_every = max(1, int(log_every_epochs))
        if ep % log_epoch_every == 0 or ep == epochs:
            val_part = f" val={cur_val_mm:.3f}" if cur_val_mm is not None else ""
            print(
                f"  ep{ep:3d} lr={opt.param_groups[0]['lr']:.6f} total={tr_total:.5f} "
                f"coarse={acc_c/denom:.5f} s2={acc_s2/denom:.5f} refined={acc_r/denom:.5f} "
                f"mix={mix_prob:.3f} radius_scale={radius_scale:.3f} jitter={jitter_std:.4f} "
                f"w=({w_coarse:.2f},{w_stage2:.2f},{w_refined:.2f}) time={epoch_secs:.1f}s{val_part}",
                flush=True,
            )
        elif log_every_batches or log_first_batch:
            val_part = f" val={cur_val_mm:.3f}" if cur_val_mm is not None else ""
            print(
                f"  ep{ep:3d} done time={epoch_secs:.1f}s total={tr_total:.5f}{val_part}",
                flush=True,
            )

    with torch.no_grad():
        model.radii.copy_(base_radii)

    return {
        "best_epoch_by_train_total": int(best_epoch),
        "best_train_total": float(best_total),
        "best_epoch_by_val_softargmax": int(best_epoch_val) if best_epoch_val > 0 else None,
        "best_val_softargmax_mm": float(best_val_mm) if best_epoch_val > 0 else None,
        "epochs": int(epochs),
        "lr": float(lr),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Prediction & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_unified(model, X, indices, diffnet_data, batch_size=8):
    model.eval()
    coarse_all, refined_all = [], []
    loader = make_diffnet_loader(X, np.zeros((len(X), N_LANDMARKS, 3), dtype=np.float32),
                                 indices, diffnet_data,
                                 batch_size=batch_size, shuffle=False)
    for pc, _, dd_list in loader:
        pc_dev = pc.to(device)
        out = forward_model_with_diffnet(
            model, pc_dev, None, dd_list,
            mix_prob=0.0, jitter_std=0.0,
            detach_coarse=False, detach_feat=False,
        )
        coarse_all.append(out["coarse"].cpu().numpy())
        refined_all.append(out["refined"].cpu().numpy())
    return np.concatenate(coarse_all, axis=0), np.concatenate(refined_all, axis=0)


@torch.no_grad()
def val_softargmax_mm(model, Xv, Yv, Sv, val_indices, diffnet_data, batch_size=8):
    _, refined = predict_unified(model, Xv, val_indices, diffnet_data, batch_size=batch_size)
    # Yv is already the subset — but we index from X via indices, so Yv should match
    errs = np.linalg.norm(refined - Yv[val_indices], axis=2) * Sv[val_indices, None]
    return float(errs.mean())


def evaluate_unified(model, Xte, Yte, Ste, full_clouds_te, triangles_te,
                     test_indices, diffnet_data, tag):
    coarse_all, refined_all = predict_unified(model, Xte, test_indices, diffnet_data, batch_size=8)

    per_lm_refined = [[] for _ in range(N_LANDMARKS)]
    per_lm_snap_pt = [[] for _ in range(N_LANDMARKS)]
    per_lm_snap_sf = [[] for _ in range(N_LANDMARKS)]
    per_lm_coarse = [[] for _ in range(N_LANDMARKS)]

    n_test = len(test_indices)
    for ii in range(n_test):
        real_idx = test_indices[ii]
        pc_full = full_clouds_te[ii]
        tri = triangles_te[ii] if triangles_te is not None else None
        sc = Ste[real_idx]

        for k in range(N_LANDMARKS):
            gt = Yte[real_idx, k]
            coarse = coarse_all[ii, k]
            refined = refined_all[ii, k]
            snap_pt = base.snap_to_nearest(refined, pc_full)
            snap_sf = base.snap_to_mesh(refined, pc_full, tri)

            per_lm_coarse[k].append(np.linalg.norm(coarse - gt) * sc)
            per_lm_refined[k].append(np.linalg.norm(refined - gt) * sc)
            per_lm_snap_pt[k].append(np.linalg.norm(snap_pt - gt) * sc)
            per_lm_snap_sf[k].append(np.linalg.norm(snap_sf - gt) * sc)

    print(f"\n{'='*70}")
    print(f"  Eval results  [{tag}]  test N={n_test}")
    print(f"{'='*70}")
    print(
        f"  {'landmark':<12} {'coarse':>8} {'soft-argmax':>12} "
        f"{'snap-pt':>9} {'snap-mesh':>10}  (mean mm)"
    )
    print(f"  {'-'*55}")

    all_refined, all_snap_pt, all_snap_sf = [], [], []
    per_lm_results = {}

    for k, lm in enumerate(LANDMARK_NAMES):
        c_m = np.mean(per_lm_coarse[k])
        r_m = np.mean(per_lm_refined[k])
        sp_m = np.mean(per_lm_snap_pt[k])
        ss_m = np.mean(per_lm_snap_sf[k])
        print(f"  {lm:<12} {c_m:>8.3f}  {r_m:>10.3f}  {sp_m:>8.3f}  {ss_m:>9.3f}")

        all_refined += per_lm_refined[k]
        all_snap_pt += per_lm_snap_pt[k]
        all_snap_sf += per_lm_snap_sf[k]

        per_lm_results[lm] = {
            "coarse": base.percentile_stats(per_lm_coarse[k]),
            "soft_argmax": base.percentile_stats(per_lm_refined[k]),
            "snap_nearest": base.percentile_stats(per_lm_snap_pt[k]),
            "snap_mesh": base.percentile_stats(per_lm_snap_sf[k]),
        }

    print(f"  {'-'*55}")
    r_stats = base.percentile_stats(all_refined)
    sp_stats = base.percentile_stats(all_snap_pt)
    ss_stats = base.percentile_stats(all_snap_sf)
    print(
        f"  {'OVERALL':<12} {'':>8}  {r_stats['mean']:>10.3f}  "
        f"{sp_stats['mean']:>8.3f}  {ss_stats['mean']:>9.3f}  mm (mean)"
    )
    print(
        f"  {'median':<12} {'':>8}  {r_stats['median']:>10.3f}  "
        f"{sp_stats['median']:>8.3f}  {ss_stats['median']:>9.3f}  mm"
    )
    print(
        f"  {'P90':<12} {'':>8}  {r_stats['p90']:>10.3f}  "
        f"{sp_stats['p90']:>8.3f}  {ss_stats['p90']:>9.3f}  mm"
    )
    print(f"{'='*70}\n")

    return {
        "per_landmark": per_lm_results,
        "overall": {
            "soft_argmax": r_stats,
            "snap_nearest": sp_stats,
            "snap_mesh": ss_stats,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment loop (mirrors PT+Patch structure exactly)
# ══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed, args, X, Y, scales, names, full_clouds, triangles, diffnet_data, diffnet_runtime_device):
    tag = f"seed{seed}_sigma{args.sigma}"
    print(f"\n{'#'*70}")
    print(f"  DiffusionNet Hybrid  |  SEED={seed}  SIGMA={args.sigma}")
    print(f"{'#'*70}\n")

    rng = np.random.RandomState(seed)
    n_total = len(X)
    n_test = max(1, int(n_total * 0.20))
    test_idx = rng.choice(n_total, n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)

    Xtr, Ytr, Str = X[train_idx], Y[train_idx], scales[train_idx]
    Xte, Yte, Ste = X[test_idx], Y[test_idx], scales[test_idx]
    full_clouds_te = [full_clouds[i] for i in test_idx]
    triangles_te = [triangles[i] for i in test_idx] if triangles else None
    print(f"  holdout split: train={len(train_idx)} test={len(test_idx)} (80/20)")

    n_inner = int(max(2, min(args.kfold_splits, len(train_idx))))
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=args.kfold_random_state)
    inner_folds = list(kf.split(train_idx))
    if args.train_fold is not None:
        fold_id = int(args.train_fold)
        if fold_id < 1 or fold_id > len(inner_folds):
            print(f"[skip] invalid --train-fold={fold_id}, valid range: 1..{len(inner_folds)}")
            return None
        inner_folds = [inner_folds[fold_id - 1]]
        fold_offset = fold_id - 1
    else:
        fold_offset = 0

    inner_cv_records = []

    # ── Skip inner-CV if flag set and summary JSON already exists ──
    if args.skip_inner_cv:
        inner_summary_path_check = os.path.join(base.RESULTS_DIR,
                                                 f"joint_hybrid_diffnet_floor_innercv_summary_seed{seed}.json")
        if os.path.exists(inner_summary_path_check):
            print(f"[skip-inner-cv] loading existing summary: {inner_summary_path_check}")
            with open(inner_summary_path_check, "r", encoding="utf-8") as f:
                _existing = json.load(f)
            inner_cv_records = _existing["inner_folds"]
            print(f"  [skip-inner-cv] loaded {len(inner_cv_records)} inner fold records from JSON")
            inner_folds = []  # clear so the loop below is skipped
        else:
            print(f"[skip-inner-cv] WARNING: summary JSON not found at {inner_summary_path_check}, running inner-CV normally")

    for i, (tr_i, val_i) in enumerate(inner_folds, start=1):
        inner_no = i + fold_offset
        inner_tag = f"{tag}_inner{inner_no}"
        # Convert relative indices (within train_idx) to absolute indices
        tr_abs = train_idx[tr_i]
        val_abs = train_idx[val_i]
        print(f"\n  [Inner {inner_no}/{n_inner}] train={len(tr_abs)} val={len(val_abs)}")

        model_path = os.path.join(base.MODELS_DIR, f"joint_hybrid_diffnet_floor_{inner_tag}.pth")
        init_radii = [float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)) for lm in LANDMARK_NAMES]

        c_in = 3 if args.input_features == 'xyz' else 16
        model = UnifiedCoarseFineNet(
            radii=init_radii,
            sigma=args.sigma,
            c_in=c_in,
            c_width=args.diffnet_c_width,
            n_block=args.diffnet_n_block,
            global_dim=args.diffnet_global_dim,
            patch_points=args.patch_points,
            s2_dropout=base.DROPOUT_S2,
            cond_dim=args.cond_dim,
            cond_dropout=args.cond_dropout,
            use_conditioning=(not args.disable_conditioning),
        ).to(device)
        model.stage1_runtime_device = diffnet_runtime_device
        model.stage1.to(diffnet_runtime_device)
        model.stage2.to(device)

        train_curriculum_info = None
        radii_s2 = [float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)) for lm in LANDMARK_NAMES]
        jitter_s2 = [float(args.jitter_end)] * N_LANDMARKS

        if args.eval_only:
            if not os.path.exists(model_path):
                print(f"[skip] model not found: {model_path}")
                continue
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            train_curriculum_info = ckpt.get("train_curriculum", None)
        else:
            print("\n[Stage2 patch params] fixed floor radii (Option A)")
            print(f"  {'landmark':<12} {'jitter':>9} {'radius':>9}")
            print("  " + "-" * 36)
            for k, lm in enumerate(LANDMARK_NAMES):
                print(f"  {lm:<12} {jitter_s2[k]:>9.4f} {radii_s2[k]:>9.4f}")
            with torch.no_grad():
                model.radii.copy_(torch.tensor(radii_s2, dtype=model.radii.dtype,
                                               device=model.radii.device))

            train_curriculum_info = train_single_phase_curriculum(
                model,
                tr_abs,
                X, Y, diffnet_data,
                epochs=args.epochs_total,
                lr=args.lr,
                sigma=args.sigma,
                batch_size=args.batch_size,
                mix_prob_start=args.mix_prob_start,
                mix_prob_end=args.mix_prob_end,
                radius_scale_start=args.radius_scale_start,
                radius_scale_end=args.radius_scale_end,
                w_coarse_start=args.w_coarse_start,
                w_coarse_end=args.w_coarse,
                w_stage2_start=args.w_stage2_start,
                w_stage2_end=args.w_stage2,
                w_refined_start=args.w_refined_start,
                w_refined_end=args.w_refined,
                jitter_start=args.jitter_start,
                jitter_end=args.jitter_end,
                Xval=X,
                Yval=Y,
                Sval=scales,
                val_indices=val_abs,
                val_batch_size=8,
                val_every=args.val_every,
                log_every_epochs=args.log_every_epochs,
                log_every_batches=args.log_every_batches,
                log_first_batch=args.log_first_batch,
            )

            os.makedirs(base.MODELS_DIR, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "seed": seed,
                    "sigma": args.sigma,
                    "inner_fold": int(inner_no),
                    "train_curriculum": train_curriculum_info,
                    "stage2_patch_params": {
                        "radii": [float(x) for x in radii_s2],
                        "jitter": [float(x) for x in jitter_s2],
                        "radius_floor": {
                            lm: float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT))
                            for lm in LANDMARK_NAMES
                        },
                    },
                    "diffnet_config": {
                        "c_width": args.diffnet_c_width,
                        "n_block": args.diffnet_n_block,
                        "global_dim": args.diffnet_global_dim,
                        "k_eig": args.k_eig,
                        "input_features": args.input_features,
                    },
                },
                model_path,
            )
            print(f"[save] Unified model -> {model_path}")

        val_mm = val_softargmax_mm(model, X, Y, scales, val_abs, diffnet_data, batch_size=8)
        print(f"  [Inner {inner_no}] val_softargmax_mean={val_mm:.3f} mm")
        record = {
            "inner_fold": int(inner_no),
            "val_softargmax_mean_mm": float(val_mm),
            "model_path": model_path,
            "train_curriculum": train_curriculum_info,
        }
        inner_cv_records.append(record)

    if not inner_cv_records:
        return None

    val_vals = [r["val_softargmax_mean_mm"] for r in inner_cv_records]
    train_epoch_candidates = [
        int(r["train_curriculum"]["best_epoch_by_val_softargmax"])
        for r in inner_cv_records
        if r.get("train_curriculum") is not None
        and r["train_curriculum"].get("best_epoch_by_val_softargmax") is not None
    ]
    if not train_epoch_candidates:
        train_epoch_candidates = [
            int(r["train_curriculum"]["best_epoch_by_train_total"])
            for r in inner_cv_records
            if r.get("train_curriculum") is not None
            and "best_epoch_by_train_total" in r["train_curriculum"]
        ]
    selected_epochs = int(np.median(train_epoch_candidates)) if train_epoch_candidates else int(args.epochs_total)
    print("\n" + "=" * 68)
    print(f"  Inner-CV selection summary (seed={seed})")
    print("=" * 68)
    print(f"  val_softargmax_mean: mean={np.mean(val_vals):.3f} std={np.std(val_vals):.3f} mm")
    print(f"  selected final epochs (median over inner folds) = {selected_epochs}")

    inner_cv_summary = {
        "seed": int(seed),
        "sigma": float(args.sigma),
        "split": "80_20_holdout",
        "train_kfolds": int(len(inner_cv_records)),
        "inner_cv": {
            "val_softargmax_mean_mm": {
                "mean": float(np.mean(val_vals)),
                "std": float(np.std(val_vals)),
            },
            "selected_epochs": int(selected_epochs),
        },
        "inner_folds": inner_cv_records,
    }

    inner_summary_path = os.path.join(base.RESULTS_DIR,
                                       f"joint_hybrid_diffnet_floor_innercv_summary_seed{seed}.json")
    with open(inner_summary_path, "w", encoding="utf-8") as f:
        json.dump(inner_cv_summary, f, indent=2, ensure_ascii=False)
    print(f"[save] inner-cv summary -> {inner_summary_path}")

    # ── Final retrain on full train set ──
    final_tag = tag
    final_model_path = os.path.join(base.MODELS_DIR, f"joint_hybrid_diffnet_floor_{final_tag}.pth")
    init_radii = [float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)) for lm in LANDMARK_NAMES]

    c_in = 3 if args.input_features == 'xyz' else 16
    final_model = UnifiedCoarseFineNet(
        radii=init_radii,
        sigma=args.sigma,
        c_in=c_in,
        c_width=args.diffnet_c_width,
        n_block=args.diffnet_n_block,
        global_dim=args.diffnet_global_dim,
        patch_points=args.patch_points,
        s2_dropout=base.DROPOUT_S2,
        cond_dim=args.cond_dim,
        cond_dropout=args.cond_dropout,
        use_conditioning=(not args.disable_conditioning),
    ).to(device)
    final_model.stage1_runtime_device = diffnet_runtime_device
    final_model.stage1.to(diffnet_runtime_device)
    final_model.stage2.to(device)

    final_curriculum_info = None
    radii_s2 = [float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)) for lm in LANDMARK_NAMES]
    jitter_s2 = [float(args.jitter_end)] * N_LANDMARKS

    if args.eval_only:
        if not os.path.exists(final_model_path):
            print(f"[skip] final model not found: {final_model_path}")
            return None
        ckpt = torch.load(final_model_path, map_location=device, weights_only=False)
        final_model.load_state_dict(ckpt["model"])
        final_curriculum_info = ckpt.get("train_curriculum", None)
    else:
        with torch.no_grad():
            final_model.radii.copy_(torch.tensor(radii_s2, dtype=final_model.radii.dtype,
                                                  device=final_model.radii.device))

        final_curriculum_info = train_single_phase_curriculum(
            final_model,
            train_idx,
            X, Y, diffnet_data,
            epochs=selected_epochs,
            lr=args.lr,
            sigma=args.sigma,
            batch_size=args.batch_size,
            mix_prob_start=args.mix_prob_start,
            mix_prob_end=args.mix_prob_end,
            radius_scale_start=args.radius_scale_start,
            radius_scale_end=args.radius_scale_end,
            w_coarse_start=args.w_coarse_start,
            w_coarse_end=args.w_coarse,
            w_stage2_start=args.w_stage2_start,
            w_stage2_end=args.w_stage2,
            w_refined_start=args.w_refined_start,
            w_refined_end=args.w_refined,
            jitter_start=args.jitter_start,
            jitter_end=args.jitter_end,
            val_every=args.val_every,
            log_every_epochs=args.log_every_epochs,
            log_every_batches=args.log_every_batches,
            log_first_batch=args.log_first_batch,
        )

        os.makedirs(base.MODELS_DIR, exist_ok=True)
        torch.save(
            {
                "model": final_model.state_dict(),
                "config": vars(args),
                "seed": seed,
                "sigma": args.sigma,
                "train_curriculum": final_curriculum_info,
                "selected_epochs_from_inner_cv": int(selected_epochs),
                "stage2_patch_params": {
                    "radii": [float(x) for x in radii_s2],
                    "jitter": [float(x) for x in jitter_s2],
                    "radius_floor": {
                        lm: float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT))
                        for lm in LANDMARK_NAMES
                    },
                },
                "diffnet_config": {
                    "c_width": args.diffnet_c_width,
                    "n_block": args.diffnet_n_block,
                    "global_dim": args.diffnet_global_dim,
                    "k_eig": args.k_eig,
                    "input_features": args.input_features,
                },
            },
            final_model_path,
        )
        print(f"[save] final model -> {final_model_path}")

    final_results = evaluate_unified(
        model=final_model,
        Xte=X,
        Yte=Y,
        Ste=scales,
        full_clouds_te=full_clouds_te,
        triangles_te=triangles_te,
        test_indices=test_idx,
        diffnet_data=diffnet_data,
        tag=final_tag,
    )
    final_results["seed"] = seed
    final_results["sigma"] = args.sigma
    final_results["selected_epochs_from_inner_cv"] = int(selected_epochs)
    final_results["inner_cv_summary_path"] = inner_summary_path
    if final_curriculum_info is not None:
        final_results["train_curriculum"] = final_curriculum_info

    out_path = os.path.join(base.RESULTS_DIR, f"joint_hybrid_diffnet_floor_eval_{final_tag}.json")
    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"[save] final eval results -> {out_path}")

    seed_summary = {
        "seed": int(seed),
        "sigma": float(args.sigma),
        "split": "80_20_holdout",
        "train_kfolds": int(len(inner_cv_records)),
        "selected_epochs": int(selected_epochs),
        "inner_cv_summary_path": inner_summary_path,
        "final_model_path": final_model_path,
        "final_eval_path": out_path,
        "overall": final_results.get("overall", {}),
        "final_overall": final_results.get("overall", {}),
    }
    summary_path = os.path.join(base.RESULTS_DIR,
                                 f"joint_hybrid_diffnet_floor_holdout_summary_seed{seed}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(seed_summary, f, indent=2, ensure_ascii=False)
    print(f"[save] seed summary -> {summary_path}")
    return seed_summary


def main():
    ap = argparse.ArgumentParser(description="DiffusionNet + Patch Hybrid Landmark Detection")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="single seed; default runs all")
    ap.add_argument("--sigma", type=float, default=base.SIGMA_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--patch-points", type=int, default=PATCH_POINTS_DEFAULT)

    # DiffusionNet-specific
    ap.add_argument("--diffnet-c-width", type=int, default=DIFFNET_C_WIDTH,
                    help="DiffusionNet internal channel width (default: 128)")
    ap.add_argument("--diffnet-n-block", type=int, default=DIFFNET_N_BLOCK,
                    help="Number of DiffusionNet blocks (default: 4)")
    ap.add_argument("--diffnet-global-dim", type=int, default=256,
                    help="DiffusionNet global output feature dimension (default: 256)")
    ap.add_argument("--k-eig", type=int, default=DIFFNET_K_EIG,
                    help="Number of eigenvalues for spectral diffusion (default: 64; "
                         "higher values like 128 may freeze system during eigendecomposition)")
    ap.add_argument("--input-features", type=str, default=DIFFNET_INPUT_FEATURES,
                    choices=['xyz', 'hks'],
                    help="Input features for DiffusionNet: xyz or hks (default: xyz)")
    ap.add_argument("--diffnet-device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Runtime device for DiffusionNet Stage1. On Windows, auto prefers CPU to avoid sparse CUDA stalls.")

    # Training schedule
    ap.add_argument("--kfold-splits", type=int, default=5)
    ap.add_argument("--kfold-random-state", type=int, default=42)
    ap.add_argument("--epochs-total", type=int, default=EPOCHS_TOTAL)
    ap.add_argument("--lr", type=float, default=LR_TOTAL)

    ap.add_argument("--mix-prob-start", type=float, default=0.7)
    ap.add_argument("--mix-prob-end", type=float, default=0.2)
    ap.add_argument("--radius-scale-start", type=float, default=1.2)
    ap.add_argument("--radius-scale-end", type=float, default=1.0)
    ap.add_argument("--jitter-start", type=float, default=0.08)
    ap.add_argument("--jitter-end", type=float, default=base.CENTER_JITTER)
    ap.add_argument("--val-every", type=int, default=1,
                    help="Run validation every N epochs (default: 1)")
    ap.add_argument("--log-every-epochs", type=int, default=20,
                    help="Print one training summary every N epochs (default: 20)")
    ap.add_argument("--log-every-batches", type=int, default=0,
                    help="If >0, print one training log every N batches")
    ap.add_argument("--log-first-batch", action="store_true",
                    help="Print detailed progress for the very first batch to diagnose stalls")

    ap.add_argument("--w-coarse", type=float, default=W_COARSE)
    ap.add_argument("--w-stage2", type=float, default=W_STAGE2)
    ap.add_argument("--w-refined", type=float, default=W_REFINED)
    ap.add_argument("--w-coarse-start", type=float, default=1.0)
    ap.add_argument("--w-stage2-start", type=float, default=0.5)
    ap.add_argument("--w-refined-start", type=float, default=0.2)
    ap.add_argument("--cond-dim", type=int, default=COND_DIM)
    ap.add_argument("--cond-dropout", type=float, default=COND_DROPOUT)
    ap.add_argument("--disable-conditioning", action="store_true")
    ap.add_argument("--train-fold", type=int, default=None,
                    help="run only one inner fold (1-based)")
    ap.add_argument("--skip-inner-cv", action="store_true",
                    help="skip inner-CV training and load existing summary JSON for final retrain")

    args = ap.parse_args()

    diffnet_runtime_device = resolve_diffnet_device(args.diffnet_device)

    print(f"[Config] Stage1 backbone = DiffusionNet (Sharp et al. 2022)")
    print(f"[Config] DiffusionNet: C_width={args.diffnet_c_width}, N_block={args.diffnet_n_block}, "
          f"global_dim={args.diffnet_global_dim}, k_eig={args.k_eig}, input={args.input_features}")
    print(f"[Config] runtime devices: stage1={diffnet_runtime_device}, stage2={device}")
    print(
        f"[Config] conditioning = {not args.disable_conditioning} "
        f"(cond_dim={args.cond_dim}, cond_dropout={args.cond_dropout})"
    )
    print(f"[Config] patch_points = {args.patch_points}")
    print(
        f"[Config] single-phase curriculum: epochs={args.epochs_total} lr={args.lr} "
        f"mix={args.mix_prob_start}->{args.mix_prob_end} "
        f"radius_scale={args.radius_scale_start}->{args.radius_scale_end}"
    )

    # Load data with DiffusionNet operators
    X, Y, scales, names, full_clouds, triangles, diffnet_data = \
        load_data_with_diffnet_ops(
            k_eig=args.k_eig,
            input_features=args.input_features,
            op_cache_dir=OP_CACHE_DIR,
        )

    seed_list = [args.seed] if args.seed is not None else base.SEEDS

    all_results = []
    for seed in seed_list:
        res = run_one_seed(
            seed, args, X, Y, scales, names, full_clouds, triangles, diffnet_data,
            diffnet_runtime_device,
        )
        if res:
            all_results.append(res)

    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print(f"  Multi-seed summary  ({len(all_results)} seeds)")
        print("=" * 70)
        metrics = ["soft_argmax", "snap_nearest", "snap_mesh"]
        for m in metrics:
            vals = [r["overall"][m]["mean"] for r in all_results]
            print(f"  {m:<15}  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  mm")


if __name__ == "__main__":
    main()
