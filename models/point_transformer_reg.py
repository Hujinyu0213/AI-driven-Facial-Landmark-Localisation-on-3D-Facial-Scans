"""
Point Transformer Stage 1 backbone for 9-landmark regression.

Architecture (3-stage, designed for 16384-point input):
  Input  (B, 3, 16384)
  Embed  XYZ -> 32-dim features
  FPS    16384 -> 2048   (initial downsample, no PT here to save memory)
  TD0    linear 32->64
  PT×2   @ 2048 pts, dim=64,  k=16
  TD1    TransitionDown 2048->512, dim=128, k=16
  PT×2   @ 512  pts, dim=128, k=16
  TD2    TransitionDown 512->128,  dim=256, k=16
  PT×2   @ 128  pts, dim=256, k=16
  Global max+mean → 512-dim
  FC     512 → 256 → output_dim (default 27)

Reference:
  Zhao et al. "Point Transformer", ICCV 2021.
  Subtraction-based self-attention with relative position encoding.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def fps_batch(xyz, n_pts):
    """Farthest Point Sampling.
    xyz : (B, N, 3)
    returns idx : (B, n_pts)  long
    """
    try:
        from pointnet2_ops.pointnet2_utils import furthest_point_sample
        return furthest_point_sample(xyz.contiguous(), n_pts).long()
    except Exception:
        # Pure-PyTorch fallback (slow, CPU-only, for debugging)
        B, N, _ = xyz.shape
        device = xyz.device
        idx = torch.zeros(B, n_pts, dtype=torch.long, device=device)
        dist = torch.full((B, N), 1e10, dtype=xyz.dtype, device=device)
        cur = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(n_pts):
            idx[:, i] = cur
            cur_xyz = xyz[torch.arange(B, device=device), cur].unsqueeze(1)  # (B,1,3)
            d = ((xyz - cur_xyz) ** 2).sum(-1)
            dist = torch.min(dist, d)
            cur = dist.argmax(dim=1)
        return idx


def gather_pts(x, idx):
    """Gather indexed rows.
    x   : (B, N, C)
    idx : (B, M)
    returns (B, M, C)
    """
    B, N, C = x.shape
    M = idx.shape[1]
    idx_e = idx.unsqueeze(-1).expand(B, M, C)
    return x.gather(1, idx_e)


def gather_knn(x, idx):
    """Gather k-nearest-neighbor features.
    x   : (B, N, C)
    idx : (B, N, k)
    returns (B, N, k, C)
    """
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_e = idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, C)
    return x.gather(1, idx_e).reshape(B, N, k, C)


def gather_knn_cross(x, idx):
    """Gather neighbors from source to query.
    x   : (B, N_src, C)
    idx : (B, N_qry, k)  -- indices into N_src
    returns (B, N_qry, k, C)
    """
    B, N_src, C = x.shape
    N_qry, k = idx.shape[1], idx.shape[2]
    idx_e = idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, C)
    return x.gather(1, idx_e).reshape(B, N_qry, k, C)


def knn_self(xyz, k):
    """k-nearest neighbors within the same point set (excluding self).
    xyz : (B, N, 3)
    returns idx : (B, N, k)
    """
    dist = torch.cdist(xyz, xyz)        # (B, N, N)
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[1])] = 1e10  # mask self
    return dist.topk(k, dim=-1, largest=False).indices  # (B, N, k)


# ─────────────────────────────────────────────────────────────────────────────
# Point Transformer Layer
# ─────────────────────────────────────────────────────────────────────────────

class PointTransformerLayer(nn.Module):
    """
    Subtraction-based self-attention layer (Zhao et al. 2021).

    y_i = sum_j softmax( MLP(q_i - k_j + pos_ij) ) * (v_j + pos_ij)
    where pos_ij = pos_enc(x_i - x_j)
    """
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.q_lin = nn.Linear(dim, dim, bias=False)
        self.k_lin = nn.Linear(dim, dim, bias=False)
        self.v_lin = nn.Linear(dim, dim, bias=False)
        # Relative position encoding: 3D offset -> dim
        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
        # Attention weight MLP (per-channel attention)
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, xyz, feat):
        """
        xyz  : (B, N, 3)
        feat : (B, N, dim)
        returns: (B, N, dim)
        """
        B, N, C = feat.shape
        k = self.k

        idx = knn_self(xyz, k)                    # (B, N, k)

        q = self.q_lin(feat)                      # (B, N, C)
        kk = self.k_lin(feat)                     # (B, N, C)
        v = self.v_lin(feat)                      # (B, N, C)

        kk_nb = gather_knn(kk, idx)               # (B, N, k, C)
        v_nb  = gather_knn(v,  idx)               # (B, N, k, C)
        xyz_nb = gather_knn(xyz, idx)             # (B, N, k, 3)

        # Relative position encoding
        rel_pos = xyz.unsqueeze(2) - xyz_nb       # (B, N, k, 3)
        rel_pos_flat = rel_pos.reshape(-1, 3)
        pos_e = self.pos_enc(rel_pos_flat).reshape(B, N, k, C)  # (B, N, k, C)

        # Subtraction attention
        attn = self.attn_mlp(q.unsqueeze(2) - kk_nb + pos_e)  # (B, N, k, C)
        attn = torch.softmax(attn, dim=2)                       # normalize over k

        # Weighted aggregation
        out = (attn * (v_nb + pos_e)).sum(dim=2)               # (B, N, C)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Transition Down (downsampling block)
# ─────────────────────────────────────────────────────────────────────────────

class TransitionDown(nn.Module):
    """
    FPS + kNN max-pool aggregation + MLP channel expansion.
    xyz_old  (B, N, 3) , feat_old (B, N, in_dim)
    -> xyz_new (B, M, 3), feat_new (B, M, out_dim)
    """
    def __init__(self, in_dim, out_dim, n_pts, k=16):
        super().__init__()
        self.n_pts = n_pts
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim))

    def forward(self, xyz_old, feat_old):
        """returns xyz_new, feat_new"""
        B = xyz_old.shape[0]

        # Farthest point sampling
        fps_idx = fps_batch(xyz_old, self.n_pts)          # (B, M)
        xyz_new = gather_pts(xyz_old, fps_idx)            # (B, M, 3)

        # kNN: for each centroid find k neighbors in old point set
        dist = torch.cdist(xyz_new, xyz_old)              # (B, M, N)
        knn_idx = dist.topk(self.k, dim=-1,
                            largest=False).indices        # (B, M, k)

        # Max-pool aggregated features
        feat_nb = gather_knn_cross(feat_old, knn_idx)    # (B, M, k, in_dim)
        feat_agg = feat_nb.max(dim=2)[0]                 # (B, M, in_dim)

        feat_new = self.mlp(feat_agg)                    # (B, M, out_dim)
        return xyz_new, feat_new


# ─────────────────────────────────────────────────────────────────────────────
# Full Point Transformer Regression Model
# ─────────────────────────────────────────────────────────────────────────────

class PointTransformerReg(nn.Module):
    """
    Drop-in replacement for PointNet2RegMSG.
    Input  : (B, 3, N)  or (B, 6, N) with normals (normals are ignored)
    Output : (B, output_dim)

    Improvements over v1:
    - Stage 0 uses two-step pre-FPS (16384->4096) + TransitionDown (4096->m0)
      so every centroid aggregates k neighbors from the dense cloud, not just
      a single point's embed features.
    - FC head width scales with d2 for larger dims configurations.
    - Default dims (64,128,256); use (128,256,512) for higher capacity.
    """
    def __init__(self, output_dim=27, dropout=0.4,
                 dims=(64, 128, 256), n_pts=(2048, 512, 128), k=16,
                 pre_fps_n=4096,
                 normal_channel=False):
        super().__init__()
        self.normal_channel = normal_channel
        d0, d1, d2 = dims
        m0, m1, m2 = n_pts
        self.pre_fps_n = pre_fps_n  # intermediate FPS before TD0

        # Initial XYZ embedding (applied to ALL input points)
        d_init = max(32, d0 // 2)
        self.embed = nn.Sequential(
            nn.Linear(3, d_init), nn.ReLU(inplace=True),
            nn.Linear(d_init, d_init))

        # Stage 0: two-step downsample
        #   step A: FPS 16384 → pre_fps_n  (pure spatial, no aggregation)
        #   step B: TD0 pre_fps_n → m0   with proper kNN aggregation (d_init→d0)
        #   Distance matrix at step B: (B, m0, pre_fps_n) = manageable
        self.td0 = TransitionDown(d_init, d0, n_pts=m0, k=k)

        # PT blocks stage 0  (m0 pts, d0 dim)
        self.pt0 = nn.ModuleList([PointTransformerLayer(d0, k=k) for _ in range(2)])
        self.norm0 = nn.ModuleList([nn.LayerNorm(d0) for _ in range(2)])

        # Transition down 0→1
        self.td1 = TransitionDown(d0, d1, n_pts=m1, k=k)

        # PT blocks stage 1
        self.pt1 = nn.ModuleList([PointTransformerLayer(d1, k=k) for _ in range(2)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d1) for _ in range(2)])

        # Transition down 1→2
        self.td2 = TransitionDown(d1, d2, n_pts=m2, k=k)

        # PT blocks stage 2
        self.pt2 = nn.ModuleList([PointTransformerLayer(d2, k=k) for _ in range(2)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d2) for _ in range(2)])

        # Global pooling + FC head (width scales with d2)
        g_dim  = d2 * 2        # max + mean concat
        fc_mid = max(256, d2)  # at least 256, or d2 for large models
        self.fc1  = nn.Linear(g_dim, fc_mid)
        self.bn1  = nn.BatchNorm1d(fc_mid)
        self.drop1 = nn.Dropout(dropout)
        self.fc2  = nn.Linear(fc_mid, fc_mid // 2)
        self.bn2  = nn.BatchNorm1d(fc_mid // 2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3  = nn.Linear(fc_mid // 2, output_dim)

    def forward(self, x):
        """x : (B, C, N)"""
        # Discard normals if present
        xyz = x[:, :3, :].transpose(1, 2).contiguous()   # (B, N, 3)
        B, N, _ = xyz.shape

        # Initial embed on ALL points
        feat = self.embed(xyz)                            # (B, N, d_init)

        # Stage 0 — two-step downsample
        # Step A: FPS 16384 → pre_fps_n  (pure spatial sampling)
        pre_idx = fps_batch(xyz, min(self.pre_fps_n, N))  # (B, pre_fps_n)
        xyz_pre  = gather_pts(xyz,  pre_idx)              # (B, pre_fps_n, 3)
        feat_pre = gather_pts(feat, pre_idx)              # (B, pre_fps_n, d_init)

        # Step B: TransitionDown pre_fps_n → m0 with kNN aggregation
        # cdist shape: (B, m0, pre_fps_n) — memory-safe
        xyz0, feat0 = self.td0(xyz_pre, feat_pre)         # (B, m0, d0)

        # PT blocks stage 0
        for pt, norm in zip(self.pt0, self.norm0):
            feat0 = norm(feat0 + pt(xyz0, feat0))

        # Stage 1
        xyz1, feat1 = self.td1(xyz0, feat0)               # (B, m1, d1)
        for pt, norm in zip(self.pt1, self.norm1):
            feat1 = norm(feat1 + pt(xyz1, feat1))

        # Stage 2
        xyz2, feat2 = self.td2(xyz1, feat1)               # (B, m2, d2)
        for pt, norm in zip(self.pt2, self.norm2):
            feat2 = norm(feat2 + pt(xyz2, feat2))

        # Global pooling: max + mean → concat
        g_max  = feat2.max(dim=1)[0]                      # (B, d2)
        g_mean = feat2.mean(dim=1)                        # (B, d2)
        g = torch.cat([g_max, g_mean], dim=1)             # (B, d2*2)

        # FC head
        out = self.drop1(F.relu(self.bn1(self.fc1(g))))
        out = self.drop2(F.relu(self.bn2(self.fc2(out))))
        out = self.fc3(out)
        return out
