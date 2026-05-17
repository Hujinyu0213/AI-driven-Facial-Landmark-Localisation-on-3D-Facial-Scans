"""
Joint 9-Landmark Heatmap Soft-Argmax Training
==============================================
两阶段联合训练：
  Stage 1 : PointNet2RegMSG(output_dim=27) — 全局粗定位，9地标联合输出
  Stage 2 : JointHeatmapS2 — 共享骨干 + 9个独立 score_mlp head
            per-point softmax → soft-argmax → 精细坐标

特性:
  - 100样本，80/20 固定拆分（3个seed: 42/123/2024，报告 mean±std）
  - K=5折CV（在80训练样本内）找最优epoch，全量重训
  - Hybrid radius（每地标不同，归一化单位）
  - 评估：mean / median / P75 / P90 / P95 分位数
  - GT覆盖率统计（GT是否落在patch内）
  - σ ablation（0.03 / 0.04 / 0.05），仅在 seed=42 时运行
  - snap：最近点云点 + 局部平面拟合（local surface snap）

用法:
  python scripts/training/train_heatmap_joint.py
  python scripts/training/train_heatmap_joint.py --eval-only
  python scripts/training/train_heatmap_joint.py --seed 42 --sigma 0.04
"""
import argparse, copy, json, os, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(os.path.dirname(BASE_DIR))
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
EXPORT_ROOT = os.path.join(ROOT_DIR, "data", "pointcloud")
UTILS_DIR   = os.path.join(ROOT_DIR, "scripts", "utils")
for p in [ROOT_DIR, MODELS_DIR, UTILS_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

from pointnet2_reg import PointNet2RegMSG
from point_transformer_reg import PointTransformerReg
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as _fps_cuda
    def furthest_point_sample(xyz_tensor, npoint):
        return _fps_cuda(xyz_tensor, npoint)
except ImportError:
    def furthest_point_sample(xyz_tensor, npoint):
        """Pure-Python FPS fallback (no CUDA required)."""
        B, N, _ = xyz_tensor.shape
        device = xyz_tensor.device
        idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
        dist = torch.full((B, N), 1e10, device=device)
        farthest = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(npoint):
            idx[:, i] = farthest
            sel = xyz_tensor[torch.arange(B), farthest].unsqueeze(1)  # (B,1,3)
            d = ((xyz_tensor - sel) ** 2).sum(-1)                      # (B,N)
            dist = torch.min(dist, d)
            farthest = dist.argmax(-1)
        return idx

try:
    from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG as PointNetSetAbstractionMsg
except ImportError:
    from pointnet2_utils import PointNetSetAbstractionMsg as _LegacySAMsg

    class PointNetSetAbstractionMsg(nn.Module):
        """Adapter: 将旧版 pointnet2_utils 接口适配为 pointnet2_ops 兼容接口。
        旧版: __init__(npoint, radius_list, nsample_list, in_channel, mlp_list)
              mlp_list[i] = [d1, d2, ...]  不含 in_channel
              forward: xyz(B,3,N), points(B,D,N) → new_xyz(B,3,S), new_pts(B,D',S)
        新版: __init__(npoint, radii, nsamples, mlps, use_xyz=True)
              mlps[i] = [in_channel, d1, d2, ...]
              forward: xyz(B,N,3), features(B,C,N) → new_xyz(B,S,3), new_pts(B,D',S)
        """
        def __init__(self, npoint, radii, nsamples, mlps, use_xyz=True):
            super().__init__()
            in_channel = mlps[0][0]
            mlp_list = [m[1:] for m in mlps]
            self._impl = _LegacySAMsg(npoint, radii, nsamples, in_channel, mlp_list)

        def forward(self, xyz, features):
            # xyz: (B,N,3) → (B,3,N) for legacy
            xyz_t = xyz.transpose(1, 2).contiguous()
            new_xyz, new_points = self._impl(xyz_t, features)
            # new_xyz: (B,3,S) → (B,S,3) to match pointnet2_ops output
            new_xyz = new_xyz.transpose(1, 2).contiguous()
            return new_xyz, new_points

# ── 地标注册表 ────────────────────────────────────────────────────────────────
LANDMARK_NAMES = [
    "glabella", "nasion", "rhinion", "nasal_tip", "subnasale",
    "alare_r", "alare_l", "zygion_r", "zygion_l",
]
N_LANDMARKS = len(LANDMARK_NAMES)

# Stage2 adaptive radius lower-bound（来自 2026-02-17 最优策略：Adaptive + Hybrid insight）
RADIUS_MIN_DEFAULT = 0.25
RADIUS_MIN_BY_LM = {
    "rhinion": 0.35,
    "nasal_tip": 0.55,
    "zygion_r": 0.57,
    "zygion_l": 0.51,
}
RADIUS_MAX = 0.80

# ── 超参数 ────────────────────────────────────────────────────────────────────
MAX_POINTS   = 16384   # 从三角网格均匀重采样（方案A）
PATCH_POINTS = 512
K_FOLDS      = 5
SEEDS        = [42, 123, 2024]
SIGMA_LIST   = [0.03, 0.04, 0.05]   # σ ablation
SIGMA_DEFAULT= 0.04
LAMBDA_H     = 0.5
CENTER_JITTER= 0.05
MIX_PROB     = 0.5

# Stage 1 backbone: 'pn2' = PointNet++, 'pt' = Point Transformer
_S1_BACKBONE = "pn2"  # overridden by --backbone argument

def build_s1_model():
    """Factory: build Stage 1 model selected by _S1_BACKBONE."""
    if _S1_BACKBONE == "pt":
        return PointTransformerReg(
            output_dim=N_LANDMARKS * 3,
            dropout=DROPOUT_S1,
            dims=(128, 256, 512),   # larger capacity: ~5M params vs 1.3M
            n_pts=(2048, 512, 128),
            k=16,
        )
    else:  # default: pn2
        return PointNet2RegMSG(
            output_dim=N_LANDMARKS * 3,
            normal_channel=False,
            dropout=DROPOUT_S1,
            sa1_radii=SA1_RADII_S1,
            sa2_radii=SA2_RADII_S1,
        )


# Stage 1
BATCH_SIZE_S1    = 8
NUM_EPOCHS_S1    = 300      # 200->300: flip aug needs more epochs to converge
LR_S1            = 0.0003   # v3: further reduced for stable convergence
LR_DECAY_STEP_S1 = 100      # kept for reference (unused; CosineAnnealingLR used)
LR_DECAY_GAMMA_S1= 0.7      # kept for reference (unused; CosineAnnealingLR used)
DROPOUT_S1       = 0.4
SA1_RADII_S1     = [0.1, 0.2, 0.4]
SA2_RADII_S1     = [0.2, 0.4, 0.8]

# Stage 2
BATCH_SIZE_S2    = 4    # ×9 地标 = 36次前向
NUM_EPOCHS_S2    = 300
LR_S2            = 0.001
LR_DECAY_STEP_S2 = 80
LR_DECAY_GAMMA_S2= 0.5
DROPOUT_S2       = 0.3
SA1_RADII_S2     = [0.05, 0.1, 0.2]
SA2_RADII_S2     = [0.1,  0.2, 0.4]

# Surface snap local plane fitting
SNAP_KNN = 10   # 局部平面拟合的近邻点数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class JointHeatmapS2(nn.Module):
    """
    共享骨干 + 9个独立 score_mlp head
    输入: list of 9 patch tensors, 每个 (B, 3, 512)
    输出: preds (9×(B,3)), scores (9×(B,512)), l1_xyz (9×(B,512,3))
    """
    def __init__(self, n_landmarks=9, dropout=0.3, sa1_radii=None):
        super().__init__()
        sa1_radii = sa1_radii or SA1_RADII_S2
        # ── 共享骨干 ──────────────────────────────────────────────────────────
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512, radii=sa1_radii, nsamples=[16, 32, 64],
            mlps=[[0,32,32,64],[0,64,64,128],[0,64,96,128]], use_xyz=True)
        # feat_dim after sa1 = 64+128+128 = 320
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128, radii=SA2_RADII_S2, nsamples=[32, 64, 128],
            mlps=[[320,64,64,128],[320,128,128,256],[320,128,128,256]], use_xyz=True)
        # feat_dim after sa2 = 128+256+256 = 640
        self.global_mlp = nn.Sequential(
            nn.Conv1d(640, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        # ── 9个独立打分头 ──────────────────────────────────────────────────────
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(320+128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 1, 1))
            for _ in range(n_landmarks)
        ])
        self.n_landmarks = n_landmarks

    def forward(self, patches_list):
        """
        patches_list: list of 9 tensors, each (B, 3, 512)
        将全部 B×9 patches 拼成一个大 batch 过骨干，再拆回 9 个 head
        """
        B = patches_list[0].shape[0]
        # (B*9, 3, 512)
        all_patches = torch.cat(patches_list, dim=0)
        xyz_t = all_patches.transpose(1, 2).contiguous()  # (B*9, 512, 3)
        l1_xyz, l1_f = self.sa1(xyz_t, None)   # l1_xyz:(B*9,512,3), l1_f:(B*9,320,512)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f)  # l2_xyz:(B*9,128,3), l2_f:(B*9,640,128)
        g = self.global_mlp(l2_f).max(dim=2)[0]  # (B*9, 128)

        preds, scores_list, l1_xyz_list = [], [], []
        for k in range(self.n_landmarks):
            lf_k  = l1_f [k*B:(k+1)*B]   # (B, 320, 512)
            g_k   = g    [k*B:(k+1)*B]   # (B, 128)
            xyz_k = l1_xyz[k*B:(k+1)*B]  # (B, 512, 3)
            g_e   = g_k.unsqueeze(2).expand(-1, -1, PATCH_POINTS)
            feat  = torch.cat([lf_k, g_e], dim=1)   # (B, 448, 512)
            scores  = self.heads[k](feat).squeeze(1)  # (B, 512)
            weights = torch.softmax(scores, dim=1)
            pred    = (weights.unsqueeze(-1) * xyz_k).sum(dim=1)  # (B, 3)
            preds.append(pred)
            scores_list.append(scores)
            l1_xyz_list.append(xyz_k)
        return preds, scores_list, l1_xyz_list


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def coord_loss(pred, target):
    return F.smooth_l1_loss(pred, target)

def gaussian_heatmap_kl(scores, l1_xyz, gt_residual, sigma):
    gt   = gt_residual.unsqueeze(1).float()
    dist = torch.norm(l1_xyz.float() - gt, dim=2)
    tgt  = torch.exp(-dist**2 / (2 * sigma**2))
    tgt  = tgt / (tgt.sum(dim=1, keepdim=True) + 1e-8)
    lp   = F.log_softmax(scores.float(), dim=1)
    return (tgt * (torch.log(tgt + 1e-8) - lp)).sum(dim=1).mean()

def joint_loss(preds, scores_list, l1_xyz_list, residuals_batch, sigma):
    """所有9地标的联合损失"""
    total = 0.0
    for k in range(N_LANDMARKS):
        res_k = residuals_batch[:, k, :]
        total += coord_loss(preds[k], res_k)
        total += LAMBDA_H * gaussian_heatmap_kl(scores_list[k], l1_xyz_list[k], res_k, sigma)
    return total / N_LANDMARKS


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def fps(pts, n, seed=None):
    if device.type == "cuda" and pts.shape[0] >= n:
        try:
            t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            idx = furthest_point_sample(t, n)
            return pts[idx[0].cpu().numpy()]
        except Exception:
            pass
    rng = np.random.RandomState(seed if seed is not None else 0)
    idx = rng.choice(pts.shape[0], n, replace=pts.shape[0] < n)
    return pts[idx]


def sample_mesh_uniform(vertices, triangles, n_points, seed=None):
    """从三角网格按面积加权均匀采样 n_points 个点（方案A）。

    采样点落在真实曲面上，不是点云插值造假，精度等于原始扫描仪精度。

    Args:
        vertices  : (V, 3) float32 — 归一化后的顶点坐标
        triangles : (T, 3) int32  — 0-indexed 三角面
        n_points  : int
        seed      : int or None
    Returns:
        (n_points, 3) float32
    """
    rng = np.random.RandomState(seed if seed is not None else 0)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    # 按面积加权选三角形
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.maximum(areas, 1e-12)          # 避免退化三角形
    probs = areas / areas.sum()
    chosen = rng.choice(len(triangles), size=n_points, p=probs)
    # 重心坐标随机采样（折叠法保证均匀分布）
    r1 = rng.rand(n_points, 1).astype(np.float32)
    r2 = rng.rand(n_points, 1).astype(np.float32)
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    pts = (1.0 - r1 - r2) * v0[chosen] + r1 * v1[chosen] + r2 * v2[chosen]
    return pts.astype(np.float32)


def augment_batch(pc, lbl, dev):
    """pc: (B,3,N), lbl: (B,9,3) or (B,3)"""
    pc = pc.to(dev); lbl = lbl.to(dev)
    B, _, N = pc.shape
    th = torch.rand(B,1,1,device=dev)*(2*np.pi/12) - np.pi/12
    c, s = torch.cos(th), torch.sin(th)
    rot = torch.zeros(B,3,3,device=dev)
    rot[:,0,0]=c.flatten(); rot[:,0,1]=-s.flatten()
    rot[:,1,0]=s.flatten(); rot[:,1,1]= c.flatten(); rot[:,2,2]=1.0
    pc_r  = torch.bmm(pc.transpose(1,2), rot)
    if lbl.dim() == 3:   # (B, 9, 3)
        lbl_r = torch.bmm(lbl.view(B*N_LANDMARKS,1,3), rot.repeat_interleave(N_LANDMARKS,dim=0)).view(B,N_LANDMARKS,3)
    else:                # (B, 3)
        lbl_r = torch.bmm(lbl.view(B,1,3), rot).view(B,3)
    sc  = torch.rand(B,1,1,device=dev)*0.1+0.95
    pc_r = pc_r*sc
    if lbl.dim() == 3:
        lbl_r = lbl_r * sc.squeeze(-1).unsqueeze(-1)
    else:
        lbl_r = lbl_r * sc.squeeze(-1)
    sh = torch.rand(B,1,3,device=dev)*0.04-0.02
    pc_r = pc_r+sh
    if lbl.dim() == 3:
        lbl_r = lbl_r + sh.squeeze(1).unsqueeze(1)
    else:
        lbl_r = lbl_r + sh.squeeze(1)
    pc_r = pc_r + torch.randn(B,N,3,device=dev)*0.005

    # 5. Left-right mirror flip (prob=0.3)
    # X axis = left-right direction after normalization.
    # Bilateral landmark pairs: alare_r(5)<->alare_l(6), zygion_r(7)<->zygion_l(8)
    flip_idx = (torch.rand(B, device=dev) < 0.2).nonzero(as_tuple=True)[0]  # v3: reduced from 0.3
    if len(flip_idx) > 0:
        pc_r[flip_idx, :, 0] = -pc_r[flip_idx, :, 0]
        if lbl.dim() == 3:   # (B, 9, 3)
            lbl_r[flip_idx, :, 0] = -lbl_r[flip_idx, :, 0]
            saved_5 = lbl_r[flip_idx, 5].clone()
            saved_7 = lbl_r[flip_idx, 7].clone()
            lbl_r[flip_idx, 5] = lbl_r[flip_idx, 6]
            lbl_r[flip_idx, 6] = saved_5
            lbl_r[flip_idx, 7] = lbl_r[flip_idx, 8]
            lbl_r[flip_idx, 8] = saved_7
        else:                # (B, 3) single landmark
            lbl_r[flip_idx, 0] = -lbl_r[flip_idx, 0]

    return pc_r.transpose(1,2), lbl_r


def snap_to_nearest(pred, pc_full):
    """snap到最近点云点（归一化空间）"""
    idx = np.argmin(np.linalg.norm(pc_full - pred, axis=1))
    return pc_full[idx]


def _closest_on_triangles_batch(p, v0, v1, v2):
    """
    Ericson (2005) 向量化版：计算点 p 到每个三角面的最近点。
    p : (3,)
    v0, v1, v2 : (T, 3) float64  三角面顶点
    返回 (T, 3) float64
    """
    p  = p.astype(np.float64)
    v0 = v0.astype(np.float64)
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)

    ab = v1 - v0; ac = v2 - v0
    ap = p - v0
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)

    bp = p - v1
    d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)

    cp = p - v2
    d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)

    vc = d1*d4 - d3*d2
    vb = d5*d2 - d1*d6
    va = d3*d6 - d5*d4
    denom = va + vb + vc

    T = len(v0)
    result = np.empty((T, 3), dtype=np.float64)

    # vertex A
    m = (d1 <= 0) & (d2 <= 0)
    result[m] = v0[m]

    # vertex B
    m = (~((d1<=0)&(d2<=0))) & (d3 >= 0) & (d4 <= d3)
    result[m] = v1[m]

    # vertex C
    m = (~((d1<=0)&(d2<=0))) & (~((d3>=0)&(d4<=d3))) & (d6 >= 0) & (d5 <= d6)
    result[m] = v2[m]

    # edge AB
    m_done = ((d1<=0)&(d2<=0)) | ((d3>=0)&(d4<=d3)) | ((d6>=0)&(d5<=d6))
    m = (~m_done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    if m.any():
        sD = d1[m] - d3[m]; s = np.where(np.abs(sD)>1e-30, np.clip(d1[m]/sD,0,1), 0.0)
        result[m] = v0[m] + s[:,None]*ab[m]

    # edge AC
    m_done2 = m_done | (m)
    m_eAB = (~m_done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    m = (~m_done) & (~m_eAB) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    if m.any():
        wD = d2[m] - d6[m]; w = np.where(np.abs(wD)>1e-30, np.clip(d2[m]/wD,0,1), 0.0)
        result[m] = v0[m] + w[:,None]*ac[m]

    # edge BC
    m_eAC = (~m_done) & (~m_eAB) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    m = (~m_done) & (~m_eAB) & (~m_eAC) & (va <= 0) & ((d4-d3) >= 0) & ((d5-d6) >= 0)
    if m.any():
        num = d4[m]-d3[m]; den = num+(d5[m]-d6[m])
        w = np.where(np.abs(den)>1e-30, np.clip(num/den,0,1), 0.0)
        result[m] = v1[m] + w[:,None]*(v2[m]-v1[m])

    # interior
    m_interior = ~(m_done | m_eAB | m_eAC |
                   ((~m_done)&(~m_eAB)&(~m_eAC)&(va<=0)&((d4-d3)>=0)&((d5-d6)>=0)))
    if m_interior.any():
        d = denom[m_interior]
        safe = np.abs(d) > 1e-30
        v_c = np.where(safe, vb[m_interior]/np.where(safe,d,1.0), 1/3.0)
        w_c = np.where(safe, vc[m_interior]/np.where(safe,d,1.0), 1/3.0)
        result[m_interior] = v0[m_interior] + v_c[:,None]*ab[m_interior] + w_c[:,None]*ac[m_interior]

    return result


def snap_to_mesh(pred, pc_full, triangles):
    """
    精确三角网格投影 snap。
    找使 pred 到最近点距离最小的三角面，将 pred 投影到该三角面上。
    triangles : (T, 3) int32 — 0-indexed 三角面索引
    若 triangles 为空，回退到最近点 snap。
    """
    if triangles is None or len(triangles) == 0:
        return snap_to_nearest(pred, pc_full)
    v0 = pc_full[triangles[:, 0]]
    v1 = pc_full[triangles[:, 1]]
    v2 = pc_full[triangles[:, 2]]
    closest = _closest_on_triangles_batch(pred, v0, v1, v2)   # (T, 3)
    dists   = np.linalg.norm(closest - pred.astype(np.float64), axis=1)
    best    = np.argmin(dists)
    return closest[best].astype(np.float32)


def snap_to_surface(pred, pc_full, k=SNAP_KNN, triangles=None):
    """
    若有三角网格则用网格投影，否则回退到局部平面拟合（兼容旧调用）。
    """
    if triangles is not None and len(triangles) > 0:
        return snap_to_mesh(pred, pc_full, triangles)
    # 局部平面拟合（fallback）
    dists = np.linalg.norm(pc_full - pred, axis=1)
    knn_idx = np.argsort(dists)[:k]
    knn_pts = pc_full[knn_idx].astype(np.float64)
    centroid = knn_pts.mean(0)
    cov = np.cov((knn_pts - centroid).T)
    try:
        _, eigvecs = np.linalg.eigh(cov)
        normal  = eigvecs[:, 0]
        snapped = pred.astype(np.float64) - np.dot(pred.astype(np.float64)-centroid, normal)*normal
        return snapped.astype(np.float32)
    except Exception:
        return snap_to_nearest(pred, pc_full)


def percentile_stats(errors):
    errs = np.array(errors)
    return {
        "mean":   float(np.mean(errs)),
        "median": float(np.median(errs)),
        "p75":    float(np.percentile(errs, 75)),
        "p90":    float(np.percentile(errs, 90)),
        "p95":    float(np.percentile(errs, 95)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    返回:
      X          : (N, 3, MAX_POINTS) float32  — 网格均匀采样归一化点云（方案A）
      Y          : (N, 9, 3)   float32  — 归一化地标坐标
      Scales     : (N,)        float32  — std scale (mm)
      names      : list[str]
      full_clouds: list of (M_i, 3) float32 — 完整归一化点云（用于snap）
      triangles  : list of (T_i, 3) int32   — 三角面0-indexed索引（用于mesh snap）
    """
    sample_names = sorted([d for d in os.listdir(EXPORT_ROOT)
                           if os.path.isdir(os.path.join(EXPORT_ROOT, d))])
    X_list, Y_list, S_list, valid, clouds, tris = [], [], [], [], [], []
    n_with_mesh = 0
    for name in tqdm(sample_names, desc="loading data", ascii=True, file=sys.stdout):
        pc_p  = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        lm_p  = os.path.join(EXPORT_ROOT, name, "nose_landmarks.npy")
        tri_p = os.path.join(EXPORT_ROOT, name, "triangles.npy")
        if not (os.path.exists(pc_p) and os.path.exists(lm_p)): continue
        pc = np.load(pc_p).astype(np.float32)
        if pc.shape[0] == 0: continue
        lm = np.load(lm_p).astype(np.float32)
        if lm.ndim == 1: lm = lm.reshape(-1, 3)
        if lm.shape[0] < 9: continue
        lm9 = lm[-9:]                          # (9, 3)
        centroid = pc.mean(0)
        pc_c = pc - centroid
        scale = float(np.std(pc_c))
        scale = scale if scale > 1e-6 else 1.0
        pc_c /= scale
        lm9_c = (lm9 - centroid) / scale       # (9, 3)
        # 方案A：从三角网格均匀采样 MAX_POINTS 个点；无网格时回退到 fps
        samp_seed = abs(hash(name)) % 10000
        if os.path.exists(tri_p):
            _tri_tmp = np.load(tri_p)
            pc_fps = sample_mesh_uniform(pc_c, _tri_tmp, MAX_POINTS, seed=samp_seed)
        else:
            pc_fps = fps(pc_c, MAX_POINTS, seed=samp_seed)
        X_list.append(pc_fps.T.astype(np.float32))    # (3, MAX_POINTS)
        Y_list.append(lm9_c.astype(np.float32))       # (9, 3)
        S_list.append(np.float32(scale))
        valid.append(name)
        clouds.append(pc_c.astype(np.float32))
        # 三角面（归一化空间的索引不变，顶点坐标已归一化）
        if os.path.exists(tri_p):
            tri = np.load(tri_p)  # (T, 3) int32
            tris.append(tri)
            n_with_mesh += 1
        else:
            tris.append(None)   # 没有网格时用 None
    X = np.stack(X_list)       # (N, 3, MAX_POINTS)
    Y = np.stack(Y_list)       # (N, 9, 3)
    S = np.array(S_list, dtype=np.float32)
    print(f"[data] {len(X)} samples, {n_with_mesh} with triangle mesh")
    if n_with_mesh == 0:
        print("  [warn] triangles.npy not found; snap will fall back to local plane fit.")
        print("  [warn] run: python scripts/data_processing/convert_inp_to_npy.py")
    return X, Y, S, valid, clouds, tris


# ══════════════════════════════════════════════════════════════════════════════
# GT Coverage stats
# ══════════════════════════════════════════════════════════════════════════════

def print_coverage(X, Y, coarse, radii, tag="训练集"):
    """统计每个地标GT是否落在patch内（以Stage1粗预测为patch中心）"""
    N = len(X)
    print(f"\n-- GT coverage [{tag}, N={N}] --")
    print(f"  {'landmark':<12} {'radius':>8} {'coverage':>9} {'GT-ctr dist':>12}")
    print("  " + "-"*45)
    for k, lm in enumerate(LANDMARK_NAMES):
        r = radii[k]
        covered = 0
        dist_list = []
        for i in range(N):
            gt  = Y[i, k]
            ctr = coarse[i, k]
            d   = np.linalg.norm(gt - ctr)
            dist_list.append(d)
            if d < r:
                covered += 1
        cov_rate = covered / N * 100
        mean_dist = np.mean(dist_list)
        print(f"  {lm:<12} {r:>8.3f} {cov_rate:>7.1f}%  {mean_dist:>10.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════════════════

def joint_collate(batch):
    patches_list = [[] for _ in range(N_LANDMARKS)]
    residuals_list = []
    for patches, residuals in batch:
        for k in range(N_LANDMARKS):
            patches_list[k].append(patches[k])
        residuals_list.append(np.stack(residuals))  # (9, 3)
    patches_tensor  = [torch.from_numpy(np.stack(patches_list[k])).float()
                       for k in range(N_LANDMARKS)]
    residuals_tensor = torch.from_numpy(np.stack(residuals_list)).float()  # (B, 9, 3)
    return patches_tensor, residuals_tensor


class JointPatchDataset(Dataset):
    def __init__(self, X, Y, coarse_all, radii, jitter_std, mode="train", mix_prob=0.0):
        self.X = X; self.Y = Y; self.coarse_all = coarse_all
        self.radii = radii; self.jitter_std = jitter_std
        self.mode = mode; self.mix_prob = mix_prob

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        pc = self.X[i].T         # (N, 3)
        gt_all  = self.Y[i]      # (9, 3)
        ctr_all = self.coarse_all[i]  # (9, 3)
        patches, residuals = [], []
        for k in range(N_LANDMARKS):
            r   = self.radii[k]
            gt  = gt_all[k]; ctr = ctr_all[k]
            use_gt = (self.mode == "train") and (np.random.rand() < self.mix_prob)
            center = gt if use_gt else ctr
            if isinstance(self.jitter_std, (list, tuple, np.ndarray)):
                jitter_k = float(self.jitter_std[k])
            else:
                jitter_k = float(self.jitter_std)
            if self.mode == "train" and jitter_k > 0:
                center = center + np.random.normal(0, jitter_k, 3).astype(np.float32)
            dists  = np.linalg.norm(pc - center, axis=1)
            inside = np.where(dists < r)[0]
            if inside.size == 0:
                inside = np.argsort(dists)[:PATCH_POINTS]
            if inside.size < PATCH_POINTS:
                inside = np.concatenate([inside,
                    np.random.choice(inside, PATCH_POINTS-inside.size, replace=True)])
            else:
                inside = np.random.choice(inside, PATCH_POINTS, replace=False)
            patch    = (pc[inside] - center).T.astype(np.float32)  # (3, 512)
            residual = (gt - center).astype(np.float32)              # (3,)
            patches.append(patch)
            residuals.append(residual)
        return patches, residuals


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1
# ══════════════════════════════════════════════════════════════════════════════

def predict_stage1(model_s1, X):
    """返回 (N, 9, 3)"""
    model_s1.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 16):
            batch = torch.from_numpy(X[i:i+16]).float().to(device)
            pred  = model_s1(batch).cpu().numpy()  # (B, 27)
            out.append(pred.reshape(-1, N_LANDMARKS, 3))
    return np.concatenate(out, axis=0)  # (N, 9, 3)


def estimate_stage2_patch_params(coarse, gt, tag=""):
    """基于 Stage1 误差自适应估计 Stage2 的每地标 radius / jitter。

    采用历史最优可泛化策略：
      - jitter_k = max(0.05, p80(err_k))
      - radius_k = clamp(max(0.25, p95(err_k)*1.2), min_by_landmark[k], RADIUS_MAX)
    """
    err = np.linalg.norm(coarse - gt, axis=2)  # (N, 9), normalized
    p80 = np.percentile(err, 80, axis=0)
    p95 = np.percentile(err, 95, axis=0)

    jitter = np.maximum(CENTER_JITTER, p80).astype(np.float32)
    radii = []
    for k, lm in enumerate(LANDMARK_NAMES):
        base_r = max(RADIUS_MIN_DEFAULT, float(p95[k]) * 1.2)
        min_r = RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)
        r = min(max(base_r, min_r), RADIUS_MAX)
        radii.append(float(r))

    print(f"\n[Stage2 adaptive patch params] {tag}".rstrip())
    print(f"  {'landmark':<12} {'p80(err)':>9} {'p95(err)':>9} {'jitter':>9} {'radius':>9}")
    print("  " + "-" * 54)
    for k, lm in enumerate(LANDMARK_NAMES):
        print(f"  {lm:<12} {p80[k]:>9.4f} {p95[k]:>9.4f} {jitter[k]:>9.4f} {radii[k]:>9.4f}")
    return radii, jitter.tolist()


def train_stage1_kfold(Xtr, Ytr, Str, tag):
    """K折CV找最优epoch，返回best_epoch"""
    print(f"\n[Stage1 KFold] {tag}  K={K_FOLDS}  epochs={NUM_EPOCHS_S1}")
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_best_eps = []

    for fold, (tr_i, val_i) in enumerate(kf.split(Xtr), 1):
        Xf, Yf = Xtr[tr_i], Ytr[tr_i]
        Xv, Yv, Sv = Xtr[val_i], Ytr[val_i], Str[val_i]
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xf).float(), torch.from_numpy(Yf.reshape(len(Yf),-1)).float()),
            batch_size=BATCH_SIZE_S1, shuffle=True)
        model = build_s1_model().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR_S1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS_S1, eta_min=5e-6)  # v3
        Xv_t  = torch.from_numpy(Xv).float().to(device)
        best_val, best_ep = float("inf"), NUM_EPOCHS_S1

        for ep in range(1, NUM_EPOCHS_S1+1):
            model.train()
            for pc, lbl in loader:
                pc_a, lbl_a = augment_batch(pc, lbl.view(-1, N_LANDMARKS, 3), device)
                opt.zero_grad()
                pred = model(pc_a)
                loss_kf = F.smooth_l1_loss(pred, lbl_a.view(-1, N_LANDMARKS*3).to(device))
                loss_kf.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # v3
                opt.step()
            sched.step()
            if ep % 20 == 0 or ep == NUM_EPOCHS_S1:
                model.eval()
                with torch.no_grad():
                    pv = model(Xv_t).cpu().numpy().reshape(-1, N_LANDMARKS, 3)
                errs = np.linalg.norm(pv - Yv, axis=2) * Sv[:, None]  # (Nv, 9)
                val_mm = float(errs.mean())
                marker = " <-- best" if val_mm < best_val else ""
                print(f"  Fold{fold} ep{ep:3d}  val={val_mm:.3f}mm{marker}")
                if val_mm < best_val:
                    best_val, best_ep = val_mm, ep
        fold_best_eps.append(best_ep)

    best_epoch = int(np.median(fold_best_eps))
    print(f"\n[Stage1] best epoch (median)={best_epoch}  per-fold={fold_best_eps}")
    return best_epoch


def train_stage1_full(Xtr, Ytr, Str, best_epoch, tag):
    """全量重训 best_epoch 步"""
    print(f"\n[Stage1] full retrain {best_epoch} epochs on {len(Xtr)} samples")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr.reshape(len(Ytr),-1)).float()),
        batch_size=BATCH_SIZE_S1, shuffle=True)
    model = build_s1_model().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_S1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=best_epoch, eta_min=5e-6)  # v3
    for ep in range(1, best_epoch+1):
        model.train()
        acc = 0.0
        for pc, lbl in loader:
            pc_a, lbl_a = augment_batch(pc, lbl.view(-1, N_LANDMARKS, 3), device)
            opt.zero_grad()
            pred = model(pc_a)
            loss = F.smooth_l1_loss(pred, lbl_a.view(-1, N_LANDMARKS*3).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # v3
            opt.step(); acc += loss.item()
        sched.step()
        if ep % 20 == 0 or ep == best_epoch:
            print(f"  ep{ep:3d}  loss={acc/max(1,len(loader)):.5f}")
    path = os.path.join(MODELS_DIR, f"joint_stage1_{tag}.pth")
    torch.save(model.state_dict(), path)
    print(f"[save] Stage1 -> {path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2
# ══════════════════════════════════════════════════════════════════════════════

def train_stage2_kfold(Xtr, Ytr, Str, coarse_tr, radii, sigma, jitter, tag):
    """K折CV找最优epoch，返回best_epoch"""
    print(f"\n[Stage2 KFold] {tag}  sigma={sigma}  K={K_FOLDS}  epochs={NUM_EPOCHS_S2}")
    print(f"  jitter(mean)={float(np.mean(jitter)):.4f}  (per-landmark p80)")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_best_eps = []

    for fold, (tr_i, val_i) in enumerate(kf.split(Xtr), 1):
        Xf, Yf, ctr_f = Xtr[tr_i], Ytr[tr_i], coarse_tr[tr_i]
        Xv, Yv, Sv, ctr_v = Xtr[val_i], Ytr[val_i], Str[val_i], coarse_tr[val_i]
        ds = JointPatchDataset(Xf, Yf, ctr_f, radii, jitter, mode="train")
        loader = DataLoader(ds, batch_size=BATCH_SIZE_S2, shuffle=True,
                            collate_fn=joint_collate)
        model = JointHeatmapS2(n_landmarks=N_LANDMARKS, dropout=DROPOUT_S2,
                               sa1_radii=SA1_RADII_S2).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR_S2)
        sched = torch.optim.lr_scheduler.StepLR(opt, LR_DECAY_STEP_S2, LR_DECAY_GAMMA_S2)
        best_val, best_ep = float("inf"), NUM_EPOCHS_S2

        for ep in range(1, NUM_EPOCHS_S2+1):
            model.train()
            for patches_list, residuals_batch in loader:
                patches_list = [p.to(device) for p in patches_list]
                residuals_batch = residuals_batch.to(device)
                opt.zero_grad()
                preds, scores_list, l1_xyz_list = model(patches_list)
                loss = joint_loss(preds, scores_list, l1_xyz_list, residuals_batch, sigma)
                loss.backward(); opt.step()
            sched.step()
            if ep % 30 == 0 or ep == NUM_EPOCHS_S2:
                model.eval()
                errs_all = []
                for i in range(len(Xv)):
                    pc = Xv[i].T; gt_all = Yv[i]; ctr_all = ctr_v[i]
                    sample_patches = []
                    for k in range(N_LANDMARKS):
                        r = radii[k]; ctr = ctr_all[k]
                        dists = np.linalg.norm(pc - ctr, axis=1)
                        inside = np.where(dists < r)[0]
                        if inside.size == 0: inside = np.argsort(dists)[:PATCH_POINTS]
                        if inside.size < PATCH_POINTS:
                            inside = np.concatenate([inside,
                                np.random.choice(inside, PATCH_POINTS-inside.size, replace=True)])
                        else:
                            inside = np.random.choice(inside, PATCH_POINTS, replace=False)
                        patch = (pc[inside] - ctr).T.astype(np.float32)
                        sample_patches.append(torch.from_numpy(patch[None]).float().to(device))
                    with torch.no_grad():
                        preds_i, _, _ = model(sample_patches)
                    for k in range(N_LANDMARKS):
                        pred_r = preds_i[k].cpu().numpy().reshape(3)
                        refined = ctr_all[k] + pred_r
                        errs_all.append(np.linalg.norm(refined - gt_all[k]) * Sv[i])
                val_mm = float(np.mean(errs_all))
                marker = " <-- best" if val_mm < best_val else ""
                print(f"  Fold{fold} ep{ep:3d}  val={val_mm:.3f}mm{marker}")
                if val_mm < best_val:
                    best_val, best_ep = val_mm, ep
        fold_best_eps.append(best_ep)

    best_epoch = int(np.median(fold_best_eps))
    print(f"\n[Stage2] best epoch (median)={best_epoch}  per-fold={fold_best_eps}")
    return best_epoch


def train_stage2_full(Xtr, Ytr, coarse_tr, radii, sigma, jitter, best_epoch, tag):
    """全量重训 best_epoch 步"""
    print(f"\n[Stage2] full retrain {best_epoch} epochs on {len(Xtr)} samples  sigma={sigma}")
    ds = JointPatchDataset(Xtr, Ytr, coarse_tr, radii, jitter, mode="train")
    loader = DataLoader(ds, batch_size=BATCH_SIZE_S2, shuffle=True, collate_fn=joint_collate)
    model = JointHeatmapS2(n_landmarks=N_LANDMARKS, dropout=DROPOUT_S2,
                           sa1_radii=SA1_RADII_S2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_S2)
    sched = torch.optim.lr_scheduler.StepLR(opt, LR_DECAY_STEP_S2, LR_DECAY_GAMMA_S2)
    for ep in range(1, best_epoch+1):
        model.train()
        acc = 0.0
        for patches_list, residuals_batch in loader:
            patches_list = [p.to(device) for p in patches_list]
            residuals_batch = residuals_batch.to(device)
            opt.zero_grad()
            preds, scores_list, l1_xyz_list = model(patches_list)
            loss = joint_loss(preds, scores_list, l1_xyz_list, residuals_batch, sigma)
            loss.backward(); opt.step(); acc += loss.item()
        sched.step()
        if ep % 30 == 0 or ep == best_epoch:
            print(f"  ep{ep:3d}  loss={acc/max(1,len(loader)):.5f}")
    path = os.path.join(MODELS_DIR, f"joint_heatmap_s2_{tag}.pth")
    torch.save(model.state_dict(), path)
    print(f"[save] Stage2 -> {path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model_s1, model_s2, Xte, Yte, Ste, full_clouds_te, triangles_te,
             radii, tag, eval_seed=42):
    """
    评估：粗定位 / soft-argmax精定位 / snap-nearest / snap-mesh
    triangles_te : list of (T,3) int32 or None，与 full_clouds_te 同长度
    返回: dict with per-landmark and overall stats
    """
    model_s1.eval(); model_s2.eval()
    per_lm_refined = [[] for _ in range(N_LANDMARKS)]
    per_lm_snap_pt = [[] for _ in range(N_LANDMARKS)]
    per_lm_snap_sf = [[] for _ in range(N_LANDMARKS)]
    per_lm_coarse  = [[] for _ in range(N_LANDMARKS)]

    rng = np.random.RandomState(eval_seed)
    for i in range(len(Xte)):
        pc_fps   = Xte[i].T
        pc_full  = full_clouds_te[i]
        tri      = triangles_te[i] if triangles_te is not None else None
        gt_all   = Yte[i]
        sc       = Ste[i]
        ctr_all  = predict_stage1(model_s1, Xte[i:i+1])[0]  # (9, 3)

        sample_patches = []
        for k in range(N_LANDMARKS):
            r = radii[k]; ctr = ctr_all[k]
            dists = np.linalg.norm(pc_fps - ctr, axis=1)
            inside = np.where(dists < r)[0]
            if inside.size == 0: inside = np.argsort(dists)[:PATCH_POINTS]
            if inside.size < PATCH_POINTS:
                inside = np.concatenate([inside,
                    rng.choice(inside, PATCH_POINTS-inside.size, replace=True)])
            else:
                inside = rng.choice(inside, PATCH_POINTS, replace=False)
            patch = (pc_fps[inside] - ctr).T.astype(np.float32)
            sample_patches.append(torch.from_numpy(patch[None]).float().to(device))

        with torch.no_grad():
            preds_i, _, _ = model_s2(sample_patches)

        for k in range(N_LANDMARKS):
            gt      = gt_all[k]
            ctr     = ctr_all[k]
            pred_r  = preds_i[k].cpu().numpy().reshape(3)
            refined = ctr + pred_r
            snap_pt = snap_to_nearest(refined, pc_full)
            snap_sf = snap_to_mesh(refined, pc_full, tri)   # 真实网格投影

            per_lm_coarse [k].append(np.linalg.norm(ctr      - gt) * sc)
            per_lm_refined[k].append(np.linalg.norm(refined   - gt) * sc)
            per_lm_snap_pt[k].append(np.linalg.norm(snap_pt   - gt) * sc)
            per_lm_snap_sf[k].append(np.linalg.norm(snap_sf   - gt) * sc)

    # ── 打印结果 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Eval results  [{tag}]  test N={len(Xte)}")
    print(f"{'='*70}")
    print(f"  {'landmark':<12} {'coarse':>8} {'soft-argmax':>12} {'snap-pt':>9} {'snap-mesh':>10}  (mean mm)")
    print(f"  {'-'*55}")
    all_refined, all_snap_pt, all_snap_sf = [], [], []
    per_lm_results = {}
    for k, lm in enumerate(LANDMARK_NAMES):
        c_m  = np.mean(per_lm_coarse [k])
        r_m  = np.mean(per_lm_refined[k])
        sp_m = np.mean(per_lm_snap_pt[k])
        ss_m = np.mean(per_lm_snap_sf[k])
        print(f"  {lm:<12} {c_m:>8.3f}  {r_m:>10.3f}  {sp_m:>8.3f}  {ss_m:>9.3f}")
        all_refined  += per_lm_refined[k]
        all_snap_pt  += per_lm_snap_pt[k]
        all_snap_sf  += per_lm_snap_sf[k]
        per_lm_results[lm] = {
            "coarse":      percentile_stats(per_lm_coarse [k]),
            "soft_argmax": percentile_stats(per_lm_refined[k]),
            "snap_nearest":percentile_stats(per_lm_snap_pt[k]),
            "snap_mesh":   percentile_stats(per_lm_snap_sf[k]),
        }

    print(f"  {'-'*55}")
    r_stats  = percentile_stats(all_refined)
    sp_stats = percentile_stats(all_snap_pt)
    ss_stats = percentile_stats(all_snap_sf)
    print(f"  {'OVERALL':<12} {'':>8}  {r_stats['mean']:>10.3f}  {sp_stats['mean']:>8.3f}  {ss_stats['mean']:>9.3f}  mm (mean)")
    print(f"  {'median':<12} {'':>8}  {r_stats['median']:>10.3f}  {sp_stats['median']:>8.3f}  {ss_stats['median']:>9.3f}  mm")
    print(f"  {'P90':<12} {'':>8}  {r_stats['p90']:>10.3f}  {sp_stats['p90']:>8.3f}  {ss_stats['p90']:>9.3f}  mm")
    print(f"{'='*70}\n")

    return {
        "per_landmark": per_lm_results,
        "overall": {
            "soft_argmax":  r_stats,
            "snap_nearest": sp_stats,
            "snap_mesh":    ss_stats,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed, sigma, X, Y, Scales, names, full_clouds, triangles, eval_only=False):
    """运行单个 seed 的完整流程"""
    tag = f"seed{seed}_sigma{sigma}"
    print(f"\n{'#'*70}")
    print(f"  SEED={seed}  SIGMA={sigma}")
    print(f"{'#'*70}\n")

    # 80/20 拆分
    rng = np.random.RandomState(seed)
    N   = len(X)
    n_test  = max(1, int(N * 0.20))
    test_idx  = rng.choice(N, n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(N), test_idx)
    Xtr, Ytr, Str = X[train_idx], Y[train_idx], Scales[train_idx]
    Xte, Yte, Ste = X[test_idx],  Y[test_idx],  Scales[test_idx]
    full_clouds_te = [full_clouds[i] for i in test_idx]
    triangles_te   = [triangles[i]   for i in test_idx] if triangles else None
    print(f"  train: {len(Xtr)}  test: {len(Ste)}")

    s1_path = os.path.join(MODELS_DIR, f"joint_stage1_{tag}.pth")
    s2_path = os.path.join(MODELS_DIR, f"joint_heatmap_s2_{tag}.pth")

    if eval_only:
        if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
            print(f"[skip] model not found: {s1_path}")
            return None
        model_s1 = build_s1_model().to(device)
        model_s1.load_state_dict(torch.load(s1_path, map_location=device, weights_only=False))
        model_s2 = JointHeatmapS2(n_landmarks=N_LANDMARKS, dropout=DROPOUT_S2,
                                  sa1_radii=SA1_RADII_S2).to(device)
        model_s2.load_state_dict(torch.load(s2_path, map_location=device, weights_only=False))
        coarse_tr = predict_stage1(model_s1, Xtr)
        radii_s2, jitter_s2 = estimate_stage2_patch_params(coarse_tr, Ytr, tag=f"train {tag} [eval-only]")
    else:
        # ── Stage 1 ───────────────────────────────────────────────────────────
        best_ep_s1 = train_stage1_kfold(Xtr, Ytr, Str, tag)
        model_s1   = train_stage1_full(Xtr, Ytr, Str, best_ep_s1, tag)

        # ── GT 覆盖率（在训练集上）────────────────────────────────────────────
        coarse_tr = predict_stage1(model_s1, Xtr)
        radii_s2, jitter_s2 = estimate_stage2_patch_params(coarse_tr, Ytr, tag=f"train {tag}")
        print_coverage(Xtr, Ytr, coarse_tr, radii_s2, tag=f"train {tag}")

        # ── Stage 2 ───────────────────────────────────────────────────────────
        best_ep_s2 = train_stage2_kfold(Xtr, Ytr, Str, coarse_tr, radii_s2, sigma, jitter_s2, tag)
        model_s2 = train_stage2_full(Xtr, Ytr, coarse_tr, radii_s2, sigma, jitter_s2, best_ep_s2, tag)

    # ── 测试集评估 ─────────────────────────────────────────────────────────────
    results = evaluate(model_s1, model_s2, Xte, Yte, Ste, full_clouds_te, triangles_te,
                       radii_s2, tag, eval_seed=seed)
    results["stage2_patch_params"] = {
        "mode": "adaptive_p95x1.2_with_hybrid_floor",
        "radius": {lm: float(radii_s2[k]) for k, lm in enumerate(LANDMARK_NAMES)},
        "jitter": {lm: float(jitter_s2[k]) for k, lm in enumerate(LANDMARK_NAMES)},
    }
    results["seed"] = seed; results["sigma"] = sigma
    out_path = os.path.join(RESULTS_DIR, f"joint_eval_{tag}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[save] eval results -> {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--backbone", default="pn2", choices=["pn2", "pt"],
                    help="Stage 1 backbone: pn2=PointNet++ (default), pt=Point Transformer")
    ap.add_argument("--seed",  type=int, default=None, help="指定单个seed（默认跑全部3个）")
    ap.add_argument("--sigma", type=float, default=None, help="指定sigma（默认只跑0.04，ablation另行处理）")
    ap.add_argument("--ablation-sigma", action="store_true", help="在seed=42上对σ做 ablation")
    args = ap.parse_args()

    global _S1_BACKBONE
    _S1_BACKBONE = args.backbone
    print(f"[Config] Stage 1 backbone = {_S1_BACKBONE}")

    # 加载数据（只加载一次）
    X, Y, Scales, names, full_clouds, triangles = load_data()

    seed_list  = [args.seed]  if args.seed  is not None else SEEDS
    sigma_list = [args.sigma] if args.sigma is not None else [SIGMA_DEFAULT]

    # ── σ ablation (seed=42, σ ∈ {0.03, 0.04, 0.05}) ──────────────────────────
    if args.ablation_sigma:
        print("\n" + "="*70)
        print("  sigma ABLATION  (seed=42)")
        print("="*70)
        ablation_results = {}
        for sig in SIGMA_LIST:
            res = run_one_seed(42, sig, X, Y, Scales, names, full_clouds, triangles, eval_only=args.eval_only)
            if res:
                ablation_results[str(sig)] = res["overall"]["soft_argmax"]["mean"]
        print("\n-- sigma ablation summary --")
        for sig, val in ablation_results.items():
            print(f"  sigma={sig}  soft-argmax mean={val:.3f} mm")
        return

    # ── 多seed运行（报告 mean±std）────────────────────────────────────────────
    all_results = []
    for seed in seed_list:
        sig = sigma_list[0]
        res = run_one_seed(seed, sig, X, Y, Scales, names, full_clouds, triangles, eval_only=args.eval_only)
        if res:
            all_results.append(res)

    if len(all_results) > 1:
        print("\n" + "="*70)
        print(f"  Multi-seed summary  ({len(all_results)} seeds)")
        print("="*70)
        metrics = ["soft_argmax", "snap_nearest", "snap_mesh"]
        for m in metrics:
            vals = [r["overall"][m]["mean"] for r in all_results]
            print(f"  {m:<15}  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  mm")
        print()

        # 各地标 mean±std
        print(f"  {'landmark':<12} {'soft mean':>10} {'+-std':>6}  {'snap-sf mean':>13} {'+-std':>6}")
        print("  " + "-"*50)
        for lm in LANDMARK_NAMES:
            sm_vals  = [r["per_landmark"][lm]["soft_argmax"]["mean"] for r in all_results]
            ss_vals  = [r["per_landmark"][lm]["snap_mesh"]["mean"]    for r in all_results]
            print(f"  {lm:<12} {np.mean(sm_vals):>10.3f} {np.std(sm_vals):>6.3f}"
                  f"  {np.mean(ss_vals):>13.3f} {np.std(ss_vals):>6.3f}")

        # 保存汇总
        summary = {
            "seeds": [r["seed"] for r in all_results],
            "sigma": all_results[0]["sigma"],
            "metrics": {
                m: {"mean": float(np.mean([r["overall"][m]["mean"] for r in all_results])),
                    "std":  float(np.std ([r["overall"][m]["mean"] for r in all_results]))}
                for m in metrics
            },
        }
        out = os.path.join(RESULTS_DIR, "joint_eval_multiseed_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[save] multi-seed summary -> {out}")


if __name__ == "__main__":
    main()
