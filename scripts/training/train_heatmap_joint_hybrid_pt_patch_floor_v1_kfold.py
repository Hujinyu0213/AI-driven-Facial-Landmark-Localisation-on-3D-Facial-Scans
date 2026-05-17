"""
Hybrid PT+Patch coarse-to-fine training for 9 facial landmarks.

Combined method:
    - Stage1: PointTransformerEncoder + CoarseHead (from soft-ROI script)
    - Stage2: explicit local patch heatmap refiner (from unified/flip_v3 route)

Radius policy:
    - Option A (fixed floor radii): directly use landmark-specific radius floors
        and DO NOT run adaptive radius estimation.
    - Optional comparison mode: estimate adaptive radii after a Stage1 warmup using
        radius_k = clamp(max(0.25, 1.2*p95(error_k)), floor_k, 0.80).

Training policy:
    - Single-phase end-to-end + curriculum (one optimizer, always joint backward)
    - Curriculum via mix_prob, loss weights, and radius scale scheduling.
"""
import argparse
import copy
import json
import os
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import train_heatmap_joint_flip_v3 as base
import train_heatmap_joint_flip_v5_cascade_soft_roi as soft_roi

N_LANDMARKS = base.N_LANDMARKS
LANDMARK_NAMES = base.LANDMARK_NAMES
PATCH_POINTS_DEFAULT = base.PATCH_POINTS
device = base.device

# Global determinism flags — ensures reproducible eval across runs.
# Must be set before any CUDA kernel is launched.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)  # older PyTorch without warn_only

# Training defaults (faster than v3 two-kfold pipeline)
BATCH_SIZE = 4
EPOCHS_TOTAL = 320

LR_TOTAL = 2e-4

W_COARSE = 0.5
W_STAGE2 = 1.0
W_REFINED = 0.5
COND_DIM = 128
COND_DROPOUT = 0.1

# Stage2 adaptive radius lower-bound floor (same policy as adaptive_floor script)
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


def reset_module_parameters(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def estimate_stage2_patch_params(coarse, gt, tag=""):
    """Estimate per-landmark Stage2 patch radius/jitter with protective floors.

    - jitter_k = max(0.05, p80(err_k))
    - radius_k = clamp(max(0.25, 1.2*p95(err_k)), floor_k, 0.80)
    """
    err = np.linalg.norm(coarse - gt, axis=2)  # (N, L)
    p80 = np.percentile(err, 80, axis=0)
    p95 = np.percentile(err, 95, axis=0)

    jitter = np.maximum(base.CENTER_JITTER, p80).astype(np.float32)
    radii = []
    for k, lm in enumerate(LANDMARK_NAMES):
        base_r = max(RADIUS_MIN_DEFAULT, float(p95[k]) * 1.2)
        floor_r = RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)
        radii.append(float(min(max(base_r, floor_r), RADIUS_MAX)))

    print(f"\n[Stage2 adaptive patch params] {tag}".rstrip())
    print(f"  {'landmark':<12} {'p80(err)':>9} {'p95(err)':>9} {'jitter':>9} {'floor':>9} {'radius':>9}")
    print("  " + "-" * 64)
    for k, lm in enumerate(LANDMARK_NAMES):
        floor_r = RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)
        print(f"  {lm:<12} {p80[k]:>9.4f} {p95[k]:>9.4f} {jitter[k]:>9.4f} {floor_r:>9.4f} {radii[k]:>9.4f}")
    return radii, jitter.tolist()


def floor_radii_by_landmark() -> list[float]:
    return [float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT)) for lm in LANDMARK_NAMES]


def select_stage2_patch_params(model, Xtr, Ytr, loader, args, tag: str):
    """Return (radii, jitter, warmup_info) for the requested radius policy."""
    if args.radius_policy == "fixed_floor":
        radii = floor_radii_by_landmark()
        jitter = [float(args.jitter_end)] * N_LANDMARKS
        print("\n[Stage2 patch params] fixed floor radii (Option A)")
        print(f"  {'landmark':<12} {'jitter':>9} {'radius':>9}")
        print("  " + "-" * 36)
        for k, lm in enumerate(LANDMARK_NAMES):
            print(f"  {lm:<12} {jitter[k]:>9.4f} {radii[k]:>9.4f}")
        return radii, jitter, None

    if args.radius_policy == "adaptive_after_warmup":
        warmup_info = {
            "epochs": int(args.adaptive_warmup_epochs),
            "lr": float(args.lr_warmup),
            "policy": "train Stage1 warmup, then estimate p80/p95 radius from training split",
        }
        train_phase_warmup(model, loader, int(args.adaptive_warmup_epochs), float(args.lr_warmup))
        coarse_tr, _ = predict_unified(model, Xtr, batch_size=8)
        radii, jitter = estimate_stage2_patch_params(coarse_tr, Ytr, tag=tag)
        return radii, jitter, warmup_info

    raise ValueError(f"Unknown radius_policy: {args.radius_policy}")


class Stage1WithGlobalFeature(nn.Module):
    """
    Wrap Stage1 and expose:
      - coarse regression output
      - global feature right before stage1.fc1
    """
    def __init__(self, stage1_model: nn.Module):
        super().__init__()
        self.stage1 = stage1_model
        self._cached_feat = None

        if not hasattr(self.stage1, "fc1"):
            raise RuntimeError("Stage1 model has no fc1 layer for feature hook.")
        self._hook_handle = self.stage1.fc1.register_forward_pre_hook(self._capture_fc1_input)

    def _capture_fc1_input(self, _module, inputs):
        if len(inputs) == 0:
            self._cached_feat = None
            return
        self._cached_feat = inputs[0]

    def forward(self, x: torch.Tensor):
        self._cached_feat = None
        pred = self.stage1(x)
        if self._cached_feat is None:
            raise RuntimeError("Failed to capture Stage1 global feature from fc1 pre-hook.")
        return pred, self._cached_feat


class PTCoarseWithGlobalFeature(nn.Module):
    """
    Stage1 replacement: PointTransformerEncoder + CoarseHead.
    Returns:
      - coarse_flat: (B, L*3)
      - global_feat: (B, D)
    """

    def __init__(self):
        super().__init__()
        self.pt_encoder = soft_roi.PointTransformerEncoder(
            dims=(64, 128, 256),
            n_pts=(2048, 512, 128),
            k=16,
            pre_fps_n=4096,
        )
        self.coarse_head = soft_roi.CoarseHead(
            in_dim=self.pt_encoder.out_dim,
            n_landmarks=N_LANDMARKS,
            hidden_dim=256,
            dropout=0.3,
        )
        self.global_dim = int(self.pt_encoder.out_dim)

    def forward(self, x: torch.Tensor):
        global_feat = self.pt_encoder(x)
        coarse = self.coarse_head(global_feat)
        return coarse.view(coarse.shape[0], -1), global_feat


class JointHeatmapS2Conditioned(nn.Module):
    """
    Stage2 heatmap refiner with optional Stage1 global-feature conditioning.
    """
    def __init__(
        self,
        n_landmarks=9,
        dropout=0.3,
        sa1_radii=None,
        stage1_feat_dim=1024,
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
        all_patches = torch.cat(patches_list, dim=0)  # (B*L,3,P)
        xyz_t = all_patches.transpose(1, 2).contiguous()  # (B*L,P,3)
        l1_xyz, l1_f = self.sa1(xyz_t, None)  # (B*L,P,3), (B*L,320,P)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f)  # (B*L,128,3), (B*L,640,128)
        g = self.global_mlp(l2_f).max(dim=2)[0]  # (B*L,128)

        cond = None
        if self.use_conditioning:
            cond_in = stage1_global_feat.detach() if detach_cond else stage1_global_feat
            cond = self.cond_proj(cond_in)  # (B,cond_dim)

        preds, scores_list, l1_xyz_list = [], [], []
        for k in range(self.n_landmarks):
            lf_k = l1_f[k * B : (k + 1) * B]  # (B,320,P)
            g_k = g[k * B : (k + 1) * B]  # (B,128)
            xyz_k = l1_xyz[k * B : (k + 1) * B]  # (B,P,3)
            patch_points = xyz_k.shape[1]

            g_e = g_k.unsqueeze(2).expand(-1, -1, patch_points)
            if cond is not None:
                c_e = cond.unsqueeze(2).expand(-1, -1, patch_points)
                feat = torch.cat([lf_k, g_e, c_e], dim=1)
            else:
                feat = torch.cat([lf_k, g_e], dim=1)

            scores = self.heads[k](feat).squeeze(1)  # (B,P)
            weights = torch.softmax(scores, dim=1)
            pred = (weights.unsqueeze(-1) * xyz_k).sum(dim=1)  # (B,3)
            preds.append(pred)
            scores_list.append(scores)
            l1_xyz_list.append(xyz_k)
        return preds, scores_list, l1_xyz_list


class UnifiedCoarseFineNet(nn.Module):
    def __init__(
        self,
        radii,
        sigma,
        patch_points=PATCH_POINTS_DEFAULT,
        s2_dropout=base.DROPOUT_S2,
        cond_dim=COND_DIM,
        cond_dropout=COND_DROPOUT,
        use_conditioning=True,
    ):
        super().__init__()
        self.stage1 = PTCoarseWithGlobalFeature()
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

    def _sample_patch_indices(self, dists: torch.Tensor, radius: float) -> torch.Tensor:
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

        # eval mode: sort by distance for deterministic selection
        order = torch.argsort(dists[inside])
        return inside[order[:self.patch_points]]

    def _prepare_centers(
        self,
        coarse_centers: torch.Tensor,
        gt_centers: Optional[torch.Tensor],
        mix_prob: float,
        jitter_std,
    ) -> torch.Tensor:
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

    def _crop_patches(
        self,
        x: torch.Tensor,
        centers: torch.Tensor,
        gt_centers: Optional[torch.Tensor],
    ):
        # x: (B,3,N), centers: (B,L,3)
        pc = x.transpose(1, 2).contiguous()  # (B,N,3)
        batch_size = pc.shape[0]

        patches_per_landmark = [[] for _ in range(self.n_landmarks)]
        residual_targets = []

        for b in range(batch_size):
            res_b = []
            pc_b = pc[b]  # (N,3)
            for k in range(self.n_landmarks):
                ctr = centers[b, k]  # (3,)
                dists = torch.norm(pc_b - ctr.unsqueeze(0), dim=1)
                idx = self._sample_patch_indices(dists, float(self.radii[k].item()))
                patch = (pc_b[idx] - ctr.unsqueeze(0)).transpose(0, 1).contiguous()  # (3,P)
                patches_per_landmark[k].append(patch)

                if gt_centers is not None:
                    res_b.append(gt_centers[b, k] - ctr)

            if gt_centers is not None:
                residual_targets.append(torch.stack(res_b, dim=0))  # (L,3)

        patches_list = [torch.stack(patches_per_landmark[k], dim=0) for k in range(self.n_landmarks)]
        residuals = torch.stack(residual_targets, dim=0) if gt_centers is not None else None
        return patches_list, residuals

    def forward(
        self,
        x: torch.Tensor,
        gt_centers: Optional[torch.Tensor] = None,
        mix_prob: float = 0.0,
        jitter_std: float = 0.0,
        detach_coarse_for_crop: bool = False,
        detach_stage1_feat_for_s2: bool = False,
    ):
        coarse_flat, stage1_global_feat = self.stage1(x)
        coarse = coarse_flat.view(-1, self.n_landmarks, 3)  # (B,L,3)
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
        pred_residual = torch.stack(preds_list, dim=1)  # (B,L,3)
        refined = centers + pred_residual  # (B,L,3)

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


def make_loader(X, Y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def train_single_phase_curriculum(
    model,
    loader,
    epochs,
    lr,
    sigma,
    mix_prob_start,
    mix_prob_end,
    radius_scale_start,
    radius_scale_end,
    w_coarse_start,
    w_coarse_end,
    w_stage2_start,
    w_stage2_end,
    w_refined_start,
    w_refined_end,
    jitter_start,
    jitter_end,
    Xval=None,
    Yval=None,
    Sval=None,
    val_batch_size=8,
    val_every=1,
):
    print(
        f"\n[SinglePhase-Curriculum] epochs={epochs}, lr={lr}, sigma={sigma}"
    )
    set_requires_grad(model.stage1, True)
    set_requires_grad(model.stage2, True)

    base_radii = model.radii.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-6)

    best_total = float("inf")
    best_epoch = 0
    best_val_mm = float("inf")
    best_epoch_val = 0
    best_state_by_val = None

    for ep in range(1, epochs + 1):
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

        for pc, gt in loader:
            pc_a, gt_a = base.augment_batch(pc, gt, device)
            opt.zero_grad()

            out = model(
                pc_a,
                gt_centers=gt_a,
                mix_prob=mix_prob,
                jitter_std=jitter_std,
                detach_coarse_for_crop=False,
                detach_stage1_feat_for_s2=False,
            )
            c_loss, s2_loss, r_loss = compute_losses(out, gt_a, sigma=sigma)
            total = w_coarse * c_loss + w_stage2 * s2_loss + w_refined * r_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            acc_total += total.item()
            acc_c += c_loss.item()
            acc_s2 += s2_loss.item()
            acc_r += r_loss.item()

        sched.step()

        denom = max(1, len(loader))
        tr_total = acc_total / denom
        if tr_total < best_total:
            best_total = tr_total
            best_epoch = ep

        cur_val_mm = None
        if Xval is not None and Yval is not None and Sval is not None and (ep % max(1, int(val_every)) == 0):
            cur_val_mm = val_softargmax_mm(model, Xval, Yval, Sval, batch_size=val_batch_size)
            if cur_val_mm < best_val_mm:
                best_val_mm = cur_val_mm
                best_epoch_val = ep
                best_state_by_val = copy.deepcopy(model.state_dict())

        if ep % 20 == 0 or ep == epochs:
            val_part = f" val={cur_val_mm:.3f}" if cur_val_mm is not None else ""
            print(
                f"  ep{ep:3d} lr={opt.param_groups[0]['lr']:.6f} total={tr_total:.5f} "
                f"coarse={acc_c/denom:.5f} s2={acc_s2/denom:.5f} refined={acc_r/denom:.5f} "
                f"mix={mix_prob:.3f} radius_scale={radius_scale:.3f} jitter={jitter_std:.4f} "
                f"w=({w_coarse:.2f},{w_stage2:.2f},{w_refined:.2f}){val_part}"
            )

    with torch.no_grad():
        model.radii.copy_(base_radii)

    if best_state_by_val is not None:
        model.load_state_dict(best_state_by_val)

    return {
        "best_epoch_by_train_total": int(best_epoch),
        "best_train_total": float(best_total),
        "best_epoch_by_val_softargmax": int(best_epoch_val) if best_epoch_val > 0 else None,
        "best_val_softargmax_mm": float(best_val_mm) if best_epoch_val > 0 else None,
        "restored_best_val_weights": bool(best_state_by_val is not None),
        "epochs": int(epochs),
        "lr": float(lr),
    }


@torch.no_grad()
def predict_unified(model, X, batch_size=8):
    model.eval()
    coarse_all, refined_all = [], []
    with torch.no_grad():
      for i in range(0, len(X), batch_size):
        pc = torch.from_numpy(X[i : i + batch_size]).float().to(device)
        out = model(
            pc,
            gt_centers=None,
            mix_prob=0.0,
            jitter_std=0.0,
            detach_coarse_for_crop=False,
            detach_stage1_feat_for_s2=False,
        )
        coarse_all.append(out["coarse"].cpu().numpy())
        refined_all.append(out["refined"].cpu().numpy())
    return np.concatenate(coarse_all, axis=0), np.concatenate(refined_all, axis=0)


@torch.no_grad()
def val_softargmax_mm(model, Xv, Yv, Sv, batch_size=8):
    _, refined = predict_unified(model, Xv, batch_size=batch_size)
    errs = np.linalg.norm(refined - Yv, axis=2) * Sv[:, None]
    return float(errs.mean())


def train_phase_warmup(model, loader, epochs, lr):
    print(f"\n[Warmup] Stage1 only, epochs={epochs}, lr={lr}")
    set_requires_grad(model.stage1, True)
    set_requires_grad(model.stage2, False)

    opt = torch.optim.Adam(model.stage1.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)

    for ep in range(1, epochs + 1):
        model.stage1.train()
        model.stage2.eval()
        acc = 0.0

        for pc, gt in loader:
            pc_a, gt_a = base.augment_batch(pc, gt, device)
            opt.zero_grad()
            coarse_flat, _ = model.stage1(pc_a)
            coarse = coarse_flat.view(-1, N_LANDMARKS, 3)
            loss = F.smooth_l1_loss(coarse, gt_a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
            opt.step()
            acc += loss.item()

        sched.step()
        if ep % 20 == 0 or ep == epochs:
            print(f"  ep{ep:3d}  coarse_loss={acc/max(1, len(loader)):.5f}")


def train_stage1_kfold_select_epoch(Xtr, Ytr, Str, k_splits, epochs, lr, random_state=42):
    print(f"\n[Stage1 KFold] K={k_splits}  epochs={epochs}")
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    fold_best_epochs = []

    for fold, (tr_i, val_i) in enumerate(kf.split(Xtr), 1):
        Xf, Yf = Xtr[tr_i], Ytr[tr_i]
        Xv, Yv, Sv = Xtr[val_i], Ytr[val_i], Str[val_i]

        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xf).float(), torch.from_numpy(Yf).float()),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        stage1_model = PTCoarseWithGlobalFeature().to(device)
        opt = torch.optim.Adam(stage1_model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)
        Xv_t = torch.from_numpy(Xv).float().to(device)

        best_val = float("inf")
        best_ep = epochs

        for ep in range(1, epochs + 1):
            stage1_model.train()
            for pc, gt in loader:
                pc_a, gt_a = base.augment_batch(pc, gt, device)
                opt.zero_grad()
                coarse_flat, _ = stage1_model(pc_a)
                pred = coarse_flat.view(-1, N_LANDMARKS, 3)
                loss = F.smooth_l1_loss(pred, gt_a)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(stage1_model.parameters(), 1.0)
                opt.step()
            sched.step()

            if ep % 20 == 0 or ep == epochs:
                stage1_model.eval()
                with torch.no_grad():
                    coarse_flat_v, _ = stage1_model(Xv_t)
                    pv = coarse_flat_v.cpu().numpy().reshape(-1, N_LANDMARKS, 3)
                errs = np.linalg.norm(pv - Yv, axis=2) * Sv[:, None]
                val_mm = float(errs.mean())
                marker = " <-- best" if val_mm < best_val else ""
                print(f"  Fold{fold} ep{ep:3d}  val={val_mm:.3f}mm{marker}")
                if val_mm < best_val:
                    best_val = val_mm
                    best_ep = ep

        fold_best_epochs.append(best_ep)

    best_epoch = int(np.median(fold_best_epochs))
    print(f"\n[Stage1] best epoch (median)={best_epoch}  per-fold={fold_best_epochs}")
    return best_epoch, fold_best_epochs


def retrain_stage1_full(model, Xtr, Ytr, epochs, lr, batch_size):
    print(f"\n[Stage1] full retrain {epochs} epochs on {len(Xtr)} samples")
    set_requires_grad(model.stage1, True)
    set_requires_grad(model.stage2, False)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.stage1.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)

    for ep in range(1, epochs + 1):
        model.stage1.train()
        model.stage2.eval()
        acc = 0.0

        for pc, gt in loader:
            pc_a, gt_a = base.augment_batch(pc, gt, device)
            opt.zero_grad()
            coarse_flat, _ = model.stage1(pc_a)
            coarse = coarse_flat.view(-1, N_LANDMARKS, 3)
            loss = F.smooth_l1_loss(coarse, gt_a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
            opt.step()
            acc += loss.item()

        sched.step()
        if ep % 20 == 0 or ep == epochs:
            print(f"  ep{ep:3d}  coarse_loss={acc/max(1, len(loader)):.5f}")


def train_phase_stage2(model, loader, epochs, lr, sigma, jitter_std, mix_prob):
    jitter_mean = float(jitter_std) if np.isscalar(jitter_std) else float(np.mean(np.asarray(jitter_std, dtype=np.float32)))
    print(
        f"\n[Phase2] Stage2 only, epochs={epochs}, lr={lr}, "
        f"jitter(mean)={jitter_mean:.4f}, mix_prob={mix_prob:.2f}"
    )
    set_requires_grad(model.stage1, False)
    set_requires_grad(model.stage2, True)

    opt = torch.optim.Adam(model.stage2.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=80, gamma=0.5)

    for ep in range(1, epochs + 1):
        model.stage1.eval()
        model.stage2.train()
        acc = 0.0

        for pc, gt in loader:
            pc_a, gt_a = base.augment_batch(pc, gt, device)
            opt.zero_grad()
            out = model(
                pc_a,
                gt_centers=gt_a,
                mix_prob=mix_prob,
                jitter_std=jitter_std,
                detach_coarse_for_crop=True,
                detach_stage1_feat_for_s2=True,
            )
            s2_loss = base.joint_loss(
                out["preds_list"],
                out["scores_list"],
                out["l1_xyz_list"],
                out["residual_targets"],
                sigma=sigma,
            )
            s2_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), 1.0)
            opt.step()
            acc += s2_loss.item()

        sched.step()
        if ep % 20 == 0 or ep == epochs:
            print(f"  ep{ep:3d}  stage2_loss={acc/max(1, len(loader)):.5f}")


def train_stage2_kfold_select_epoch(
    model,
    Xtr,
    Ytr,
    Str,
    k_splits,
    epochs,
    lr,
    sigma,
    jitter_std,
    mix_prob,
    random_state=42,
):
    print(f"\n[Stage2 KFold] K={k_splits}  epochs={epochs}  sigma={sigma}")
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    fold_best_epochs = []

    for fold, (tr_i, val_i) in enumerate(kf.split(Xtr), 1):
        Xf, Yf = Xtr[tr_i], Ytr[tr_i]
        Xv, Yv, Sv = Xtr[val_i], Ytr[val_i], Str[val_i]

        fold_model = copy.deepcopy(model).to(device)
        set_requires_grad(fold_model.stage1, False)
        set_requires_grad(fold_model.stage2, True)
        fold_model.stage1.eval()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xf).float(), torch.from_numpy(Yf).float()),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        opt = torch.optim.Adam(fold_model.stage2.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=80, gamma=0.5)

        best_val = float("inf")
        best_ep = epochs

        for ep in range(1, epochs + 1):
            fold_model.stage1.eval()
            fold_model.stage2.train()
            for pc, gt in loader:
                pc_a, gt_a = base.augment_batch(pc, gt, device)
                opt.zero_grad()
                out = fold_model(
                    pc_a,
                    gt_centers=gt_a,
                    mix_prob=mix_prob,
                    jitter_std=jitter_std,
                    detach_coarse_for_crop=True,
                    detach_stage1_feat_for_s2=True,
                )
                s2_loss = base.joint_loss(
                    out["preds_list"],
                    out["scores_list"],
                    out["l1_xyz_list"],
                    out["residual_targets"],
                    sigma=sigma,
                )
                s2_loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.stage2.parameters(), 1.0)
                opt.step()
            sched.step()

            if ep % 30 == 0 or ep == epochs:
                fold_model.eval()
                preds = []
                with torch.no_grad():
                    for i in range(0, len(Xv), 8):
                        pc = torch.from_numpy(Xv[i:i + 8]).float().to(device)
                        out = fold_model(
                            pc,
                            gt_centers=None,
                            mix_prob=0.0,
                            jitter_std=0.0,
                            detach_coarse_for_crop=False,
                            detach_stage1_feat_for_s2=False,
                        )
                        preds.append(out["refined"].cpu().numpy())
                refined_v = np.concatenate(preds, axis=0)
                errs = np.linalg.norm(refined_v - Yv, axis=2) * Sv[:, None]
                val_mm = float(errs.mean())
                marker = " <-- best" if val_mm < best_val else ""
                print(f"  Fold{fold} ep{ep:3d}  val={val_mm:.3f}mm{marker}")
                if val_mm < best_val:
                    best_val = val_mm
                    best_ep = ep

        fold_best_epochs.append(best_ep)

    best_epoch = int(np.median(fold_best_epochs))
    print(f"\n[Stage2] best epoch (median)={best_epoch}  per-fold={fold_best_epochs}")
    return best_epoch, fold_best_epochs


def train_phase_joint(
    model,
    loader,
    epochs,
    lr,
    sigma,
    jitter_std,
    mix_prob,
    w_coarse,
    w_stage2,
    w_refined,
):
    jitter_mean = float(jitter_std) if np.isscalar(jitter_std) else float(np.mean(np.asarray(jitter_std, dtype=np.float32)))
    print(
        f"\n[Phase3] Joint finetune, epochs={epochs}, lr={lr}, "
        f"weights=({w_coarse:.2f},{w_stage2:.2f},{w_refined:.2f}), jitter(mean)={jitter_mean:.4f}"
    )
    set_requires_grad(model.stage1, True)
    set_requires_grad(model.stage2, True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-6)

    for ep in range(1, epochs + 1):
        model.train()
        acc_total, acc_c, acc_s2, acc_r = 0.0, 0.0, 0.0, 0.0

        for pc, gt in loader:
            pc_a, gt_a = base.augment_batch(pc, gt, device)
            opt.zero_grad()

            out = model(
                pc_a,
                gt_centers=gt_a,
                mix_prob=mix_prob,
                jitter_std=jitter_std,
                detach_coarse_for_crop=False,
                detach_stage1_feat_for_s2=False,
            )
            c_loss, s2_loss, r_loss = compute_losses(out, gt_a, sigma=sigma)
            total = w_coarse * c_loss + w_stage2 * s2_loss + w_refined * r_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            acc_total += total.item()
            acc_c += c_loss.item()
            acc_s2 += s2_loss.item()
            acc_r += r_loss.item()

        sched.step()
        if ep % 20 == 0 or ep == epochs:
            denom = max(1, len(loader))
            print(
                f"  ep{ep:3d}  total={acc_total/denom:.5f}  "
                f"coarse={acc_c/denom:.5f}  s2={acc_s2/denom:.5f}  refined={acc_r/denom:.5f}"
            )


def evaluate_unified(
    model,
    Xte,
    Yte,
    Ste,
    full_clouds_te,
    triangles_te,
    tag,
):
    coarse_all, refined_all = predict_unified(model, Xte, batch_size=8)

    per_lm_refined = [[] for _ in range(N_LANDMARKS)]
    per_lm_snap_pt = [[] for _ in range(N_LANDMARKS)]
    per_lm_snap_sf = [[] for _ in range(N_LANDMARKS)]
    per_lm_coarse = [[] for _ in range(N_LANDMARKS)]

    for i in range(len(Xte)):
        pc_full = full_clouds_te[i]
        tri = triangles_te[i] if triangles_te is not None else None
        sc = Ste[i]

        for k in range(N_LANDMARKS):
            gt = Yte[i, k]
            coarse = coarse_all[i, k]
            refined = refined_all[i, k]
            snap_pt = base.snap_to_nearest(refined, pc_full)
            snap_sf = base.snap_to_mesh(refined, pc_full, tri)

            per_lm_coarse[k].append(np.linalg.norm(coarse - gt) * sc)
            per_lm_refined[k].append(np.linalg.norm(refined - gt) * sc)
            per_lm_snap_pt[k].append(np.linalg.norm(snap_pt - gt) * sc)
            per_lm_snap_sf[k].append(np.linalg.norm(snap_sf - gt) * sc)

    print(f"\n{'='*70}")
    print(f"  Eval results  [{tag}]  test N={len(Xte)}")
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
            "per_sample": {
                "coarse": [float(x) for x in per_lm_coarse[k]],
                "soft_argmax": [float(x) for x in per_lm_refined[k]],
                "snap_nearest": [float(x) for x in per_lm_snap_pt[k]],
                "snap_mesh": [float(x) for x in per_lm_snap_sf[k]],
            },
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
        "per_sample_errors": {
            # shape: {landmark: [err_sample0, err_sample1, ...]}  (snap_mesh only)
            lm: [float(x) for x in per_lm_snap_sf[k]]
            for k, lm in enumerate(LANDMARK_NAMES)
        },
    }


def run_one_seed(seed, args, X, Y, scales, names, full_clouds, triangles):
    tag = f"seed{seed}_sigma{args.sigma}"
    if args.radius_policy != "fixed_floor":
        tag = f"{tag}_{args.radius_policy}"
    print(f"\n{'#'*70}")
    print(f"  SEED={seed}  SIGMA={args.sigma}")
    print(f"  RADIUS_POLICY={args.radius_policy}")
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
    print(f"  holdout split: train={len(Xtr)} test={len(Xte)} (80/20)")

    n_inner = int(max(2, min(args.kfold_splits, len(Xtr))))
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=args.kfold_random_state)
    inner_folds = list(kf.split(Xtr))
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
    for i, (tr_i, val_i) in enumerate(inner_folds, start=1):
        inner_no = i + fold_offset
        inner_tag = f"{tag}_inner{inner_no}"
        Xtr_sub, Ytr_sub = Xtr[tr_i], Ytr[tr_i]
        Xva, Yva, Sva = Xtr[val_i], Ytr[val_i], Str[val_i]
        print(f"\n  [Inner {inner_no}/{n_inner}] train={len(Xtr_sub)} val={len(Xva)}")

        model_path = os.path.join(base.MODELS_DIR, f"joint_hybrid_pt_patch_floor_{inner_tag}.pth")
        init_radii = floor_radii_by_landmark()
        model = UnifiedCoarseFineNet(
            radii=init_radii,
            sigma=args.sigma,
            patch_points=args.patch_points,
            s2_dropout=base.DROPOUT_S2,
            cond_dim=args.cond_dim,
            cond_dropout=args.cond_dropout,
            use_conditioning=(not args.disable_conditioning),
        ).to(device)

        train_curriculum_info = None
        radius_policy_info = None
        radii_s2 = floor_radii_by_landmark()
        jitter_s2 = [float(args.jitter_end)] * N_LANDMARKS

        if args.eval_only:
            if not os.path.exists(model_path):
                print(f"[skip] model not found: {model_path}")
                continue
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            train_curriculum_info = ckpt.get("train_curriculum", None)
        else:
            train_loader = make_loader(Xtr_sub, Ytr_sub, batch_size=args.batch_size, shuffle=True)

            radii_s2, jitter_s2, radius_policy_info = select_stage2_patch_params(
                model,
                Xtr_sub,
                Ytr_sub,
                train_loader,
                args,
                tag=f"train {inner_tag}",
            )
            with torch.no_grad():
                model.radii.copy_(torch.tensor(radii_s2, dtype=model.radii.dtype, device=model.radii.device))

            coarse_tr, _ = predict_unified(model, Xtr_sub, batch_size=8)
            base.print_coverage(Xtr_sub, Ytr_sub, coarse_tr, radii_s2, tag=f"train {inner_tag}")

            train_curriculum_info = train_single_phase_curriculum(
                model,
                train_loader,
                epochs=args.epochs_total,
                lr=args.lr,
                sigma=args.sigma,
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
                Xval=Xva,
                Yval=Yva,
                Sval=Sva,
                val_batch_size=8,
                val_every=1,
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
                        "radius_policy": args.radius_policy,
                        "radii": [float(x) for x in radii_s2],
                        "jitter": [float(x) for x in jitter_s2],
                        "adaptive_warmup": radius_policy_info,
                        "radius_floor": {
                            lm: float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT))
                            for lm in LANDMARK_NAMES
                        },
                    },
                },
                model_path,
            )
            coarse_tr_post, _ = predict_unified(model, Xtr_sub, batch_size=8)
            base.print_coverage(Xtr_sub, Ytr_sub, coarse_tr_post, radii_s2, tag=f"train {inner_tag} (post)")
            print(f"[save] Unified model -> {model_path}")

        val_mm = val_softargmax_mm(model, Xva, Yva, Sva, batch_size=8)
        print(f"  [Inner {inner_no}] val_softargmax_mean={val_mm:.3f} mm")
        record = {
            "inner_fold": int(inner_no),
            "val_softargmax_mean_mm": float(val_mm),
            "model_path": model_path,
            "radius_policy": args.radius_policy,
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
            if r.get("train_curriculum") is not None and "best_epoch_by_train_total" in r["train_curriculum"]
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
        "radius_policy": args.radius_policy,
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

    inner_summary_path = os.path.join(base.RESULTS_DIR, f"joint_hybrid_pt_patch_floor_innercv_summary_seed{seed}.json")
    with open(inner_summary_path, "w", encoding="utf-8") as f:
        json.dump(inner_cv_summary, f, indent=2, ensure_ascii=False)
    print(f"[save] inner-cv summary -> {inner_summary_path}")

    final_tag = tag
    final_model_path = os.path.join(base.MODELS_DIR, f"joint_hybrid_pt_patch_floor_{final_tag}.pth")
    init_radii = floor_radii_by_landmark()
    final_model = UnifiedCoarseFineNet(
        radii=init_radii,
        sigma=args.sigma,
        patch_points=args.patch_points,
        s2_dropout=base.DROPOUT_S2,
        cond_dim=args.cond_dim,
        cond_dropout=args.cond_dropout,
        use_conditioning=(not args.disable_conditioning),
    ).to(device)

    final_curriculum_info = None
    final_radius_policy_info = None
    radii_s2 = floor_radii_by_landmark()
    jitter_s2 = [float(args.jitter_end)] * N_LANDMARKS

    if args.eval_only:
        if not os.path.exists(final_model_path):
            print(f"[skip] final model not found: {final_model_path}")
            return None
        ckpt = torch.load(final_model_path, map_location=device, weights_only=False)
        final_model.load_state_dict(ckpt["model"])
        final_curriculum_info = ckpt.get("train_curriculum", None)
    else:
        train_loader_full = make_loader(Xtr, Ytr, batch_size=args.batch_size, shuffle=True)

        radii_s2, jitter_s2, final_radius_policy_info = select_stage2_patch_params(
            final_model,
            Xtr,
            Ytr,
            train_loader_full,
            args,
            tag=f"full-train {final_tag}",
        )

        with torch.no_grad():
            final_model.radii.copy_(torch.tensor(radii_s2, dtype=final_model.radii.dtype, device=final_model.radii.device))

        coarse_full_pre, _ = predict_unified(final_model, Xtr, batch_size=8)
        base.print_coverage(Xtr, Ytr, coarse_full_pre, radii_s2, tag=f"full-train {final_tag}")

        final_curriculum_info = train_single_phase_curriculum(
            final_model,
            train_loader_full,
            epochs=selected_epochs,
            lr=args.lr,
            sigma=args.sigma,
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
                    "radius_policy": args.radius_policy,
                    "radii": [float(x) for x in radii_s2],
                    "jitter": [float(x) for x in jitter_s2],
                    "adaptive_warmup": final_radius_policy_info,
                    "radius_floor": {
                        lm: float(RADIUS_MIN_BY_LM.get(lm, RADIUS_MIN_DEFAULT))
                        for lm in LANDMARK_NAMES
                    },
                },
            },
            final_model_path,
        )
        coarse_full_post, _ = predict_unified(final_model, Xtr, batch_size=8)
        base.print_coverage(Xtr, Ytr, coarse_full_post, radii_s2, tag=f"full-train {final_tag} (post)")
        print(f"[save] final model -> {final_model_path}")

    final_results = evaluate_unified(
        model=final_model,
        Xte=Xte,
        Yte=Yte,
        Ste=Ste,
        full_clouds_te=full_clouds_te,
        triangles_te=triangles_te,
        tag=final_tag,
    )
    final_results["seed"] = seed
    final_results["sigma"] = args.sigma
    final_results["radius_policy"] = args.radius_policy
    final_results["selected_epochs_from_inner_cv"] = int(selected_epochs)
    final_results["inner_cv_summary_path"] = inner_summary_path
    if final_curriculum_info is not None:
        final_results["train_curriculum"] = final_curriculum_info

    out_path = os.path.join(base.RESULTS_DIR, f"joint_hybrid_pt_patch_floor_eval_{final_tag}.json")
    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"[save] final eval results -> {out_path}")

    seed_summary = {
        "seed": int(seed),
        "sigma": float(args.sigma),
        "radius_policy": args.radius_policy,
        "split": "80_20_holdout",
        "train_kfolds": int(len(inner_cv_records)),
        "selected_epochs": int(selected_epochs),
        "inner_cv_summary_path": inner_summary_path,
        "final_model_path": final_model_path,
        "final_eval_path": out_path,
        "overall": final_results.get("overall", {}),
        "final_overall": final_results.get("overall", {}),
    }
    summary_path = os.path.join(base.RESULTS_DIR, f"joint_hybrid_pt_patch_floor_holdout_summary_seed{seed}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(seed_summary, f, indent=2, ensure_ascii=False)
    print(f"[save] seed summary -> {summary_path}")
    return seed_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--backbone", default="pn2", choices=["pn2", "pt"], help="Stage1 backbone")
    ap.add_argument("--seed", type=int, default=None, help="single seed; default runs all")
    ap.add_argument("--sigma", type=float, default=base.SIGMA_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--patch-points", type=int, default=PATCH_POINTS_DEFAULT)

    ap.add_argument("--epochs-warmup", type=int, default=120)
    ap.add_argument("--kfold-splits", type=int, default=5)
    ap.add_argument("--kfold-epochs", type=int, default=300)
    ap.add_argument("--kfold-random-state", type=int, default=42)
    ap.add_argument("--s2-kfold-splits", type=int, default=5)
    ap.add_argument("--s2-kfold-epochs", type=int, default=300)
    ap.add_argument("--s2-kfold-random-state", type=int, default=42)
    ap.add_argument("--epochs-stage2", type=int, default=120)
    ap.add_argument("--epochs-joint", type=int, default=80)
    ap.add_argument("--epochs-total", type=int, default=EPOCHS_TOTAL)

    ap.add_argument("--lr-warmup", type=float, default=3e-4)
    ap.add_argument("--lr-stage2", type=float, default=1e-3)
    ap.add_argument("--lr-joint", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=LR_TOTAL)

    ap.add_argument("--mix-prob-stage2", type=float, default=0.5)
    ap.add_argument("--mix-prob-joint", type=float, default=0.2)
    ap.add_argument("--mix-prob-start", type=float, default=0.7)
    ap.add_argument("--mix-prob-end", type=float, default=0.2)
    ap.add_argument("--radius-scale-start", type=float, default=1.2)
    ap.add_argument("--radius-scale-end", type=float, default=1.0)
    ap.add_argument(
        "--radius-policy",
        choices=["fixed_floor", "adaptive_after_warmup"],
        default="fixed_floor",
        help="Stage2 patch radius policy: existing fixed floor values, or adaptive p95*1.2 after Stage1 warmup",
    )
    ap.add_argument(
        "--adaptive-warmup-epochs",
        type=int,
        default=120,
        help="Stage1 warmup epochs used only when --radius-policy adaptive_after_warmup",
    )
    ap.add_argument("--jitter-min", type=float, default=base.CENTER_JITTER)
    ap.add_argument("--jitter-start", type=float, default=0.08)
    ap.add_argument("--jitter-end", type=float, default=base.CENTER_JITTER)

    ap.add_argument("--w-coarse", type=float, default=W_COARSE)
    ap.add_argument("--w-stage2", type=float, default=W_STAGE2)
    ap.add_argument("--w-refined", type=float, default=W_REFINED)
    ap.add_argument("--w-coarse-start", type=float, default=1.0)
    ap.add_argument("--w-stage2-start", type=float, default=0.5)
    ap.add_argument("--w-refined-start", type=float, default=0.2)
    ap.add_argument("--cond-dim", type=int, default=COND_DIM)
    ap.add_argument("--cond-dropout", type=float, default=COND_DROPOUT)
    ap.add_argument("--disable-conditioning", action="store_true")
    ap.add_argument("--train-fold", type=int, default=None, help="run only one inner fold (1-based)")

    args = ap.parse_args()

    base._S1_BACKBONE = args.backbone
    print(f"[Config] Stage1 backbone = {base._S1_BACKBONE}")
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
    print(
        f"[Config] radius_policy = {args.radius_policy} "
        f"(adaptive_warmup_epochs={args.adaptive_warmup_epochs})"
    )

    X, Y, scales, names, full_clouds, triangles = base.load_data()
    seed_list = [args.seed] if args.seed is not None else base.SEEDS

    all_results = []
    for seed in seed_list:
        res = run_one_seed(seed, args, X, Y, scales, names, full_clouds, triangles)
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
