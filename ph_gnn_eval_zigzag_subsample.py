#!/usr/bin/env python3
"""
ph_gnn_eval_zigzag_subsample.py  — multi-epoch sweep edition (subsample checkpoints)
Evaluates every model_best_subsample_{epoch}.pt found in MODELS_DIR on the zigzag test set.
Logs to W&B:
  epoch_sweep/pos_mse        – average position MSE per epoch model
  epoch_sweep/vel_mse        – average velocity MSE per epoch model
  epoch_sweep/total_loss     – average total loss per epoch model
  epoch_sweep/R_psd_loss     – dissipation matrix PSD penalty per epoch model
  epoch_sweep/R_norm         – Frobenius norm of R per epoch model
  epoch_sweep/R_eigmin       – smallest eigenvalue of R per epoch model
Plus a summary table and comparison charts (pos_mse, vel_mse, total_loss vs epoch).
"""

import os, sys, math, argparse, json, re
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import pytorch_kinematics as pk
import wandb
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="PH-GNN zigzag evaluation — subsample checkpoints sweep")
parser.add_argument("--urdf",         type=str,
                    default="/home/rllab/msc_student/Vikram/assets/spot.urdf")
parser.add_argument("--dataset_root", type=str,
                    default="/scratch/work/venkatv1/Dataset")
parser.add_argument("--test_subdir",  type=str, default="DatasetTest_zigzag")
parser.add_argument("--models_dir",   type=str,
                    default="/scratch/work/venkatv1/ph_gnn_outputs",
                    help="Directory containing model_best_subsample_1.pt, etc.")
parser.add_argument("--out_dir",      type=str,
                    default="/scratch/work/venkatv1/ph_gnn_outputs")
parser.add_argument("--wandb_project", type=str, default="ph-gnn-phase1")
parser.add_argument("--wandb_run",    type=str,
                    default="zigzag_eval_subsample_epochs")
args, _ = parser.parse_known_args()

DATASET_ROOT = Path(args.dataset_root)
TEST_ROOT    = DATASET_ROOT / args.test_subdir
OUT_DIR      = Path(args.out_dir)
MODELS_DIR   = Path(args.models_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyper-parameters (identical to training) ─────────────────────────────────
ROLLOUT_STEPS    = 5
N_STAR           = 6
R_PARTICLE       = 0.03
R_THRESH         = 2 * R_PARTICLE
R_WALL           = 0.01
PARTICLE_DENSITY = 2000.0
PARTICLE_MASS    = PARTICLE_DENSITY * (4.0 / 3.0) * math.pi * R_PARTICLE**3
DT               = 1.0 / 60.0
USE_AMP          = torch.cuda.is_available()
LAMBDA1          = 1000.0
LAMBDA2          = 0.5
LAMBDA4          = 1e-4
PARTICLE_SET_OFFSET = np.array([0.4, 0.0, 0.09279], dtype=np.float64)

# ─── Robot / FK constants ─────────────────────────────────────────────────────
END_EFFECTOR      = "arm0.link_wr1"
ROBOT_BASE_WORLD  = np.array([-0.55, 1.43801, 0.50328], dtype=np.float64)
DTYPE_FK          = torch.float64
DATASET_JOINT_IDX = [0, 1, 3, 4, 5, 6]
WR1_LIMIT         = 2.88
SCOOP_OFFSET_EE   = np.array(
    [0.167 + 0.09454, -0.077 - 0.06384, -0.103 + 0.07927], dtype=np.float64)
SCOOP_HALF_EXTENTS = np.array(
    [0.08759/2 + 0.008, 0.14759/2 + 0.008, 0.23/2 + 0.008])
SCOOP_HALF      = SCOOP_HALF_EXTENTS + 0.005
R_scoop_local   = Rotation.from_euler(
    "xyz", [-149.515, 61.231, 145.323], degrees=True).as_matrix()

WALL_CENTERS = torch.tensor([
    [0.2962 + 0.10371, 1.5250 + 0.3218, 0.0750 - 0.5401],
    [0.9361 + 0.10371, 1.5108 + 0.3218, 0.1024 - 0.5401],
    [0.4698 + 0.10371, 1.0382 + 0.3218, 0.1024 - 0.5401],
    [0.4698 + 0.10371, 1.9894 + 0.3218, 0.1024 - 0.5401],
], dtype=torch.float32)
WALL_NORMALS = torch.tensor([
    [ 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.0], [ 0.0,-1.0, 0.0],
], dtype=torch.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R_PHASES   = ["flat_start"]
FP_PHASES  = ["move_forward", "turn", "move_forward2"]
ALL_PHASES = R_PHASES + FP_PHASES

# ─── Load FK chain (once, shared across all epoch evaluations) ─────────────────
print(f"Loading kinematic chain from: {args.urdf}")
fk_chain = pk.build_serial_chain_from_urdf(
    open(args.urdf).read(), end_link_name=END_EFFECTOR, root_link_name="body")
fk_chain = fk_chain.to(dtype=DTYPE_FK, device=DEVICE)
N_CHAIN  = len(fk_chain.get_joint_parameter_names())

def fk_ee(arm_pos_np: np.ndarray):
    th_full = np.zeros(N_CHAIN, dtype=np.float64)
    th_full[DATASET_JOINT_IDX] = arm_pos_np
    th_full[6] = np.clip(th_full[6], -WR1_LIMIT, WR1_LIMIT)
    th = torch.tensor(th_full, dtype=DTYPE_FK, device=DEVICE).unsqueeze(0)
    T  = fk_chain.forward_kinematics(th).get_matrix().squeeze(0).cpu().numpy()
    return ROBOT_BASE_WORLD + T[3, :3], T[:3, :3]

def obb_contact_np(particles_world, p_ee_world, R_ee_world):
    centre = p_ee_world + R_ee_world @ SCOOP_OFFSET_EE
    R_obb  = R_ee_world @ R_scoop_local
    local  = (particles_world - centre[np.newaxis]) @ R_obb
    return (np.abs(local[:, 0]) <= SCOOP_HALF[0]) & \
           (np.abs(local[:, 1]) <= SCOOP_HALF[1]) & \
           (np.abs(local[:, 2]) <= SCOOP_HALF[2])

# ─── Graph helpers ─────────────────────────────────────────────────────────────
try:
    from torch_cluster import radius_graph as tc_radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

def radius_graph_cpu(q_np, r_thresh):
    tree  = KDTree(q_np)
    pairs = tree.query_pairs(r_thresh, output_type="ndarray")
    if len(pairs):
        s = torch.from_numpy(pairs[:, 0]).long()
        d = torch.from_numpy(pairs[:, 1]).long()
        return torch.cat([s, d]), torch.cat([d, s])
    return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

def radius_graph_gpu(q_dev, r_thresh):
    ei = tc_radius_graph(q_dev, r=r_thresh, loop=False, max_num_neighbors=64)
    return ei[1], ei[0]

def bfs_hops(adj, seeds, max_hops, N):
    hop_dist = torch.full((N,), -1, dtype=torch.long)
    queue = deque()
    for idx in seeds:
        if hop_dist[idx] < 0:
            hop_dist[idx] = 0
            queue.append(idx)
    while queue:
        node = queue.popleft()
        h = hop_dist[node].item()
        if h >= max_hops:
            continue
        for nb in adj[node]:
            if hop_dist[nb] < 0:
                hop_dist[nb] = h + 1
                queue.append(nb)
    return hop_dist

def build_proximity_graph(q, arm_pos_np=None, r_thresh=R_THRESH,
                           r_wall=R_WALL, n_star=N_STAR):
    N = q.shape[0]
    if HAS_TORCH_CLUSTER and q.device.type == "cuda":
        bb_src, bb_dst = radius_graph_gpu(q, r_thresh)
    else:
        q_np = q.cpu().numpy()
        bb_src, bb_dst = radius_graph_cpu(q_np, r_thresh)
    q_np = q.cpu().numpy()
    if arm_pos_np is not None:
        p_ee, R_ee = fk_ee(arm_pos_np)
        mask_contact = obb_contact_np(q_np, p_ee, R_ee)
        br_src = torch.tensor(np.where(mask_contact)[0], dtype=torch.long)
    else:
        br_src = torch.empty(0, dtype=torch.long)
    bw_src_list = []
    for widx in range(4):
        wc = WALL_CENTERS[widx].numpy()
        wn = WALL_NORMALS[widx].numpy()
        dist = np.abs((q_np - wc) @ wn)
        bw_src_list.append(
            torch.tensor(np.where(dist <= r_wall)[0], dtype=torch.long))
    active = torch.ones(N, dtype=torch.bool)
    return bb_src, bb_dst, br_src, bw_src_list, active

def subsample_to_active(x_t, x_t1, bb_src, bb_dst, br_src, bw_src_list, active_mask):
    active_mask = active_mask.cpu()
    idx_active  = active_mask.nonzero(as_tuple=True)[0]
    if idx_active.numel() == 0:
        return (x_t, x_t1, bb_src.cpu(), bb_dst.cpu(),
                br_src.cpu(), [w.cpu() for w in bw_src_list])
    N = x_t.shape[0]
    remap = torch.full((N,), -1, dtype=torch.long)
    remap[idx_active] = torch.arange(idx_active.numel(), dtype=torch.long)
    x_t_sub  = x_t[idx_active.to(x_t.device)]
    x_t1_sub = x_t1[idx_active.to(x_t1.device)]

    def remap_edges(src, dst):
        sc = src.cpu(); dc = dst.cpu()
        mask = (remap[sc] >= 0) & (remap[dc] >= 0)
        return remap[sc[mask]], remap[dc[mask]]

    if bb_src.numel() > 0:
        new_bb_src, new_bb_dst = remap_edges(bb_src, bb_dst)
    else:
        new_bb_src = new_bb_dst = torch.empty(0, dtype=torch.long)
    new_br_src = remap[br_src.cpu()]
    new_br_src = new_br_src[new_br_src >= 0]
    new_bw_list = []
    for bws in bw_src_list:
        if bws.numel() > 0:
            bw_cpu = bws.cpu(); valid = remap[bw_cpu]
            new_bw_list.append(valid[valid >= 0])
        else:
            new_bw_list.append(torch.empty(0, dtype=torch.long))
    return x_t_sub, x_t1_sub, new_bb_src, new_bb_dst, new_br_src, new_bw_list

# ─── Model definition (identical to training) ─────────────────────────────────
class HamiltonianNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x)
    def grad(self, x):
        with torch.amp.autocast("cuda", enabled=False), torch.enable_grad():
            x_in = x.float().clone().requires_grad_(True)
            H    = self.net(x_in)
            g    = torch.autograd.grad(H.sum(), x_in,
                       create_graph=False, retain_graph=False,
                       allow_unused=False)[0]
        return g

class DissipationMatrix(nn.Module):
    def __init__(self):
        super().__init__()
        self.lower_off  = nn.Parameter(torch.zeros(15))
        self.lower_diag = nn.Parameter(torch.zeros(6))
    def L(self):
        L = torch.zeros(6, 6, device=self.lower_diag.device)
        idx = torch.tril_indices(6, 6, offset=-1)
        L[idx[0], idx[1]] = self.lower_off
        L[range(6), range(6)] = F.softplus(self.lower_diag) + 1e-6
        return L
    def forward(self): L = self.L(); return L @ L.T

class BallBallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 18))
    def forward(self, xi, xj):
        diff = xi[:, :3] - xj[:, :3]
        return self.net(torch.cat([xi, xj, diff], dim=1)).view(-1, 6, 3)

class BallRobotMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(54, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 36))
    def forward(self, xi, arm_state):
        return self.net(torch.cat([xi, arm_state], dim=1)).view(-1, 6, 6)

class BallWallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 18))
    def forward(self, xi, wall_normal):
        return self.net(torch.cat([xi, wall_normal], dim=1)).view(-1, 6, 3)

class PHGNNPhase1(nn.Module):
    _J_tmp = torch.zeros(6, 6)
    _J_tmp[3:, :3] = torch.eye(3)
    J_FIXED = _J_tmp - _J_tmp.T
    def __init__(self, n_walls=4):
        super().__init__()
        self.Hnet   = HamiltonianNet()
        self.Rparam = DissipationMatrix()
        self.Bbb    = BallBallMLP()
        self.Bbr    = BallRobotMLP()
        self.Bbw    = nn.ModuleList([BallWallMLP() for _ in range(n_walls)])
        self.register_buffer("J", self.J_FIXED.clone())

    def compute_dx(self, xp, bb_src, bb_dst, br_src, arm_state, arm_u,
                   bw_src, wall_normals):
        grad_H        = self.Hnet.grad(xp)
        grad_H_scaled = grad_H.clone()
        grad_H_scaled[:, 3:6] = grad_H[:, 3:6] / PARTICLE_MASS
        JmR = self.J - self.Rparam()
        dx  = grad_H_scaled @ JmR.T
        if bb_src.numel() > 0:
            xi_bb = xp[bb_src]; xj_bb = xp[bb_dst]
            grad_j        = self.Hnet.grad(xj_bb)
            grad_j_scaled = grad_j.clone()
            grad_j_scaled[:, 3:6] = grad_j[:, 3:6] / PARTICLE_MASS
            Bij = self.Bbb(xi_bb, xj_bb); Bji = self.Bbb(xj_bb, xi_bb)
            u_ij = -torch.einsum("eij,ej->ei", Bji.transpose(1, 2), grad_j_scaled)
            bbc  = torch.einsum("eij,ej->ei", Bij, u_ij)
            dx.scatter_add_(0, bb_dst.unsqueeze(1).expand_as(bbc), bbc)
        if br_src.numel() > 0:
            xi_br = xp[br_src]
            Bbr   = self.Bbr(xi_br, arm_state)
            u_ik6 = torch.cat([arm_u, torch.zeros_like(arm_u)], dim=1)
            brc   = torch.einsum("eij,ej->ei", Bbr, u_ik6)
            dx.scatter_add_(0, br_src.unsqueeze(1).expand_as(brc), brc)
        for widx, (bws, Bbww) in enumerate(zip(bw_src, self.Bbw)):
            if bws.numel() == 0: continue
            xi_bw = xp[bws]
            nw    = wall_normals[widx].unsqueeze(0).expand(bws.shape[0], 3)
            Bout  = Bbww(xi_bw, nw)
            vel_w = xi_bw[:, 3:6]
            u_iw  = -torch.einsum("ei,ei->e", vel_w, nw).unsqueeze(1) * nw
            bwc   = torch.einsum("eij,ej->ei", Bout, u_iw)
            dx.scatter_add_(0, bws.unsqueeze(1).expand_as(bwc), bwc)
        return dx

    def forward(self, xp, bb_src, bb_dst, br_src, arm_state, arm_u,
                bw_src, wall_normals, dt):
        args = (bb_src, bb_dst, br_src, arm_state, arm_u, bw_src, wall_normals)
        dx_n   = self.compute_dx(xp, *args)
        p_half = xp.clone()
        p_half[:, 3:6] = xp[:, 3:6] + dt / 2.0 * dx_n[:, 3:6]
        x_half  = torch.cat([xp[:, :3], p_half[:, 3:6]], dim=1)
        dx_half = self.compute_dx(x_half, *args)
        q_next  = xp[:, :3] + dt * dx_half[:, :3]
        x_for_p = torch.cat([q_next, p_half[:, 3:6]], dim=1)
        dx_next = self.compute_dx(x_for_p, *args)
        p_next  = p_half[:, 3:6] + dt / 2.0 * dx_next[:, 3:6]
        return torch.cat([q_next, p_next], dim=1)

# ─── Loss functions ────────────────────────────────────────────────────────────
def loss_state(x_pred, x_true): return F.mse_loss(x_pred, x_true)

def loss_R_psd(R_matrix):
    eigv = torch.linalg.eigvalsh(R_matrix.float())
    return (F.relu(-eigv) ** 2).sum()

def total_loss(x_pred, x_true, R_matrix):
    Ls      = loss_state(x_pred, x_true)
    Lr      = loss_R_psd(R_matrix)
    Lr_norm = R_matrix.float().norm()
    L = (LAMBDA1 * Ls
         + LAMBDA2 * Lr.to(x_pred.dtype)
         + LAMBDA4 * Lr_norm.to(x_pred.dtype))
    return L, Ls.item(), Lr.item()

def val_metrics(x_pred, x_true):
    x_pred_c = torch.nan_to_num(x_pred.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x_true_c = x_true.float()
    pos_mse  = F.mse_loss(x_pred_c[:, :3], x_true_c[:, :3]).item()
    vel_pred = x_pred_c[:, 3:6] / PARTICLE_MASS
    vel_true = x_true_c[:, 3:6] / PARTICLE_MASS
    vel_mse  = F.mse_loss(vel_pred, vel_true).item()
    return pos_mse, vel_mse

# ─── Test Dataset ──────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, phases=None):
        p_idx  = pd.read_csv(TEST_ROOT / "particles_index.csv")
        s_idx  = pd.read_csv(TEST_ROOT / "scooper_index.csv")
        merged = p_idx.merge(s_idx[["frame_idx", "loop", "phase", "step"]],
                             on=["frame_idx", "loop", "phase", "step"])
        if phases:
            merged = merged[merged["phase"].isin(phases)]
        self.runs   = []
        self.phases = []
        for (_, _), grp in merged.groupby(["loop", "phase"]):
            grp   = grp.sort_values("step").reset_index(drop=True)
            steps = grp["step"].tolist()
            fids  = grp["frame_idx"].tolist()
            ph    = grp["phase"].iloc[0]
            n     = len(grp)
            run_start = 0
            for k in range(1, n):
                consecutive = (steps[k] - steps[k - 1] == 1)
                run_end = k if not consecutive else None
                if run_end is not None or k == n - 1:
                    run_end = k if run_end is None else run_end
                    run_fids = fids[run_start:run_end + 1]
                    if len(run_fids) > ROLLOUT_STEPS:
                        self.runs.append(run_fids)
                        self.phases.append(ph)
                    run_start = k
        print(f"Pre-loading test frames into RAM...")
        p = np.load(TEST_ROOT / "particles_full.npz", allow_pickle=True)
        s = np.load(TEST_ROOT / "scooper_full.npz",   allow_pickle=True)
        needed = set(fi for run in self.runs for fi in run)
        print(f"  Caching {len(needed)} unique frames...")
        self.pcache = {i: p[f"frame_{i}"]     for i in needed}
        self.spos   = {i: s[f"frame_{i}_pos"] for i in needed}
        self.svel   = {i: s[f"frame_{i}_vel"] for i in needed}
        self.sft    = {i: s[f"frame_{i}_ft"]  for i in needed}
        del p, s
        n_windows = sum(len(r) // ROLLOUT_STEPS for r in self.runs)
        print(f"  Cache ready — {len(self.runs)} runs, ~{n_windows} 5-step windows.")

    def __len__(self): return len(self.runs)

    def __getitem__(self, idx):
        fids  = self.runs[idx]
        phase = self.phases[idx]
        offset = torch.tensor(PARTICLE_SET_OFFSET, dtype=torch.float32)
        x_seq, arm_states, arm_us, arm_pos_nps = [], [], [], []
        for fi in fids:
            p = torch.tensor(self.pcache[fi], dtype=torch.float32)
            p[:, :3] += offset
            p[:, 3:6] *= PARTICLE_MASS
            x_seq.append(p)
            arm_pos = torch.tensor(self.spos[fi], dtype=torch.float32)
            arm_vel = torch.tensor(self.svel[fi], dtype=torch.float32)
            arm_ft  = torch.tensor(self.sft[fi],  dtype=torch.float32)
            arm_states.append(torch.cat([arm_pos, arm_vel, arm_ft.flatten()]))
            arm_us.append(arm_ft[5, :3])
            arm_pos_nps.append(self.spos[fi].astype(np.float64))
        return x_seq, arm_states, arm_us, arm_pos_nps, phase

# ─── Load model weights with key remapping ────────────────────────────────────
KEY_MAP = {
    "H_net.": "Hnet.", "R_param.": "Rparam.",
    "B_bb.":  "Bbb.",  "B_br.":    "Bbr.",   "B_bw.": "Bbw.",
}

def load_model(ckpt_path):
    model = PHGNNPhase1().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    remapped = {}
    for k, v in ckpt.items():
        new_k = k
        for old_k, new_name in KEY_MAP.items():
            if new_k.startswith(old_k):
                new_k = new_name + new_k[len(old_k):]
                break
        remapped[new_k] = v
    model.load_state_dict(remapped)
    model.eval()
    return model

# ─── Evaluate one model on the full test set ──────────────────────────────────
def evaluate_model(model, loader, wall_normals):
    """Returns dict of aggregate metrics for this model checkpoint."""
    pos_mse_acc    = 0.0
    vel_mse_acc    = 0.0
    tot_loss_acc   = 0.0
    r_psd_acc      = 0.0
    n_windows      = 0
    n_loss_samples = 0

    with torch.no_grad():
        for sample in loader:
            x_seq, arm_states, arm_us, arm_pos_nps, phase = sample
            T = len(x_seq)
            window_start = 0
            while window_start + ROLLOUT_STEPS < T:
                x_seed = x_seq[window_start].to(DEVICE)
                q_seed = x_seed[:, :3].detach().contiguous().float()
                anp    = arm_pos_nps[window_start].astype(np.float64)
                bbs, bbd, brs, bwl, act = build_proximity_graph(
                    q_seed, arm_pos_np=anp)

                win_pos_acc = 0.0
                win_vel_acc = 0.0
                win_tot_acc = 0.0
                x_rolling   = x_seed

                for step in range(ROLLOUT_STEPS):
                    frame_idx = window_start + step + 1
                    x_true = x_seq[frame_idx].to(DEVICE)
                    (x_roll_s, x_true_s,
                     bbs_s, bbd_s, brs_s, bwl_s) = subsample_to_active(
                        x_rolling, x_true, bbs, bbd, brs, bwl, act)
                    bbs_d = bbs_s.to(DEVICE); bbd_d = bbd_s.to(DEVICE)
                    brs_d = brs_s.to(DEVICE)
                    bwl_d = [w.to(DEVICE) for w in bwl_s]
                    arm_sb = arm_states[window_start + step].to(DEVICE)
                    arm_ub = arm_us[window_start + step].to(DEVICE)
                    if brs_d.numel() > 0:
                        arm_se = arm_sb.unsqueeze(0).expand(brs_d.shape[0], -1)
                        arm_ue = arm_ub.unsqueeze(0).expand(brs_d.shape[0], -1)
                    else:
                        arm_se = torch.empty(0, 48, device=DEVICE)
                        arm_ue = torch.empty(0, 3,  device=DEVICE)

                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        x_pred = model(x_roll_s, bbs_d, bbd_d, brs_d,
                                       arm_se, arm_ue, bwl_d, wall_normals, DT)
                    R_mat = model.Rparam()
                    L, Ls, Lr = total_loss(
                        x_pred.float(), x_true_s.float(), R_mat.float())
                    pm, vm = val_metrics(x_pred, x_true_s)
                    w = 0.9 ** step
                    win_pos_acc += w * pm
                    win_vel_acc += w * vm
                    win_tot_acc += w * L.item()
                    x_rolling = x_pred

                rollout_loss  = win_tot_acc / ROLLOUT_STEPS
                pos_mse_acc  += win_pos_acc / ROLLOUT_STEPS
                vel_mse_acc  += win_vel_acc / ROLLOUT_STEPS
                n_windows    += 1
                if math.isfinite(rollout_loss):
                    tot_loss_acc  += rollout_loss
                    r_psd_acc     += Lr
                    n_loss_samples += 1

                window_start += ROLLOUT_STEPS

    R_mat    = model.Rparam()
    R_norm   = R_mat.float().norm().item()
    R_eigmin = torch.linalg.eigvalsh(R_mat.float()).min().item()

    return {
        "pos_mse":    pos_mse_acc   / max(n_windows,      1),
        "vel_mse":    vel_mse_acc   / max(n_windows,      1),
        "total_loss": tot_loss_acc  / max(n_loss_samples, 1),
        "R_psd_loss": r_psd_acc     / max(n_loss_samples, 1),
        "R_norm":     R_norm,
        "R_eigmin":   R_eigmin,
        "n_windows":  n_windows,
    }

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device       : {DEVICE}")
    print(f"Test dataset : {TEST_ROOT}")
    print(f"Models dir   : {MODELS_DIR}")

    # ── Find all model_best_subsample_N.pt files, sorted by epoch number ─────
    pattern     = re.compile(r"^model_best_subsample_(\d+)\.pt$")
    model_files = []
    for f in MODELS_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            model_files.append((int(m.group(1)), f))
    if not model_files:
        raise FileNotFoundError(
            f"No model_best_subsample_N.pt files found in {MODELS_DIR}.\n"
            f"Expected filenames like: model_best_subsample_1.pt, "
            f"model_best_subsample_2.pt, ...")
    model_files.sort(key=lambda x: x[0])
    epochs_found = [ep for ep, _ in model_files]
    print(f"\nFound {len(model_files)} subsample checkpoints: epochs {epochs_found}")

    # ── Validate test dataset files ───────────────────────────────────────────
    required = [
        TEST_ROOT / "particles_full.npz",
        TEST_ROOT / "particles_index.csv",
        TEST_ROOT / "scooper_full.npz",
        TEST_ROOT / "scooper_index.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing test dataset files:\n" + "\n".join(missing))

    # ── Build dataset once — reuse across all epoch evaluations ──────────────
    test_ds = TestDataset(phases=ALL_PHASES)
    loader  = DataLoader(test_ds, batch_size=None, shuffle=False,
                         num_workers=0,
                         pin_memory=(DEVICE.type == "cuda"),
                         collate_fn=lambda x: x)
    wall_normals = WALL_NORMALS.to(DEVICE)

    # ── W&B init ──────────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run,
        config  = dict(
            checkpoint_prefix = "model_best_subsample",
            test_dataset      = str(TEST_ROOT),
            models_dir        = str(MODELS_DIR),
            n_checkpoints     = len(model_files),
            epochs_found      = epochs_found,
            rollout_steps     = ROLLOUT_STEPS,
            n_star            = N_STAR,
            dt                = DT,
            lambda1           = LAMBDA1,
            lambda2           = LAMBDA2,
            lambda4           = LAMBDA4,
            amp               = USE_AMP,
        ),
        tags = ["evaluation", "zigzag", "epoch-sweep", "subsample"],
    )

    summary_tbl = wandb.Table(columns=[
        "epoch", "pos_mse", "vel_mse", "total_loss",
        "R_psd_loss", "R_norm", "R_eigmin", "n_windows"
    ])
    all_results = {}

    # ── Sweep over all subsample checkpoints ──────────────────────────────────
    for epoch, ckpt_path in tqdm(model_files, desc="Subsample checkpoints",
                                  unit="model", file=sys.stderr,
                                  dynamic_ncols=True):
        print(f"\n── Epoch {epoch:3d}  ({ckpt_path.name}) ──────────────────")
        model   = load_model(ckpt_path)
        metrics = evaluate_model(model, loader, wall_normals)
        all_results[epoch] = metrics

        wandb.log({
            "epoch_sweep/pos_mse":    metrics["pos_mse"],
            "epoch_sweep/vel_mse":    metrics["vel_mse"],
            "epoch_sweep/total_loss": metrics["total_loss"],
            "epoch_sweep/R_psd_loss": metrics["R_psd_loss"],
            "epoch_sweep/R_norm":     metrics["R_norm"],
            "epoch_sweep/R_eigmin":   metrics["R_eigmin"],
            "epoch_sweep/n_windows":  metrics["n_windows"],
            "epoch": epoch,
        }, step=epoch)

        summary_tbl.add_data(
            epoch,
            metrics["pos_mse"], metrics["vel_mse"], metrics["total_loss"],
            metrics["R_psd_loss"], metrics["R_norm"], metrics["R_eigmin"],
            metrics["n_windows"],
        )
        print(f"   pos_mse={metrics['pos_mse']:.6f}  "
              f"vel_mse={metrics['vel_mse']:.6f}  "
              f"total_loss={metrics['total_loss']:.6f}")

        del model
        torch.cuda.empty_cache()

    # ── Log summary table ─────────────────────────────────────────────────────
    wandb.log({"epoch_sweep/summary_table": summary_tbl})

    # ── Identify best epoch per metric ────────────────────────────────────────
    best_pos_epoch  = min(all_results, key=lambda e: all_results[e]["pos_mse"])
    best_vel_epoch  = min(all_results, key=lambda e: all_results[e]["vel_mse"])
    best_loss_epoch = min(all_results, key=lambda e: all_results[e]["total_loss"])
    wandb.run.summary.update({
        "best_epoch/pos_mse":    best_pos_epoch,
        "best_epoch/vel_mse":    best_vel_epoch,
        "best_epoch/total_loss": best_loss_epoch,
        "best_pos_mse":          all_results[best_pos_epoch]["pos_mse"],
        "best_vel_mse":          all_results[best_vel_epoch]["vel_mse"],
        "best_total_loss":       all_results[best_loss_epoch]["total_loss"],
    })

    # ── Save full results JSON ────────────────────────────────────────────────
    results_path = OUT_DIR / "zigzag_eval_subsample_epochs.json"
    json.dump(
        {str(ep): m for ep, m in all_results.items()},
        open(results_path, "w"), indent=2
    )
    print(f"\nAll results saved to {results_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n── Subsample Epoch Sweep Results ───────────────────────────────")
    print(f"{'Epoch':>6}  {'pos_mse':>12}  {'vel_mse':>12}  {'total_loss':>12}")
    print("─" * 48)
    for ep in sorted(all_results):
        m = all_results[ep]
        print(f"  {ep:4d}   {m['pos_mse']:12.6f}  "
              f"{m['vel_mse']:12.6f}  {m['total_loss']:12.6f}")
    print(f"\nBest pos_mse    → epoch {best_pos_epoch}"
          f"  ({all_results[best_pos_epoch]['pos_mse']:.6f})")
    print(f"Best vel_mse    → epoch {best_vel_epoch}"
          f"  ({all_results[best_vel_epoch]['vel_mse']:.6f})")
    print(f"Best total_loss → epoch {best_loss_epoch}"
          f"  ({all_results[best_loss_epoch]['total_loss']:.6f})")

    wandb.finish()


if __name__ == "__main__":
    main()
