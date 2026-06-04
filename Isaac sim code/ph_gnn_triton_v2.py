"""
Port-Hamiltonian Graph Neural Network — Phase 1 (Revised)

Changes from previous version:
1. n* fixed at 6 — no dynamic computation or curve fitting
2. 100 epochs, all parameters unfrozen from the start (no warmup freeze)
3. CUDA-optimised: pinned memory, persistent workers, AMP (torch.amp),
   torch.backends.cudnn.benchmark, fused Adam, graph on device
4. Particle sampling: only particles within N_STAR hop-distance of scooper
   contact are used — no random 500-subsample
5. Weights & Biases (wandb) tracking with W&B Tables for predictions
6. 90/10 train/val split (by frame-pair index), printed at startup
7. Validation loss based on position MSE and velocity MSE (not energy)
8. Per-epoch timer using time.perf_counter
"""

import os
import sys

# Reduce CUDA memory fragmentation — important when many small graph tensors
# are allocated and freed during multi-step rollout with retain_graph=True.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import math
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

# Use torch_cluster for GPU-accelerated radius graph if available
try:
    from torch_cluster import radius_graph as tc_radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False
import pytorch_kinematics as pk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm



# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--urdf", type=str,
                    default="/scratch/work/venkatv1/dataset/spot.urdf")
parser.add_argument("--wandb_project", type=str, default="ph-gnn-phase1",
                    help="W&B project name")
parser.add_argument("--wandb_run", type=str, default=None,
                    help="Optional W&B run name")
args, _ = parser.parse_known_args()

# ============================================================
# Config
# ============================================================
DATASET_ROOT = Path("/scratch/work/venkatv1/dataset")
OUT_DIR      = Path("/scratch/work/venkatv1/ph_gnn_outputs")
OUT_DIR.mkdir(exist_ok=True)

# --- Change 1: Fixed n* = 6 ---
N_STAR = 6

# --- Change 2: 100 epochs, no warmup ---
EPOCHS     = 20
BATCH_SIZE = 1  # reduced from 4 — rollout of 5 steps acts as implicit batch
LR         = 1e-3
STEP_LR_STEP  = 20
STEP_LR_GAMMA = 0.5
DT             = 1.0 / 60.0

# --- Change 3: CUDA optimisations ---
# torch.backends.cudnn.benchmark intentionally disabled — no convolutions in this model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"                # automatic mixed precision
scaler  = torch.amp.GradScaler('cuda', enabled=USE_AMP)

# Loss weights
LAMBDA1 = 1000.0   # state (position + velocity) MSE
LAMBDA2 = 0.5   # R PSD penalty
LAMBDA3 = 0.0   # FP loss disabled (n* is fixed)
LAMBDA4 = 1e-4  # R Frobenius regularisation

EPSILON_SCALE = 0.01

# Sliding-window epoch sampling:
# Each epoch uses a 10% window of the training pool, sliding by 5% per epoch.
# Windows overlap (epoch 1 = 0-10%, epoch 2 = 5-15%, ..., epoch 20 = 95-105% → wraps to 0-10%).
# Within each window, only complete instruction sequences (multiples of 510 frames) are used.
EPOCH_WINDOW_FRACTION = 0.10   # 10% of training pool per epoch
EPOCH_SLIDE_FRACTION  = 0.05   # slide by 5% each epoch
FRAMES_PER_INSTRUCTION = 510   # each complete motion = 510 steps (8 phases × ~60-90 steps)

# Particle physics
R_PARTICLE      = 0.03
R_THRESH        = 2 * R_PARTICLE
R_WALL          = 0.01
PARTICLE_DENSITY= 2000.0
PARTICLE_MASS   = PARTICLE_DENSITY * (4.0 / 3.0) * math.pi * R_PARTICLE ** 3

# FK / robot-world
END_EFFECTOR      = "arm0_link_wr1"
ROBOT_BASE_WORLD  = np.array([-0.55, 1.43801, 0.50328], dtype=np.float64)
DTYPE_FK          = torch.float64
DATASET_JOINT_IDX = [0, 1, 3, 4, 5, 6]
WR1_LIMIT         = 2.88

# Scoop OBB geometry (local EE frame)
SCOOP_OFFSET_EE = np.array(
    [0.167 + 0.09454, -0.077 - 0.06384, -0.103 + 0.07927], dtype=np.float64
)
SCOOP_HALF_EXTENTS = np.array(
    [0.08759 / 2 + 0.008, 0.14759 / 2 + 0.008, 0.23 / 2 + 0.008]
)
SCOOP_HALF      = SCOOP_HALF_EXTENTS + 0.005
R_scoop_local   = Rotation.from_euler(
    "xyz", [-149.515, 61.231, 145.323], degrees=True
).as_matrix()

PARTICLE_SET_OFFSET = np.array([0.4, 0.0, 0.09279], dtype=np.float64)

R_PHASES   = ["flat_start"]
FP_PHASES  = ["move_forward", "turn", "move_forward_2"]
ALL_PHASES = R_PHASES + FP_PHASES

# ============================================================
# Build FK chain (once at module load)
# ============================================================
print(f"Loading kinematic chain from: {args.urdf}")
_fk_chain = pk.build_serial_chain_from_urdf(
    open(args.urdf).read(),
    end_link_name=END_EFFECTOR,
    root_link_name="body",
)
_fk_chain  = _fk_chain.to(dtype=DTYPE_FK, device=DEVICE)
N_CHAIN    = len(_fk_chain.get_joint_parameter_names())
print(f"  Chain joints ({N_CHAIN}): {_fk_chain.get_joint_parameter_names()}")


def fk_ee(arm_pos_np: np.ndarray):
    th_full = np.zeros(N_CHAIN, dtype=np.float64)
    th_full[DATASET_JOINT_IDX] = arm_pos_np
    th_full[6] = np.clip(th_full[6], -WR1_LIMIT, WR1_LIMIT)
    th = torch.tensor(th_full, dtype=DTYPE_FK, device=DEVICE).unsqueeze(0)
    T  = _fk_chain.forward_kinematics(th).get_matrix().squeeze(0).cpu().numpy()
    return ROBOT_BASE_WORLD + T[:3, 3], T[:3, :3]


def obb_contact_np(particles_world, p_ee_world, R_ee_world):
    centre = p_ee_world + R_ee_world @ SCOOP_OFFSET_EE
    R_obb  = R_ee_world @ R_scoop_local
    local  = (particles_world - centre[np.newaxis, :]) @ R_obb
    return (
        (np.abs(local[:, 0]) <= SCOOP_HALF[0]) &
        (np.abs(local[:, 1]) <= SCOOP_HALF[1]) &
        (np.abs(local[:, 2]) <= SCOOP_HALF[2])
    )


# ============================================================
# Scene geometry
# ============================================================
WALL_CENTERS = torch.tensor([
    [0.2962 + 0.10371, 1.5250 + 0.3218, 0.0750 - 0.5401],
    [0.9361 + 0.10371, 1.5108 + 0.3218, 0.1024 - 0.5401],
    [0.4698 + 0.10371, 1.0382 + 0.3218, 0.1024 - 0.5401],
    [0.4698 + 0.10371, 1.9894 + 0.3218, 0.1024 - 0.5401],
], dtype=torch.float32)

WALL_NORMALS = torch.tensor([
    [ 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.0],
    [ 0.0,-1.0, 0.0],
], dtype=torch.float32)


# ============================================================
# Model blocks
# ============================================================
class HamiltonianNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

    def grad(self, x):
        """
        ∇H_θ(xᵢ) for all N particles, differentiable w.r.t. network parameters.

        The canonical pattern for input-gradient networks (HNN, SympNet, etc.):
          - .clone() creates a new tensor that participates in the autograd graph
            through its clone relationship — unlike .detach() which severs it.
          - .requires_grad_(True) on the clone adds x_in as a leaf in the INNER
            graph (w.r.t. x) while the OUTER graph (w.r.t. θ) flows through
            the network weights as normal.
          - autocast disabled: autograd.grad + create_graph is unreliable in fp16.
          - torch.enable_grad(): guarantees grad computation even if the caller
            used torch.no_grad() (e.g. during validation).

        Why NOT .detach():
          .detach() breaks the outer graph connection — PyTorch cannot then
          differentiate the returned gradient w.r.t. θ during the backward pass,
          causing "does not require grad" errors.
        """
        with torch.amp.autocast("cuda", enabled=False):
            with torch.enable_grad():
                x_in = x.float().clone().requires_grad_(True)
                H    = self.net(x_in)
                g    = torch.autograd.grad(
                           H.sum(), x_in,
                           create_graph=self.training,
                           retain_graph=True,   # must be True — Verlet calls _compute_dx 3× sharing the same outer graph
                           allow_unused=False,
                       )[0]
        return g


class DissipationMatrix(nn.Module):
    def __init__(self):
        super().__init__()
        self.lower_off  = nn.Parameter(torch.zeros(15))
        self.lower_diag = nn.Parameter(torch.zeros(6))

    def L(self):
        L   = torch.zeros(6, 6, device=self.lower_diag.device)
        idx = torch.tril_indices(6, 6, offset=-1)
        L[idx[0], idx[1]] = self.lower_off
        L[range(6), range(6)] = F.softplus(self.lower_diag) + 1e-6
        return L

    def forward(self):
        L = self.L()
        return L @ L.T


class BallBallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 18),
        )

    def forward(self, xi, xj):
        diff = xi[:, :3] - xj[:, :3]
        return self.net(torch.cat([xi, xj, diff], dim=1)).view(-1, 6, 3)


class BallRobotMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(54, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 36),
        )

    def forward(self, xi, arm_state):
        return self.net(torch.cat([xi, arm_state], dim=1)).view(-1, 6, 6)


class BallWallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 18),
        )

    def forward(self, xi, wall_normal):
        return self.net(torch.cat([xi, wall_normal], dim=1)).view(-1, 6, 3)


class PHGNNPhase1(nn.Module):
    _J = torch.zeros(6, 6)
    _J[:3, 3:] = torch.eye(3)
    J_FIXED = _J - _J.T

    def __init__(self, n_walls=4):
        super().__init__()
        self.H_net  = HamiltonianNet()
        self.R_param = DissipationMatrix()
        self.B_bb    = BallBallMLP()
        self.B_br    = BallRobotMLP()
        self.B_bw    = nn.ModuleList([BallWallMLP() for _ in range(n_walls)])
        self.register_buffer("J", self.J_FIXED.clone())

    def _compute_dx(self, x_p, bb_src, bb_dst, br_src, arm_state, arm_u,
                    bw_src, wall_normals):
        """
        Compute the pH state derivative dx/dt at state x_p.
        Factored out so it can be called at multiple sub-steps by the integrator.
        Returns dx ∈ R^(N,6) — same shape as x_p.
        """
        grad_H = self.H_net.grad(x_p)
        grad_H_scaled = grad_H.clone()
        grad_H_scaled[:, 3:6] = grad_H[:, 3:6] / PARTICLE_MASS
        JmR = self.J - self.R_param()
        dx  = grad_H_scaled @ JmR.T

        if bb_src.numel() > 0:
            xi_bb = x_p[bb_src]
            xj_bb = x_p[bb_dst]
            grad_j = self.H_net.grad(xj_bb)
            grad_j_scaled = grad_j.clone()
            grad_j_scaled[:, 3:6] = grad_j[:, 3:6] / PARTICLE_MASS
            B_ij  = self.B_bb(xi_bb, xj_bb)
            B_ji  = self.B_bb(xj_bb, xi_bb)
            u_ij  = -torch.einsum("eij,ej->ei", B_ji.transpose(1, 2), grad_j_scaled)
            bb_c  = torch.einsum("eij,ej->ei", B_ij, u_ij)
            dx.scatter_add_(0, bb_dst.unsqueeze(1).expand_as(bb_c), bb_c)

        if br_src.numel() > 0:
            xi_br  = x_p[br_src]
            B_br   = self.B_br(xi_br, arm_state)
            u_ik6  = torch.cat([arm_u, torch.zeros_like(arm_u)], dim=1)
            br_c   = torch.einsum("eij,ej->ei", B_br, u_ik6)
            dx.scatter_add_(0, br_src.unsqueeze(1).expand_as(br_c), br_c)

        for w_idx, (bw_s, B_bw_w) in enumerate(zip(bw_src, self.B_bw)):
            if bw_s.numel() == 0:
                continue
            xi_bw  = x_p[bw_s]
            n_w    = wall_normals[w_idx].unsqueeze(0).expand(bw_s.shape[0], 3)
            B_out  = B_bw_w(xi_bw, n_w)
            vel_w  = xi_bw[:, 3:6]
            u_iw   = -torch.einsum("ei,ei->e", vel_w, n_w).unsqueeze(1) * n_w
            bw_c   = torch.einsum("eij,ej->ei", B_out, u_iw)
            dx.scatter_add_(0, bw_s.unsqueeze(1).expand_as(bw_c), bw_c)

        return dx

    def forward(self, x_p, bb_src, bb_dst, br_src, arm_state, arm_u,
                bw_src, wall_normals, dt):
        """
        Störmer–Verlet (velocity Verlet) symplectic integrator.

        Replaces the original Euler step  x_{n+1} = x_n + dt * f(x_n).

        Störmer–Verlet preserves the symplectic 2-form of Hamiltonian phase space,
        meaning it conserves a *shadow Hamiltonian* close to the true H over long
        rollouts — Euler accumulates O(dt) energy drift per step, Verlet only O(dt²).

        Split on (q, p):
          state x = [q | p],  columns 0:3 = position q,  columns 3:6 = momentum p

        Step 1 — half-step momentum update (uses force at x_n):
          dx_n   = f(x_n)       ← full pH derivative at current state
          p_half = p_n + (dt/2) * dx_n[:, 3:6]

        Step 2 — full-step position update (uses velocity at half-state):
          x_half        = [q_n | p_half]
          dx_half_pos   = f(x_half)[:, 0:3]   ← only need q-rows (∂H/∂p = v)
          q_next        = q_n + dt * dx_half_pos

        Step 3 — half-step momentum update (uses force at x_next):
          x_next_half   = [q_next | p_half]
          dx_next       = f(x_next_half)
          p_next        = p_half + (dt/2) * dx_next[:, 3:6]

        Cost vs Euler:
          Euler:  1 × _compute_dx  (1 H_net.grad call for x_p, 1 for xj_bb)
          Verlet: 3 × _compute_dx  (3 H_net.grad calls for x_p, 3 for xj_bb)
          → ~3× more GPU compute per forward pass
          → Each epoch goes from ~8 min to ~20-24 min
          → 100 epochs: ~35-40 hrs total (vs ~19 hrs for Euler)
        """
        args = (bb_src, bb_dst, br_src, arm_state, arm_u, bw_src, wall_normals)

        # ── Step 1: half-step momentum ────────────────────────────────────────
        dx_n   = self._compute_dx(x_p, *args)
        p_half = x_p[:, 3:6] + (dt / 2.0) * dx_n[:, 3:6]

        # ── Step 2: full-step position using half-step momentum ───────────────
        x_half    = torch.cat([x_p[:, 0:3], p_half], dim=1)
        dx_half   = self._compute_dx(x_half, *args)
        q_next    = x_p[:, 0:3] + dt * dx_half[:, 0:3]

        # ── Step 3: half-step momentum using updated position ─────────────────
        x_for_p   = torch.cat([q_next, p_half], dim=1)
        dx_next   = self._compute_dx(x_for_p, *args)
        p_next    = p_half + (dt / 2.0) * dx_next[:, 3:6]

        return torch.cat([q_next, p_next], dim=1)


# ============================================================
# Dataset
# ============================================================
# Number of steps in the multi-step rollout loss: predict t+1 through t+ROLLOUT_STEPS
ROLLOUT_STEPS = 5

# Minimum consecutive frames per training sequence drawn during epoch sampling.
# Each sequence is ROLLOUT_STEPS+1 frames long and must have at least this many
# consecutive frames before a new sequence starts.
SEQ_STRIDE    = 30   # sample a new start every SEQ_STRIDE frames within a run


class PHGNNDataset(Dataset):
    """
    Stores sequences of (ROLLOUT_STEPS+1) consecutive frames from the same
    loop+phase. Each item returns:
      - x_seq:     list of (ROLLOUT_STEPS+1) particle state tensors (t .. t+R)
      - arm_states: list of ROLLOUT_STEPS arm state tensors (t .. t+R-1)
      - arm_us:     list of ROLLOUT_STEPS arm force tensors
      - arm_pos_nps: list of ROLLOUT_STEPS numpy arm positions
      - phase:      string phase name
    """
    def __init__(self, phases=None):
        p_idx  = pd.read_csv(DATASET_ROOT / "particles_index.csv")
        s_idx  = pd.read_csv(DATASET_ROOT / "scooper_index.csv")
        merged = p_idx.merge(
            s_idx[["frame_idx", "loop", "phase", "step"]],
            on=["frame_idx", "loop", "phase", "step"],
        )
        if phases:
            merged = merged[merged["phase"].isin(phases)]

        # Build sequences of ROLLOUT_STEPS+1 consecutive frames
        self.seqs = []   # each entry: list of frame_idxs [t, t+1, ..., t+ROLLOUT_STEPS]
        self.phases = []

        for (_, _), grp in merged.groupby(["loop", "phase"]):
            grp   = grp.sort_values("step").reset_index(drop=True)
            steps = grp["step"].tolist()
            fids  = grp["frame_idx"].tolist()
            ph    = grp["phase"].iloc[0]
            n     = len(grp)

            # Walk through the group finding runs of consecutive steps
            # and sample a sequence start every SEQ_STRIDE frames
            run_start = 0
            for k in range(1, n):
                consecutive = (steps[k] == steps[k-1] + 1)
                run_end     = k if not consecutive else None
                if run_end is not None or k == n - 1:
                    run_end = k if run_end is None else run_end
                    # Extract all valid sequence starts within this run
                    run_len = run_end - run_start
                    for s_start in range(run_start, run_end - ROLLOUT_STEPS + 1, SEQ_STRIDE):
                        seq_fids = fids[s_start : s_start + ROLLOUT_STEPS + 1]
                        if len(seq_fids) == ROLLOUT_STEPS + 1:
                            self.seqs.append(seq_fids)
                            self.phases.append(ph)
                    run_start = k

        print("Pre-loading frames into RAM (one-time)...")
        _p = np.load(DATASET_ROOT / "particles_full.npz", allow_pickle=True)
        _s = np.load(DATASET_ROOT / "scooper_full.npz",  allow_pickle=True)

        needed = set()
        for seq in self.seqs:
            for fi in seq:
                needed.add(fi)
        print(f"  Caching {len(needed)} unique frames...")

        self.p_cache = {i: _p[f"frame_{i}"] for i in needed}
        self.s_pos   = {i: _s[f"frame_{i}_pos"] for i in needed}
        self.s_vel   = {i: _s[f"frame_{i}_vel"] for i in needed}
        self.s_ft    = {i: _s[f"frame_{i}_ft"]  for i in needed}
        del _p, _s
        print(f"  Cache ready — {len(self.seqs):,} sequences of {ROLLOUT_STEPS+1} frames.\n")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        fids  = self.seqs[idx]
        phase = self.phases[idx]
        offset = torch.tensor(PARTICLE_SET_OFFSET, dtype=torch.float32)

        x_seq      = []   # ROLLOUT_STEPS+1 particle state tensors
        arm_states = []   # ROLLOUT_STEPS arm state tensors (one per transition)
        arm_us     = []
        arm_pos_nps = []

        for k, fi in enumerate(fids):
            p = torch.tensor(self.p_cache[fi], dtype=torch.float32)
            p[:, :3]  += offset
            p[:, 3:6] *= PARTICLE_MASS
            x_seq.append(p)

            if k < ROLLOUT_STEPS:
                arm_pos = torch.tensor(self.s_pos[fi], dtype=torch.float32)
                arm_vel = torch.tensor(self.s_vel[fi], dtype=torch.float32)
                arm_ft  = torch.tensor(self.s_ft[fi],  dtype=torch.float32)
                arm_states.append(torch.cat([arm_pos, arm_vel, arm_ft.flatten()]))
                arm_us.append(arm_ft[5, :3])
                arm_pos_nps.append(self.s_pos[fi].astype(np.float64))

        return x_seq, arm_states, arm_us, arm_pos_nps, phase


# ============================================================
# Graph construction
# ============================================================
def bfs_hops(adj: list, seeds: list, max_hops: int, N: int) -> torch.Tensor:
    """Return integer tensor of length N with hop distance from seeds (-1 = unreachable)."""
    hop_dist = torch.full((N,), -1, dtype=torch.long)
    queue = deque()
    for idx in seeds:
        if hop_dist[idx] < 0:
            hop_dist[idx] = 0
            queue.append(idx)
    while queue:
        node = queue.popleft()
        h    = hop_dist[node].item()
        if h >= max_hops:
            continue
        for nb in adj[node]:
            if hop_dist[nb] < 0:
                hop_dist[nb] = h + 1
                queue.append(nb)
    return hop_dist


def _radius_graph_cpu(q_np: np.ndarray, r_thresh: float):
    """Fallback: scipy KDTree (used when torch_cluster is absent)."""
    tree  = KDTree(q_np)
    pairs = tree.query_pairs(r_thresh, output_type="ndarray")
    if len(pairs):
        s = torch.from_numpy(pairs[:, 0]).long()
        d = torch.from_numpy(pairs[:, 1]).long()
        return torch.cat([s, d]), torch.cat([d, s])
    return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)


def _radius_graph_gpu(q_dev: torch.Tensor, r_thresh: float):
    """GPU radius graph via torch_cluster — ~10-20x faster than scipy KDTree."""
    # radius_graph returns edges where dst is within r of src (directed).
    # max_num_neighbors caps memory; 64 is safe for granular packing at r=0.06m.
    edge_index = tc_radius_graph(q_dev, r=r_thresh, loop=False, max_num_neighbors=64)
    # edge_index[0]=dst, edge_index[1]=src in torch_cluster convention
    # Return bidirectional: src->dst and dst->src already included by radius_graph
    return edge_index[1], edge_index[0]


# --- Change 4: only particles within N_STAR hops of contact ---
def build_proximity_graph(q, arm_pos_np=None, r_thresh=R_THRESH,
                           r_wall=R_WALL, n_star=N_STAR):
    """
    Build ball-ball, ball-robot, ball-wall edge lists.
    Uses GPU radius_graph (torch_cluster) when available, else scipy KDTree.
    Returns also a boolean mask (N,) for the 'active' subset:
      - For FP phases: particles within n_star hops of scooper contact
      - For R phases: all particles
    """
    N = q.shape[0]

    # ── Ball-ball edges ──────────────────────────────────────────────────────
    if HAS_TORCH_CLUSTER and q.device.type == "cuda":
        bb_src, bb_dst = _radius_graph_gpu(q, r_thresh)
    else:
        q_np = q.cpu().numpy()
        bb_src, bb_dst = _radius_graph_cpu(q_np, r_thresh)

    # ── Ball-robot (OBB contact) ─────────────────────────────────────────────
    q_np = q.cpu().numpy()   # needed for FK + wall checks regardless
    if arm_pos_np is not None:
        p_ee, R_ee = fk_ee(arm_pos_np)
        mask_contact = obb_contact_np(q_np, p_ee, R_ee)
        br_src = torch.tensor(np.where(mask_contact)[0], dtype=torch.long)
    else:
        br_src = torch.empty(0, dtype=torch.long)

    # ── Ball-wall ────────────────────────────────────────────────────────────
    bw_src_list = []
    for w_idx in range(4):
        wc   = WALL_CENTERS[w_idx].numpy()
        wn   = WALL_NORMALS[w_idx].numpy()
        dist = np.abs((q_np - wc) @ wn)
        bw_src_list.append(torch.tensor(np.where(dist < r_wall)[0], dtype=torch.long))

    # ── Active subset via BFS ────────────────────────────────────────────────
    if br_src.numel() > 0:
        bb_src_cpu = bb_src.cpu()
        bb_dst_cpu = bb_dst.cpu()
        adj = [[] for _ in range(N)]
        for s_i, d_i in zip(bb_src_cpu.tolist(), bb_dst_cpu.tolist()):
            adj[s_i].append(d_i)
        hop_dist = bfs_hops(adj, br_src.cpu().tolist(), max_hops=n_star, N=N)
        active   = (hop_dist >= 0)
    else:
        active = torch.ones(N, dtype=torch.bool)   # R-phase: use all particles

    return bb_src, bb_dst, br_src, bw_src_list, active


def subsample_to_active(x_t, x_t1, bb_src, bb_dst, br_src,
                         bw_src_list, active_mask):
    """
    Restrict particle tensors and edge indices to the active subset.
    Returns remapped tensors and edge lists.
    All edge tensors are moved to CPU for remapping, then returned on CPU
    (the training loop moves them to DEVICE afterwards).
    """
    # active_mask may be a bool tensor (CPU) — keep everything on CPU here
    active_mask = active_mask.cpu()
    idx_active  = active_mask.nonzero(as_tuple=True)[0]   # CPU, sorted
    if idx_active.numel() == 0:
        return (x_t, x_t1,
                bb_src.cpu(), bb_dst.cpu(),
                br_src.cpu(),
                [w.cpu() for w in bw_src_list])

    N = x_t.shape[0]

    # Build CPU remap table: old_idx -> new_idx  (-1 = not active)
    remap = torch.full((N,), -1, dtype=torch.long)          # CPU
    remap[idx_active] = torch.arange(idx_active.numel(), dtype=torch.long)

    # Slice particle tensors (may be on DEVICE — index with CPU idx_active)
    x_t_sub  = x_t[idx_active.to(x_t.device)]
    x_t1_sub = x_t1[idx_active.to(x_t1.device)]

    def remap_edges(src, dst):
        src_cpu = src.cpu(); dst_cpu = dst.cpu()
        mask = (remap[src_cpu] >= 0) & (remap[dst_cpu] >= 0)
        return remap[src_cpu[mask]], remap[dst_cpu[mask]]

    if bb_src.numel() > 0:
        new_bb_src, new_bb_dst = remap_edges(bb_src, bb_dst)
    else:
        new_bb_src = new_bb_dst = torch.empty(0, dtype=torch.long)

    # br_src remap
    br_cpu = br_src.cpu()
    valid_br = br_cpu[remap[br_cpu] >= 0]
    new_br_src = remap[valid_br]

    # bw_src remap
    new_bw_list = []
    for bw_s in bw_src_list:
        if bw_s.numel() > 0:
            bw_cpu  = bw_s.cpu()
            valid   = bw_cpu[remap[bw_cpu] >= 0]
            new_bw_list.append(remap[valid])
        else:
            new_bw_list.append(torch.empty(0, dtype=torch.long))

    return x_t_sub, x_t1_sub, new_bb_src, new_bb_dst, new_br_src, new_bw_list


# ============================================================
# Losses
# ============================================================
def loss_state(x_pred, x_true):
    return F.mse_loss(x_pred, x_true)


def loss_R_psd(R_matrix):
    # eigvalsh has no FP16 CUDA kernel — always promote to float32
    eigvals = torch.linalg.eigvalsh(R_matrix.float())
    return torch.clamp(-eigvals.min(), min=0.0) ** 2


# --- Change 7: position + velocity MSE for validation ---
def val_metrics(x_pred, x_true):
    """Return (pos_mse, vel_mse) in original (non-momentum) units."""
    pos_mse = F.mse_loss(x_pred[:, :3], x_true[:, :3])
    vel_pred = x_pred[:, 3:6] / PARTICLE_MASS
    vel_true = x_true[:, 3:6] / PARTICLE_MASS
    vel_mse  = F.mse_loss(vel_pred, vel_true)
    return pos_mse.item(), vel_mse.item()


def total_loss(x_pred, x_true, R_matrix):
    Ls      = loss_state(x_pred, x_true)
    Lr      = loss_R_psd(R_matrix)                   # internally casts to float32
    Lr_norm = R_matrix.float().norm() ** 2           # float32 norm
    L       = LAMBDA1 * Ls + LAMBDA2 * Lr.to(x_pred.dtype) + LAMBDA4 * Lr_norm.to(x_pred.dtype)
    return L, Ls.item(), Lr.item()


# ============================================================
# Training
# ============================================================
def main():
    print(f"Device      : {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Particle mass: {PARTICLE_MASS:.4f} kg")
    print(f"R_THRESH    : {R_THRESH} m")
    print(f"N_STAR      : {N_STAR} (fixed)")
    print(f"Epochs      : {EPOCHS}")
    print(f"AMP         : {USE_AMP}")

    required = [
        DATASET_ROOT / "particles_full.npz",
        DATASET_ROOT / "particles_index.csv",
        DATASET_ROOT / "scooper_full.npz",
        DATASET_ROOT / "scooper_index.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))

    # ── W&B init ────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "n_star":        N_STAR,
            "lambda1":       LAMBDA1,
            "lambda2":       LAMBDA2,
            "lambda4":       LAMBDA4,
            "step_lr_step":  STEP_LR_STEP,
            "step_lr_gamma": STEP_LR_GAMMA,
            "dt":            DT,
            "amp":           USE_AMP,
        },
    )

    # ── CUDA warm-up ────────────────────────────────────────────────────────
    # The first radius_graph / autograd call on a cold CUDA context triggers
    # PTX JIT compilation which can stall for 2-5 minutes silently.
    # Running a tiny warm-up here surfaces that wait with a visible message.
    if DEVICE.type == "cuda":
        print("Warming up CUDA (JIT compilation — may take 1-3 min on first run)...")
        _dummy = torch.randn(64, 6, device=DEVICE, requires_grad=True)
        _H     = torch.nn.Linear(6, 1).to(DEVICE)(_dummy).sum()
        torch.autograd.grad(_H, _dummy, create_graph=False)
        # Warm up AMP scaler path
        with torch.amp.autocast("cuda"):
            _ = torch.mm(torch.randn(128, 128, device=DEVICE),
                         torch.randn(128, 128, device=DEVICE))
        # Warm up torch_cluster radius_graph if available (first call triggers PTX JIT)
        if HAS_TORCH_CLUSTER:
            print("  Warming up torch_cluster radius_graph...", flush=True)
            _q_warm = torch.rand(200, 3, device=DEVICE)
            try:
                tc_radius_graph(_q_warm, r=0.1, loop=False, max_num_neighbors=32)
                torch.cuda.synchronize()
                print("  torch_cluster warm-up done.", flush=True)
            except Exception as e:
                print(f"  torch_cluster warm-up failed ({e}) — will use scipy fallback.", flush=True)
            del _q_warm
        torch.cuda.synchronize()
        del _dummy, _H, _
        print("CUDA warm-up done.\n")

    # ── Dataset & 90/10 split ───────────────────────────────────────────────
    full_dataset = PHGNNDataset(phases=ALL_PHASES)
    n_total = len(full_dataset)

    # ── Fixed 10% validation set (last 10%, never trained on) ──────────────
    n_val   = int(0.10 * n_total)
    n_train = n_total - n_val

    # Ordered split: first 90% = training pool, last 10% = validation
    train_idx = list(range(n_train))
    val_idx   = list(range(n_train, n_total))

    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)

    # ── Sliding window parameters ───────────────────────────────────────────
    # 20% window per epoch, floor to multiple of FRAMES_PER_INSTRUCTION
    window_size_raw = int(n_train * EPOCH_WINDOW_FRACTION)
    window_size     = (window_size_raw // FRAMES_PER_INSTRUCTION) * FRAMES_PER_INSTRUCTION
    slide_step      = int(n_train * EPOCH_SLIDE_FRACTION)

    print("\n" + "=" * 60)
    print(f"  DATASET SPLIT  (90% train / 10% val, sliding-window epochs)")
    print(f"  Total sequences  : {n_total:,}")
    print(f"  Training pool    : {n_train:,}  (indices 0 .. {n_train-1})")
    print(f"  Validation set   : {n_val:,}   (indices {n_train} .. {n_total-1})")
    print(f"  Window per epoch : {window_size:,} seqs (floored to x{FRAMES_PER_INSTRUCTION})")
    print(f"  Slide per epoch  : {slide_step:,} seqs (~5%)")
    print(f"  Epochs           : {EPOCHS}")
    print("=" * 60 + "\n")

    wandb.config.update({
        "n_train": n_train, "n_val": n_val, "n_total": n_total,
        "epoch_window_size": window_size, "epoch_slide_step": slide_step,
    })

    # num_workers=0: the dataset pre-loads 120k frames into the main process RAM.
    # Forking workers would clone that entire cache into each worker process,
    # causing a multi-minute startup hang and OOM risk.
    # pin_memory still works with num_workers=0 on CUDA.
    loader_kw = dict(batch_size=1, num_workers=0,
                     pin_memory=(DEVICE.type == "cuda"))

    # train_loader is NOT used directly in the epoch loop.
    # Each epoch builds its own epoch_loader from a fresh random Subset of train_ds.
    # This gives different 10k samples every epoch (full dataset seen over ~10 epochs).
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ── Model — all params unfrozen from epoch 1 (Change 2) ─────────────────
    model     = PHGNNPhase1(n_walls=4).to(DEVICE)
    # Fused Adam utilises CUDA more efficiently when available
    try:
        optimizer = optim.Adam(model.parameters(), lr=LR, fused=(DEVICE.type == "cuda"))
    except TypeError:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler    = StepLR(optimizer, step_size=STEP_LR_STEP, gamma=STEP_LR_GAMMA)
    wall_normals = WALL_NORMALS.to(DEVICE)

    history = {
        "total": [], "state": [], "R": [],
        "val_pos_mse": [], "val_vel_mse": [],
        "R_norm": [], "R_eigmin": [],
        "epoch_time_s": [],
    }

    best_val_pos = float("inf")

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch", dynamic_ncols=True, leave=True, file=sys.stderr):
        t_epoch_start = time.perf_counter()         # --- Change 8: timer start

        # ── Train ──────────────────────────────────────────────────────────
        # ── Sliding-window epoch sampling ──────────────────────────────────
        # Window starts at slide_step*(epoch-1), wraps around the training pool.
        # Size is already a multiple of FRAMES_PER_INSTRUCTION so every
        # instruction set (motion sequence) is complete — no half-sequences.
        start = (slide_step * (epoch - 1)) % n_train
        end   = start + window_size
        if end <= n_train:
            epoch_idx = list(range(start, end))
        else:
            # Wrap around the training pool
            epoch_idx = list(range(start, n_train)) + list(range(0, end - n_train))
        # Shuffle within the window so the model doesn't see loops in order
        epoch_rng = torch.randperm(len(epoch_idx)).tolist()
        epoch_idx = [epoch_idx[i] for i in epoch_rng]
        epoch_subset = Subset(train_ds, epoch_idx)
        epoch_loader = DataLoader(epoch_subset, **loader_kw, shuffle=False)  # already shuffled

        model.train()
        ep_tot = ep_st = ep_rp = 0.0
        n_batches = 0
        batch_buf = []

        for sample in epoch_loader:
            # sample = (x_seq, arm_states, arm_us, arm_pos_nps, phase)
            # Each element is a list of ROLLOUT_STEPS+1 (or ROLLOUT_STEPS) items,
            # wrapped in a batch dim of 1 by the DataLoader — unpack accordingly.
            x_seq_batch, arm_states_batch, arm_us_batch, arm_pos_nps_batch, phase_batch = sample

            # Unpack the single-item batch (batch_size=1 in loader, we accumulate manually)
            x_seq      = [x_seq_batch[k][0]      for k in range(ROLLOUT_STEPS + 1)]
            arm_states = [arm_states_batch[k][0]  for k in range(ROLLOUT_STEPS)]
            arm_us     = [arm_us_batch[k][0]      for k in range(ROLLOUT_STEPS)]
            arm_pos_nps = [arm_pos_nps_batch[k][0].numpy().astype(np.float64)
                           for k in range(ROLLOUT_STEPS)]
            ph = phase_batch[0]

            batch_buf.append((x_seq, arm_states, arm_us, arm_pos_nps, ph))
            if len(batch_buf) < BATCH_SIZE:
                continue

            # ── TBPTT: backward once per rollout step to cap VRAM ─────────────
            # Verlet×3 sub-steps + retain_graph=True = 6 graph fragments per step.
            # 5 steps × 6 fragments held together = OOM on RTX 3080 (10 GB).
            # Fix: call backward() after EACH step — frees that step's graph
            # immediately, keeping only 6 fragments live at any time.
            # Gradients accumulate in .grad buffers across steps; one optimizer
            # step fires after all ROLLOUT_STEPS are done.
            optimizer.zero_grad(set_to_none=True)
            seq_ep_loss = 0.0

            for seq_item in batch_buf:
                x_seq_b, arm_states_b, arm_us_b, arm_pos_nps_b, ph_b = seq_item
                x_rolling = x_seq_b[0].to(DEVICE, non_blocking=True)

                for step in range(ROLLOUT_STEPS):
                    arm_sb  = arm_states_b[step].to(DEVICE, non_blocking=True)
                    arm_ub  = arm_us_b[step].to(DEVICE, non_blocking=True)
                    arm_np  = arm_pos_nps_b[step]
                    x_true  = x_seq_b[step + 1].to(DEVICE, non_blocking=True)

                    q = x_rolling[:, :3]
                    bb_src, bb_dst, br_src, bw_src_list, active = \
                        build_proximity_graph(q, arm_pos_np=arm_np)

                    x_rolling_sub, x_true_sub, bb_src, bb_dst, br_src, bw_src_list = \
                        subsample_to_active(
                            x_rolling, x_true, bb_src, bb_dst, br_src,
                            bw_src_list, active
                        )

                    bb_src      = bb_src.to(DEVICE, non_blocking=True)
                    bb_dst      = bb_dst.to(DEVICE, non_blocking=True)
                    br_src      = br_src.to(DEVICE, non_blocking=True)
                    bw_src_list = [w.to(DEVICE, non_blocking=True) for w in bw_src_list]

                    if br_src.numel() > 0:
                        arm_state_e = arm_sb.unsqueeze(0).expand(br_src.shape[0], -1)
                        arm_u_e     = arm_ub.unsqueeze(0).expand(br_src.shape[0], -1)
                    else:
                        arm_state_e = torch.empty(0, 48, device=DEVICE)
                        arm_u_e     = torch.empty(0, 3,  device=DEVICE)

                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        x_pred     = model(x_rolling_sub, bb_src, bb_dst, br_src,
                                           arm_state_e, arm_u_e, bw_src_list, wall_normals, DT)
                        R_mat      = model.R_param()
                        L, Ls, Lr  = total_loss(x_pred, x_true_sub, R_mat)

                    step_weight = 0.9 ** step
                    step_loss   = step_weight * L / (BATCH_SIZE * ROLLOUT_STEPS)

                    # Backward immediately — releases this step's graph from VRAM
                    scaler.scale(step_loss).backward()
                    torch.cuda.empty_cache()

                    ep_st      += Ls / (BATCH_SIZE * ROLLOUT_STEPS)
                    ep_rp      += Lr / (BATCH_SIZE * ROLLOUT_STEPS)
                    seq_ep_loss += step_loss.item()

                    # TBPTT truncation: detach state before next step
                    x_rolling = x_rolling.clone()
                    x_rolling[active] = x_pred.detach()

            # Single optimizer step — gradients accumulated across all steps
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            ep_tot   += seq_ep_loss
            n_batches += 1
            batch_buf.clear()

        scheduler.step()
        n_batches = max(n_batches, 1)

        # ── Validation — multi-step rollout (same as training loss) ────────
        # Rolls forward ROLLOUT_STEPS times per sequence, accumulates
        # pos_MSE and vel_MSE at each step, weighted by 0.9^step to match
        # the training loss weighting exactly.
        model.eval()
        val_pos_sum  = 0.0
        val_vel_sum  = 0.0
        val_loss_sum = 0.0
        val_n        = 0

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
            for sample in val_loader:
                x_seq_batch, arm_states_batch, arm_us_batch, arm_pos_nps_batch, _ = sample

                x_seq      = [x_seq_batch[k][0]     for k in range(ROLLOUT_STEPS + 1)]
                arm_states = [arm_states_batch[k][0] for k in range(ROLLOUT_STEPS)]
                arm_us     = [arm_us_batch[k][0]     for k in range(ROLLOUT_STEPS)]
                arm_pos_nps = [arm_pos_nps_batch[k][0].numpy().astype(np.float64)
                               for k in range(ROLLOUT_STEPS)]

                seq_pos = 0.0
                seq_vel = 0.0
                seq_loss = 0.0
                x_rolling = x_seq[0].to(DEVICE, non_blocking=True)

                for step in range(ROLLOUT_STEPS):
                    arm_sb  = arm_states[step].to(DEVICE, non_blocking=True)
                    arm_ub  = arm_us[step].to(DEVICE, non_blocking=True)
                    arm_np  = arm_pos_nps[step]
                    x_true  = x_seq[step + 1].to(DEVICE, non_blocking=True)

                    q = x_rolling[:, :3]
                    bb_src, bb_dst, br_src, bw_src_list, active =                         build_proximity_graph(q, arm_pos_np=arm_np)

                    x_rolling_sub, x_true_sub, bb_src, bb_dst, br_src, bw_src_list =                         subsample_to_active(
                            x_rolling, x_true, bb_src, bb_dst, br_src,
                            bw_src_list, active
                        )

                    bb_src      = bb_src.to(DEVICE)
                    bb_dst      = bb_dst.to(DEVICE)
                    br_src      = br_src.to(DEVICE)
                    bw_src_list = [w.to(DEVICE) for w in bw_src_list]

                    if br_src.numel() > 0:
                        arm_state_e = arm_sb.unsqueeze(0).expand(br_src.shape[0], -1)
                        arm_u_e     = arm_ub.unsqueeze(0).expand(br_src.shape[0], -1)
                    else:
                        arm_state_e = torch.empty(0, 48, device=DEVICE)
                        arm_u_e     = torch.empty(0, 3,  device=DEVICE)

                    x_pred = model(x_rolling_sub, bb_src, bb_dst, br_src,
                                   arm_state_e, arm_u_e, bw_src_list, wall_normals, DT)

                    step_weight = 0.9 ** step
                    pm, vm      = val_metrics(x_pred, x_true_sub)
                    R_mat       = model.R_param()
                    L, _, _     = total_loss(x_pred, x_true_sub, R_mat)

                    seq_pos  += step_weight * pm
                    seq_vel  += step_weight * vm
                    seq_loss += step_weight * L.item()

                    # Propagate prediction as next input (same as training)
                    x_rolling = x_rolling.clone()
                    x_rolling[active] = x_pred

                val_pos_sum  += seq_pos  / ROLLOUT_STEPS
                val_vel_sum  += seq_vel  / ROLLOUT_STEPS
                val_loss_sum += seq_loss / ROLLOUT_STEPS
                val_n        += 1

        val_n        = max(val_n, 1)
        val_pos_mse  = val_pos_sum  / val_n
        val_vel_mse  = val_vel_sum  / val_n
        val_loss     = val_loss_sum / val_n

        # ── R matrix stats ──────────────────────────────────────────────────
        with torch.no_grad():
            R_mat    = model.R_param()
            R_norm   = R_mat.float().norm().item()
            R_eigmin = torch.linalg.eigvalsh(R_mat.float()).min().item()

        # --- Change 8: timer end ---
        epoch_time = time.perf_counter() - t_epoch_start

        history["total"].append(ep_tot / n_batches)
        history["state"].append(ep_st  / n_batches)
        history["R"].append(ep_rp / n_batches)
        history["val_pos_mse"].append(val_pos_mse)
        history["val_vel_mse"].append(val_vel_mse)
        history["R_norm"].append(R_norm)
        history["R_eigmin"].append(R_eigmin)
        history["epoch_time_s"].append(epoch_time)

        # ── W&B logging (Change 5) ──────────────────────────────────────────
        log_dict = {
            "train/total_loss":  ep_tot / n_batches,
            "train/state_loss":  ep_st  / n_batches,
            "train/R_psd_loss":  ep_rp  / n_batches,
            "val/pos_mse":       val_pos_mse,
            "val/vel_mse":       val_vel_mse,
            "val/rollout_loss":  val_loss,
            "metrics/R_norm":    R_norm,
            "metrics/R_eigmin":  R_eigmin,
            "metrics/lr":        scheduler.get_last_lr()[0],
            "perf/epoch_time_s": epoch_time,
        }

        # W&B Table: log a per-epoch sample of predicted vs. true positions
        if val_n > 0 and (epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS):
            # Re-run one val sample for the table (lightweight)
            sample_iter = iter(val_loader)
            sv          = next(sample_iter)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
                x_t_s, x_t1_s, arm_s_s, arm_u_s, phase_s, anp_s = sv
                anp      = anp_s[0].numpy().astype(np.float64)
                xt_dev   = x_t_s[0].to(DEVICE)
                xt1_dev  = x_t1_s[0].to(DEVICE)
                armsb    = arm_s_s[0].to(DEVICE)
                armub    = arm_u_s[0].to(DEVICE)
                q        = xt_dev[:, :3]
                bbs, bbd, brs, bwl, act = build_proximity_graph(q, arm_pos_np=anp)
                xt_dev, xt1_dev, bbs, bbd, brs, bwl = subsample_to_active(
                    xt_dev, xt1_dev, bbs, bbd, brs, bwl, act)
                bbs = bbs.to(DEVICE); bbd = bbd.to(DEVICE)
                brs = brs.to(DEVICE)
                bwl = [w.to(DEVICE) for w in bwl]
                if brs.numel() > 0:
                    ase = armsb.unsqueeze(0).expand(brs.shape[0], -1)
                    aue = armub.unsqueeze(0).expand(brs.shape[0], -1)
                else:
                    ase = torch.empty(0, 48, device=DEVICE)
                    aue = torch.empty(0, 3,  device=DEVICE)
                xp = model(xt_dev, bbs, bbd, brs, ase, aue, bwl, wall_normals, DT)

            n_rows = min(10, xp.shape[0])
            tbl = wandb.Table(columns=[
                "particle_id", "phase",
                "pred_x", "pred_y", "pred_z",
                "true_x", "true_y", "true_z",
                "pred_vx", "pred_vy", "pred_vz",
                "true_vx", "true_vy", "true_vz",
            ])
            for pi in range(n_rows):
                pp  = xp[pi].cpu()
                tp  = xt1_dev[pi].cpu()
                vp  = (pp[3:6] / PARTICLE_MASS).tolist()
                vt  = (tp[3:6] / PARTICLE_MASS).tolist()
                tbl.add_data(
                    pi, phase_s[0],
                    *pp[:3].tolist(), *tp[:3].tolist(),
                    *vp, *vt,
                )
            log_dict["val/prediction_table"] = tbl

        wandb.log(log_dict, step=epoch)

        _epoch_msg = (
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Total={ep_tot/n_batches:.6f} State={ep_st/n_batches:.6f} "
            f"R={ep_rp/n_batches:.6f} | "
            f"Val pos_MSE={val_pos_mse:.6f} vel_MSE={val_vel_mse:.6f} rollout_loss={val_loss:.6f} | "
            f"R_norm={R_norm:.4f} R_eigmin={R_eigmin:.2e} | "
            f"lr={scheduler.get_last_lr()[0]:.5f} | "
            f"time={epoch_time:.1f}s"
        )
        tqdm.write(_epoch_msg, file=sys.stderr)
        sys.stderr.flush()

        # Save best model checkpoint
        if val_pos_mse < best_val_pos:
            best_val_pos = val_pos_mse
            torch.save(model.state_dict(), OUT_DIR / "model_best.pt")
            wandb.run.summary["best_val_pos_mse"] = best_val_pos
            wandb.run.summary["best_epoch"]        = epoch

    # ── Save final outputs ───────────────────────────────────────────────────
    torch.save(model.R_param().detach().cpu(), OUT_DIR / "R_phase1.pt")
    (OUT_DIR / "n_star.txt").write_text(str(N_STAR))
    json.dump(history, open(OUT_DIR / "loss_history.json", "w"), indent=2)

    # ── Loss curve ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    ep_range  = range(1, EPOCHS + 1)

    ax = axes[0]
    ax.plot(ep_range, history["total"], label="Total",   lw=2)
    ax.plot(ep_range, history["state"], label="L_state", lw=1.5, ls="--")
    ax.plot(ep_range, history["R"],     label="L_R",     lw=1.5, ls="-.")
    ax.set_ylabel("Train loss"); ax.set_title("Phase 1 Training Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ep_range, history["val_pos_mse"], label="Val pos MSE", lw=2)
    ax.plot(ep_range, history["val_vel_mse"], label="Val vel MSE", lw=2, ls="--")
    ax.set_ylabel("Validation MSE"); ax.set_title("Validation: Position & Velocity MSE")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ep_range, history["epoch_time_s"], label="Epoch time (s)", lw=1.5, color="purple")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Seconds"); ax.set_title("Per-Epoch Training Time")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=150)
    plt.close()
    wandb.log({"charts/loss_curve": wandb.Image(str(OUT_DIR / "loss_curve.png"))})

    # Upload final R matrix as W&B artifact
    artifact = wandb.Artifact("R_phase1", type="model")
    artifact.add_file(str(OUT_DIR / "R_phase1.pt"))
    run.log_artifact(artifact)
    run.finish()

    print(f"\nDone. Best val pos_MSE={best_val_pos:.6f}")
    print(f"Outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
