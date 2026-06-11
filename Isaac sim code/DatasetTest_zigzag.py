import asyncio
import numpy as np
import omni.kit.app
import omni.usd
import csv, os
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World

# ── CONFIG ──────────────────────────────────────────────
NUM_LOOPS         = 50
MAX_SH0_DELTA     = 0.01
PARTICLE_SET_PATH = "/World/GranularMedium/ParticleSet"
SAVE_DIR          = "/home/rllab_msc_student/Vikram/assets/DatasetTest_zigzag"
# ────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

async def shovel_sequence():
    app   = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()

    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    for _ in range(30):
        await app.next_update_async()

    robot = SingleArticulation(prim_path="/World/Spot")
    robot.initialize()

    names = list(robot.dof_names)
    if robot.num_dof == 0:
        print("ERROR: DOFs still 0 — re-run DriveAPI step and restart Play")
        return

    def idx(name):
        try:    return names.index(name)
        except: print(f"WARNING: '{name}' not found"); return None

    sh0 = idx("arm0_sh0")
    sh1 = idx("arm0_sh1")
    el0 = idx("arm0_el0")
    el1 = idx("arm0_el1")
    wr0 = idx("arm0_wr0")
    wr1 = idx("arm0_wr1")
    ARM_INDICES = [sh0, sh1, el0, el1, wr0, wr1]
    ARM_NAMES   = ["sh0", "sh1", "el0", "el1", "wr0", "wr1"]

    rng      = np.random.default_rng(seed=42)
    base_pos = np.array(robot.get_joint_positions(), dtype=float)

    summary_rows         = []
    full_particle_frames = []
    full_scooper_frames  = []
    meta_rows            = []
    scooper_meta_rows    = []

    particle_set = stage.GetPrimAtPath(PARTICLE_SET_PATH)

    for loop_iter in range(NUM_LOOPS):

        # ── One independent x per non-flat_start keyframe ──
        # Range is [-MAX_SH0_DELTA, +MAX_SH0_DELTA]
        x_lower_and_turn  = rng.uniform(-MAX_SH0_DELTA, MAX_SH0_DELTA)
        x_move_forward    = rng.uniform(-MAX_SH0_DELTA, MAX_SH0_DELTA)
        x_turn            = rng.uniform(-MAX_SH0_DELTA, MAX_SH0_DELTA)
        x_retract         = rng.uniform(-MAX_SH0_DELTA, MAX_SH0_DELTA)
        x_turn_2          = rng.uniform(-MAX_SH0_DELTA, MAX_SH0_DELTA)
        x_move_forward_2  = rng.uniform(-MAX_SH0_DELTA, MAX_SH0_DELTA)
        x_rise_1          = rng.uniform(-MAX_SH0_DELTA, MAX_SH0_DELTA)

        print(f"\n{'='*60}")
        print(f" LOOP {loop_iter+1}/{NUM_LOOPS}")
        print(f"  lower_and_turn  x = {x_lower_and_turn:+.5f}")
        print(f"  move_forward    x = {x_move_forward:+.5f}")
        print(f"  turn            x = {x_turn:+.5f}")
        print(f"  retract         x = {x_retract:+.5f}")
        print(f"  turn_2          x = {x_turn_2:+.5f}")
        print(f"  move_forward_2  x = {x_move_forward_2:+.5f}")
        print(f"  rise_1          x = {x_rise_1:+.5f}")
        print(f"{'='*60}")

        keyframes = [
        # Phase             Steps   sh0    sh1     el0    el1     wr0     wr1
         ("flat_start",      60,  {sh0:  0.0,  sh1: -1.225, el0: 1.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
         ("Lower_and_turn",      60,  {sh0:  0.6,  sh1: -1, el0: 2.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
         ("Move_forward",      60,  {sh0:  0.6,  sh1: 0.05, el0: 0.9, el1: 0.375, wr0: -0.5, wr1: 3.14}),
         ("Turn",      60,  {sh0:  0,  sh1: 0.3, el0: 0.5, el1: 0.375, wr0: -0.5, wr1: 3.14}),
          ("Retract",      90,  {sh0:  0,  sh1: -1, el0: 2.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
          ("Turn_2",      60,  {sh0:  -0.4,  sh1: -1, el0: 2.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
         ("Move_forward_2",      60,  {sh0:  -0.4,  sh1: 0.05, el0: 0.9, el1: 0.375, wr0: -0.5, wr1: 3.14}),
         ("rise_1",      120,  {sh0:  -0.4,  sh1: -1.225, el0: 1.2, el1: 0.375, wr0: 0.75, wr1: 3.14}),
         ("flat_start_2",      60,  {sh0:  0.0,  sh1: -1.225, el0: 1.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
        ]

        # Map each phase name → its x value for logging in dataset
        phase_x_map = {
            "flat_start":      0.0,
            "lower_and_turn":  x_lower_and_turn,
            "move_forward":    x_move_forward,
            "turn":            x_turn,
            "retract":         x_retract,
            "turn_2":          x_turn_2,
            "move_forward_2":  x_move_forward_2,
            "rise_1":          x_rise_1,
        }

        prev = base_pos.copy()

        for phase_name, steps, targets in keyframes:
            print(f"  → {phase_name}")
            end = base_pos.copy()
            for ji, angle in targets.items():
                if ji is not None:
                    end[ji] = angle

            for step in range(steps):
                t   = step / steps
                t_s = t * t * (3.0 - 2.0 * t)
                interp = prev + (end - prev) * t_s
                robot.apply_action(ArticulationAction(joint_positions=interp))
                await app.next_update_async()

                # ── 1. Particle 6D state ───────────────────────
                p_pos = np.zeros((0, 3), dtype=np.float32)
                p_vel = np.zeros((0, 3), dtype=np.float32)
                if particle_set and particle_set.IsValid():
                    pts_attr = particle_set.GetAttribute("points")
                    vel_attr = particle_set.GetAttribute("velocities")
                    pts  = pts_attr.Get() if pts_attr else None
                    vels = vel_attr.Get() if vel_attr else None
                    if pts:
                        p_pos = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
                    if vels and len(vels) == len(pts):
                        p_vel = np.array([[v[0], v[1], v[2]] for v in vels], dtype=np.float32)

                n_p      = len(p_pos)
                mean_pos = p_pos.mean(axis=0) if n_p > 0 else np.zeros(3)
                mean_vel = p_vel.mean(axis=0) if n_p > 0 else np.zeros(3)
                std_pos  = p_pos.std(axis=0)  if n_p > 0 else np.zeros(3)
                std_vel  = p_vel.std(axis=0)  if n_p > 0 else np.zeros(3)

                if n_p > 0:
                    particle_6d = np.hstack([p_pos, p_vel])
                    full_particle_frames.append({"loop": loop_iter, "phase": phase_name, "step": step, "data": particle_6d})
                    meta_rows.append({"frame_idx": len(full_particle_frames)-1, "loop": loop_iter, "phase": phase_name, "step": step, "n_particles": n_p})

                # ── 2. Full scooper state ──────────────────────
                j_pos_all = np.array(robot.get_joint_positions(), dtype=np.float32)
                j_vel_all = np.array(robot.get_joint_velocities(), dtype=np.float32)
                arm_pos   = np.array([j_pos_all[i] for i in ARM_INDICES], dtype=np.float32)
                arm_vel   = np.array([j_vel_all[i] for i in ARM_INDICES], dtype=np.float32)

                arm_ft = np.zeros((6, 6), dtype=np.float32)
                try:
                    raw = robot.get_measured_joint_forces()
                    if raw is not None:
                        for k, ji in enumerate(ARM_INDICES):
                            if ji is not None and ji < len(raw):
                                arm_ft[k] = raw[ji]
                except Exception:
                    pass

                frame_idx_s = len(full_scooper_frames)
                full_scooper_frames.append({
                    "loop": loop_iter, "phase": phase_name, "step": step,
                    "arm_pos": arm_pos, "arm_vel": arm_vel, "arm_ft": arm_ft,
                })
                scooper_meta_rows.append({
                    "frame_idx": frame_idx_s,
                    "loop": loop_iter, "phase": phase_name, "step": step,
                    "delta_sh0": phase_x_map[phase_name],
                })

                # ── 3. Summary row ─────────────────────────────
                ee_ft = arm_ft[5]
                summary_rows.append({
                    "loop": loop_iter, "phase": phase_name, "step": step,
                    "delta_sh0": phase_x_map[phase_name],
                    "n_particles": n_p,
                    "part_mean_px": mean_pos[0], "part_mean_py": mean_pos[1], "part_mean_pz": mean_pos[2],
                    "part_mean_vx": mean_vel[0], "part_mean_vy": mean_vel[1], "part_mean_vz": mean_vel[2],
                    "part_std_px":  std_pos[0],  "part_std_py":  std_pos[1],  "part_std_pz":  std_pos[2],
                    "part_std_vx":  std_vel[0],  "part_std_vy":  std_vel[1],  "part_std_vz":  std_vel[2],
                    "sh0_pos": arm_pos[0], "sh1_pos": arm_pos[1], "el0_pos": arm_pos[2],
                    "el1_pos": arm_pos[3], "wr0_pos": arm_pos[4], "wr1_pos": arm_pos[5],
                    "sh0_vel": arm_vel[0], "sh1_vel": arm_vel[1], "el0_vel": arm_vel[2],
                    "el1_vel": arm_vel[3], "wr0_vel": arm_vel[4], "wr1_vel": arm_vel[5],
                    "sh0_fx": arm_ft[0,0], "sh0_fy": arm_ft[0,1], "sh0_fz": arm_ft[0,2],
                    "sh0_tx": arm_ft[0,3], "sh0_ty": arm_ft[0,4], "sh0_tz": arm_ft[0,5],
                    "sh1_fx": arm_ft[1,0], "sh1_fy": arm_ft[1,1], "sh1_fz": arm_ft[1,2],
                    "sh1_tx": arm_ft[1,3], "sh1_ty": arm_ft[1,4], "sh1_tz": arm_ft[1,5],
                    "el0_fx": arm_ft[2,0], "el0_fy": arm_ft[2,1], "el0_fz": arm_ft[2,2],
                    "el0_tx": arm_ft[2,3], "el0_ty": arm_ft[2,4], "el0_tz": arm_ft[2,5],
                    "el1_fx": arm_ft[3,0], "el1_fy": arm_ft[3,1], "el1_fz": arm_ft[3,2],
                    "el1_tx": arm_ft[3,3], "el1_ty": arm_ft[3,4], "el1_tz": arm_ft[3,5],
                    "wr0_fx": arm_ft[4,0], "wr0_fy": arm_ft[4,1], "wr0_fz": arm_ft[4,2],
                    "wr0_tx": arm_ft[4,3], "wr0_ty": arm_ft[4,4], "wr0_tz": arm_ft[4,5],
                    "wr1_fx": arm_ft[5,0], "wr1_fy": arm_ft[5,1], "wr1_fz": arm_ft[5,2],
                    "wr1_tx": arm_ft[5,3], "wr1_ty": arm_ft[5,4], "wr1_tz": arm_ft[5,5],
                })

            prev = end.copy()

        # ── Periodic save every 50 loops to avoid data loss ──
        if (loop_iter + 1) % 50 == 0:
            print(f"\n  [Checkpoint] Saving after loop {loop_iter+1}...")
            csv_path = os.path.join(SAVE_DIR, "summary.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"  ✓ Checkpoint saved — {len(summary_rows)} rows so far")

    # ── Save summary CSV ───────────────────────────────────
    csv_path = os.path.join(SAVE_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n✓ Summary CSV         →  {csv_path}  ({len(summary_rows)} rows)")

    # ── Save particle NPZ ──────────────────────────────────
    npz_path = os.path.join(SAVE_DIR, "particles_full.npz")
    np.savez_compressed(npz_path, **{f"frame_{i}": f["data"] for i, f in enumerate(full_particle_frames)})
    meta_path = os.path.join(SAVE_DIR, "particles_index.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"✓ Particles NPZ       →  {npz_path}  ({len(full_particle_frames)} frames)")

    # ── Save scooper NPZ ───────────────────────────────────
    scoop_npz_path = os.path.join(SAVE_DIR, "scooper_full.npz")
    scoop_npz_data = {}
    for i, fr in enumerate(full_scooper_frames):
        scoop_npz_data[f"frame_{i}_pos"] = fr["arm_pos"]
        scoop_npz_data[f"frame_{i}_vel"] = fr["arm_vel"]
        scoop_npz_data[f"frame_{i}_ft"]  = fr["arm_ft"]
    np.savez_compressed(scoop_npz_path, **scoop_npz_data)
    scoop_meta_path = os.path.join(SAVE_DIR, "scooper_index.csv")
    with open(scoop_meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scooper_meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scooper_meta_rows)
    print(f"✓ Scooper NPZ         →  {scoop_npz_path}  ({len(full_scooper_frames)} frames)")
    print(f"✓ Scooper index CSV   →  {scoop_meta_path}")
    print("\nAll done!")

asyncio.ensure_future(shovel_sequence())
print("Started!")
