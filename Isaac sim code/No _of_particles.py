import omni.usd
import numpy as np
from pxr import Vt, Gf

stage = omni.usd.get_context().get_stage()
particle_set = stage.GetPrimAtPath("/World/GranularMedium/ParticleSet")
points_attr = particle_set.GetAttribute("points")
vel_attr = particle_set.GetAttribute("velocities")

# --- CHANGE THIS: your desired particle count ---
TARGET_COUNT = 10000

current_points = list(points_attr.Get())
current_count = len(current_points)
print(f"Current count: {current_count} → Target: {TARGET_COUNT}")

pts = np.array([[p[0], p[1], p[2]] for p in current_points])

if TARGET_COUNT <= current_count:
    # DECREASE: evenly subsample
    indices = [int(i * current_count / TARGET_COUNT) for i in range(TARGET_COUNT)]
    new_pts = pts[indices]

else:
    # INCREASE: keep all existing + generate new ones by jittering existing points
    extra_needed = TARGET_COUNT - current_count
    # Randomly pick existing points and add small random offsets
    rng = np.random.default_rng(seed=42)
    base_indices = rng.integers(0, current_count, size=extra_needed)
    base_pts = pts[base_indices]

    # Estimate particle spacing from bounding box
    bbox_size = pts.max(axis=0) - pts.min(axis=0)
    jitter_scale = (bbox_size / current_count ** (1/3)) * 0.5
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=(extra_needed, 3))
    extra_pts = base_pts + jitter

    new_pts = np.vstack([pts, extra_pts])

# Write back
new_points_vt = Vt.Vec3fArray([Gf.Vec3f(*p) for p in new_pts])
points_attr.Set(new_points_vt)

# Match velocities — zero velocity for new particles
if vel_attr:
    zero_vel = Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * len(new_pts))
    vel_attr.Set(zero_vel)

print(f"New count: {len(new_pts)}")
print("Done — save with Ctrl+S")
