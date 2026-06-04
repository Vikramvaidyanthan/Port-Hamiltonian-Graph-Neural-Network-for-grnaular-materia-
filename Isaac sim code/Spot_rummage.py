import asyncio
import numpy as np
import omni.kit.app
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World

async def shovel_sequence():
    app = omni.kit.app.get_app()

    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    for _ in range(30):
        await app.next_update_async()

    robot = SingleArticulation(prim_path="/World/Spot")
    robot.initialize()

    names = list(robot.dof_names)
    print(f"DOF count: {robot.num_dof}")
    print(f"Joints: {names}")

    if robot.num_dof == 0:
        print("ERROR: DOFs still 0 — re-run Step 1 and restart Play")
        return

    def idx(name):
        try:    return names.index(name)
        except: print(f"WARNING: '{name}' not found"); return None

    # ── All 6 arm joints ──
    sh0 = idx("arm0_sh0")   # shoulder left/right  (body connection)
    sh1 = idx("arm0_sh1")   # shoulder up/down
    el0 = idx("arm0_el0")   # elbow bend
    el1 = idx("arm0_el1")   # elbow roll
    wr0 = idx("arm0_wr0")   # wrist tilt → controls scoop angle
    wr1 = idx("arm0_wr1")   # wrist roll

    # ── Resting angles from your USD (in radians) ──
    # sh0=-0.2°→0.0  sh1=-12.9°→-0.225  el0=47.9°→0.836
    # el1=21.5°→0.375  wr0=-11.0°→-0.192  wr1=39.3°→0.686
    REST = {
        sh0:  0.0,    # arm0_sh0 — at body, natural center
        sh1: -0.225,  # arm0_sh1 — slight downward
        el0:  0.836,  # arm0_el0 — elbow slightly bent
        el1:  0.375,  # arm0_el1 — forearm slight roll
        wr0: -0.192,  # arm0_wr0 — scoop near flat
        wr1:  0,  # arm0_wr1 — wrist natural roll
    }

    # ── Keyframes: all 6 joints explicitly controlled ──
    # steps = duration (60 steps ≈ 1 second at 60fps)
    keyframes = [
        # Phase             Steps   sh0    sh1     el0    el1     wr0     wr1
        ("flat_start",      60,  {sh0:  0.0,  sh1: -1.225, el0: 1.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
        ("Lower_and_turn",      60,  {sh0:  0.4,  sh1: -1, el0: 2.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
         ("Move_forward",      60,  {sh0:  0.4,  sh1: 0.05, el0: 0.9, el1: 0.375, wr0: -0.5, wr1: 3.14}),
         ("Turn",      60,  {sh0:  0,  sh1: 0.3, el0: 0.5, el1: 0.375, wr0: -0.5, wr1: 3.14}),
          ("Retract",      90,  {sh0:  0,  sh1: -1, el0: 2.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
          ("Turn_2",      60,  {sh0:  -0.4,  sh1: -1, el0: 2.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
         ("Move_forward_2",      60,  {sh0:  -0.4,  sh1: 0.05, el0: 0.9, el1: 0.375, wr0: -0.5, wr1: 3.14}),
         ("rise_1",      120,  {sh0:  -0.4,  sh1: -1.225, el0: 1.2, el1: 0.375, wr0: 0.75, wr1: 3.14}),
         ("flat_start_2",      60,  {sh0:  0.0,  sh1: -1.225, el0: 1.2, el1: 0.375, wr0: -0.192, wr1: 3.14}),
    ]

    # Capture full robot pose — legs are frozen at these values throughout
    base_pos = np.array(robot.get_joint_positions(), dtype=float)
    prev = base_pos.copy()

    for phase_name, steps, targets in keyframes:
        print(f"\n--- {phase_name.upper()} ---")
        end = base_pos.copy()  # start from base so legs never drift
        for joint_idx, angle in targets.items():
            if joint_idx is not None:
                end[joint_idx] = angle

        for step in range(steps):
            t = step / steps
            t_s = t * t * (3.0 - 2.0 * t)  # smoothstep easing
            interp = prev + (end - prev) * t_s
            robot.apply_action(ArticulationAction(joint_positions=interp))
            await app.next_update_async()

        prev = end.copy()

    print("\nSequence complete!")

asyncio.ensure_future(shovel_sequence())
print("Started!")
