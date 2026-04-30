from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import time
import numpy as np
import omni.usd
import omni.timeline
from pxr import UsdGeom
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


class PegReachEnv:
    def __init__(self, stage, world):
        self.stage = stage
        self.world = world

        self.robot = SingleArticulation("/World/open_manipulator_x")
        self.controller = self.robot.get_articulation_controller()

        self.target = np.array([0.2925, 0.0, 0.05], dtype=np.float32)

        self.hole_center = np.array([0.2925, 0.0, 0.005], dtype=np.float32)

    def get_peg_tip_position(self):
        prim = self.stage.GetPrimAtPath("/World/open_manipulator_x/gripper_left_link/AttachedPeg/peg_tip")

        xform = UsdGeom.Xformable(prim)
        world_tf = xform.ComputeLocalToWorldTransform(0.0)
        pos = world_tf.ExtractTranslation()

        return np.array([pos[0], pos[1], pos[2]], dtype=np.float32)

    def get_observation(self):
        q_full = np.array(self.robot.get_joint_positions(), dtype=np.float32)
        qd_full = np.array(self.robot.get_joint_velocities(), dtype=np.float32)

        arm_indices = [0, 1, 2, 3]

        q = q_full[arm_indices]
        qd = qd_full[arm_indices]

        peg_tip = self.get_peg_tip_position()
        rel = self.target - peg_tip

        obs = np.concatenate([q, qd, peg_tip, self.target, rel]).astype(np.float32)

        return obs


    def reset(self):
        q_full = np.array(self.robot.get_joint_positions(), dtype=np.float32)

        arm_indices = [0, 1, 2, 3]

        base_q = np.array([0.4, -0.7, 0.4, 0.2], dtype=np.float32)
        noise = np.random.uniform(-0.2, 0.2, size=4)

        q_full[arm_indices] = base_q + noise

        cmd = ArticulationAction(joint_positions=q_full)
        self.controller.apply_action(cmd)

        self.world.step()

        return self.get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -0.05, 0.05)

        q_full = np.array(self.robot.get_joint_positions(), dtype=np.float32)
        arm_indices = [0, 1, 2, 3]

        q_full[arm_indices] = q_full[arm_indices] + action

        cmd = ArticulationAction(joint_positions=q_full)
        self.controller.apply_action(cmd)

        self.world.step()

        peg_tip = self.get_peg_tip_position()

        # ---------------------------------------------
        # distances
        # ---------------------------------------------
        xy_dist =np.linalg.norm(peg_tip[:2] - self.hole_center[:2])
        z_dist = abs(peg_tip[2] - self.hole_center[2])

        # ---------------------------------------------
        # reward components
        # --------------------------------------------
        reward = 0.0

        # 1. encourage XY alignment
        reward -= xy_dist * 5.0

        # 2. encourage going down only when aligned
        if xy_dist < 0.02:
            reward -= z_dist * 2.0

        # 3. insertion bonus
        done = False
        if xy_dist < 0.01 and z_dist < 0.01:
            reward += 20.0
            done = True

        # ------------------------------------------
        info = {"xy_dist": float(xy_dist), "z_dist": float(z_dist)}

        obs = self.get_observation()
        return obs, reward, done, info


# Load stage
USD_PATH = "/home/sargye/isaac_projects/IsaacLab/assets/OMX/OMX/ok.usd"
omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

time.sleep(1.0)

# Start timeline
timeline = omni.timeline.get_timeline_interface()
timeline.play()
time.sleep(1.0)

# Create world
world = World()

# Create env
env = PegReachEnv(stage, world)

# Add robot to world
world.scene.add(env.robot)

# Initialize EVERYTHING
world.reset()

print("Env ready")

# ------------------------
# TEST LOOP
# ------------------------
obs = env.reset()
print("obs shape:", obs.shape)

for i in range(1500):
    world.step()
    action = np.random.uniform(-0.1, 0.1, size=4)

    obs, reward, done, info = env.step(action)

    print(f"step {i} | xy {info['xy_dist']:.3f} | z {info['z_dist']:.3f} | reward {reward:.3f}")

    time.sleep(1/60)

simulation_app.close()
