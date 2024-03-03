"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Atlas Operational Space Control
----------------
Operational Space Control of Atlas robot to demonstrate Jacobian and Mass Matrix Tensor APIs
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


# Parse arguments
args = gymutil.parse_arguments(description="Atlas Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

# Initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load atlas asset
asset_root = "/home/poplar/Desktop/IsaacGymEnvs/assets"
atlas_asset_file = "urdf/atlas/urdf/atlas.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01
asset_options.disable_gravity = True

print("Loading asset '%s' from '%s'" % (atlas_asset_file, asset_root))
atlas_asset = gym.load_asset(
    sim, asset_root, atlas_asset_file, asset_options)

# get joint limits and ranges for atlas
atlas_dof_props = gym.get_asset_dof_properties(atlas_asset)
atlas_lower_limits = atlas_dof_props['lower']
atlas_upper_limits = atlas_dof_props['upper']
atlas_ranges = atlas_upper_limits - atlas_lower_limits
atlas_num_dofs = len(atlas_dof_props)

# set default DOF states
default_dof_state = np.zeros(atlas_num_dofs, gymapi.DofState.dtype)

# set DOF control properties (except grippers)
atlas_dof_props["driveMode"][:7].fill(3)
# atlas_dof_props["stiffness"][:7].fill(0.0)
# atlas_dof_props["damping"][:7].fill(0.0)

# set DOF control properties for grippers
atlas_dof_props["driveMode"][7:].fill(3)
# atlas_dof_props["stiffness"][7:].fill(800.0)
# atlas_dof_props["damping"][7:].fill(40.0)

# Set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default atlas pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1.8)
pose.r = gymapi.Quat(0, 0, 0, 1)

print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []
init_pos_list = []
init_orn_list = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add atlas
    atlas_handle = gym.create_actor(env, atlas_asset, pose, "atlas", i, 1, 0)

    # Set initial DOF states
    gym.set_actor_dof_states(env, atlas_handle, default_dof_state, gymapi.STATE_ALL)

    # Set DOF control properties
    gym.set_actor_dof_properties(env, atlas_handle, atlas_dof_props)


# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API to access and control the physics simulation
gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):
    # Update jacobian and mass matrix
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torch.zeros(atlas_num_dofs, device='cuda')))

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    # gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
