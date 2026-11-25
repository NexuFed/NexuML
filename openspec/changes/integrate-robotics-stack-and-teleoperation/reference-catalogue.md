# Selected Isaac reference catalogue

All ten exact rows are required delivered Isaac learning examples. Source locators are documentation/repository findings, not execution proof or authorization to download/redistribute. Resolve/pin actual native source, asset hashes and licensing. USD paths are relative to Isaac Sim 6.2 robot asset root; multiphysics paths use a different root. Reuse valid prior receipts; no invented task ID or substitute robot.

| ID | Required example | Native source / asset candidate | Minimum learned workflow |
| --- | --- | --- | --- |
| R01 | Cartpole | Existing `isaac-cartpole-ppo`, native `Isaac-Cartpole` | Continuous force balance, held-out return/balance and play. Not categorical Gym CartPole. |
| R02 | SO-101 new calibration | `RobotStudio/so101_new_calib/so101_new_calib.usd`, existing `NexuML-SO101-Joint-Reach` asset manifest | Five-arm-joint fixed-gripper reach, separate six-control physical-contact grasp/place with release/success per umbrella. Reuse BC/handoff evidence. |
| R03 | Standard Franka Panda | Existing `Isaac-Reach-Franka`; native Panda config, catalogue `FrankaRobotics/FrankaPanda/franka.usd` or task's pinned Panda instanceable asset | Default gripper, reach/tracking; optional native lift/pick task. Not FR3 substitution. |
| R04 | NVIDIA Kaya | `NVIDIA/Kaya/kaya.usd`, native Kaya Gamepad/holonomic controller reference | Learned waypoint/velocity tracking with lateral motion; compare controller baseline. Verify actual three-wheel control and passive rollers, not all 33 joints as actions. |
| R05 | Clearpath RidgebackFranka | `Clearpath/RidgebackFranka/ridgeback_franka.usd`, maintained Isaac Lab task where available | Base + Panda arm target/reach or contact placement, actual coordinated controls and task outcome, not fixed arm alone. |
| R06 | Boston Dynamics Spot **with arm** | `BostonDynamics/spot/spot_with_arm.usd`, not bare `spot.usd` | Verify arm/gripper articulation; qualified locomotion or supported stance plus learned arm reach/contact skill, demonstrate base/arm workflow in simulation. No unsupported whole-body claim. |
| R07 | Agibot A2D | `Agibot/A2D/A2D.usd` (documented mimic-joint API) | Bounded supported upper-body reach/manipulation; verify independent versus mimic joints, base/stance control and actual success. No assumption native learning task exists. |
| R08 | Unitree G1 EDU-target 29-DoF body | Official unitree_rl_lab / unitree_sim_isaaclab G1 29-DoF / unitree_mujoco; NVIDIA `Unitree/G1/g1.usd` candidate (43 documented articulation DoFs, not 29 body controls) | Exact body/hand mapping, learned locomotion velocity tracking plus independent manipulation/IL, held-out falls/tracking/contact; separate DDS/system and physical gates. 23-DoF optional, not substitute. |
| R09 | Bitcraze Crazyflie | `Bitcraze/Crazyflie/cf2x.usd`, native Crazyflie task; verify exact 2.1+ identity before hardware-equivalence claim | Learned hover/goal tracking, thrust/control timing and held-out disturbances. PX4 SITL is separate airframe. |
| R10 | LeKiwi V1 | Official URDF/meshes/joint map plus existing Lightwheel LeIsaac USD/Loft and input providers | Three-wheel holonomic base + SO-101; human teleop, record/convert, learned manipulation and full navigate/pick/transport/place system workflow. |

## LeKiwi files

- Official source, reviewed commit efa608d7ee5a495a4803b1d28cd0c955b4f1e033: https://github.com/SIGRobotics-UIUC/LeKiwi/tree/efa608d7ee5a495a4803b1d28cd0c955b4f1e033/URDF
- URDF: https://github.com/SIGRobotics-UIUC/LeKiwi/blob/efa608d7ee5a495a4803b1d28cd0c955b4f1e033/URDF/LeKiwi.urdf
- `URDF/meshes/` (STL) and `URDF/JOINT_NAMES.md`; official README/BOM/CAD: https://github.com/SIGRobotics-UIUC/LeKiwi
- Prefer existing native Isaac model/task: https://github.com/LightwheelAI/leisaac and https://lightwheelai.github.io/leisaac/resources/available_robots/
- Asset download instructions: https://lightwheelai.github.io/leisaac/docs/getting_started/installation/ ; upstream robot/scene assets: https://huggingface.co/LightwheelAI/leisaac_env/tree/main
- Native input and remote teleop: https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation/

Audit LeIsaac asset revision and pinned Lab 3.0/Sim 6.2 compatibility. If necessary use maintained URDF importer/asset transformer and official meshes, preserve inertial/collision/roller and joint calibration semantics. Validate actual control and relative paths; V1, not separate Dynamixel variant. No copied upstream framework or unauthorized assets in Git.

## Isaac asset references

- https://isaac-sim.github.io/IsaacLab/main/
- https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_manipulator.html
- https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_holonomic.html
- https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_mobile_manipulator.html
- https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_humanoid.html
- https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_aerial.html
- https://docs.isaacsim.omniverse.nvidia.com/6.2.0/introduction/menu_examples.html

Spot and A2D assets have 3D Content Sharing Agreement terms, Kaya NVIDIA terms; verify acquisition and redistribution rights individually. Reuse authorized resolver/cache, not binary vendoring.

## Official Unitree integration sources

- https://github.com/unitreerobotics/unitree_rl_lab
- https://github.com/unitreerobotics/unitree_sim_isaaclab (29-DoF body, Dex1/Dex3, DDS, whole-body tasks)
- https://github.com/unitreerobotics/unitree_mujoco
- https://github.com/unitreerobotics/unitree_sdk2
- https://github.com/unitreerobotics/xr_teleoperate
- https://github.com/unitreerobotics/unitree_lerobot
- https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_physical_ai/tutorials/tutorials.html

Exact EDU variant, body/finger joints, firmware, controller, SDK access, sensor/compute and safety equipment must be confirmed before hardware admission. Do not equate total NVIDIA G1 articulation with exact 29-DoF controller.

## Simulator coverage

- Isaac Lab/Sim: R01–R10, appropriate native physics (PhysX where asset requires it; Newton where qualified). Not every engine/viewer combination required.
- MuJoCo: foundation small matched task plus matching G1 native simulator/controller and compatible actor learning/evaluation for sim-to-sim and local-fit proof. Other robot ports optional if maintained/easy; LeKiwi MuJoCo optional.
- Gazebo/ROS 2: preserve independent ros2_control reference; selected mobile system workflow where compatible assets available, Isaac full-stack primary for LeKiwi.
- CARLA/PX4/other embodiments remain independent broader roadmap obligations.

## Evidence

Each example discoverable runnable scenario and native config, setup/train/evaluate/play instructions, known-action/native transition and reset proof, actual optimizer progress and changed finite parameters, resume, complete export/reload, held-out task outcome and playback/cleanup. Record exact asset/license/source, task, contract, runtime, seeds, actual budget, commands and measurements. Separate smoke/quality and representative native media proof. Asset-only, upstream-trained, scripted-only and skipped work do not qualify. Missing assets/license/task or failed learning remains explicit blocked/incomplete; no silent substitution. Physical trials only required on approved G1 under separate gate.
