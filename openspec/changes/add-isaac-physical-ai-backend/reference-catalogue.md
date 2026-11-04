# Physical AI reference catalogue — unified implementation scope

**Authority:** add-isaac-physical-ai-backend is the sole implementation queue for the Isaac, ordinary MuJoCo, Meta-World, MuJoCo Playground and named Physical AI examples. Earlier add-mujoco-reference-workflows and add-isaaclab-gpu-robotics documents are historical sources/evidence, not parallel implementation changes. Do not erase their task counts or completion records. No row in this catalogue is considered implemented merely because it appears here.

## Evidence labels (per row, per pinned release)

Each reference has independently recorded stages: **described**, **native constructed/stepped**, **actual NexuML learning/optimizer update**, **partial-budget resume**, **complete portable inference export/reload**, **fresh process/task evaluation and media**, **multi-seed quality/held-out evidence**, and where relevant **approved hardware evidence**. Reports SHALL record upstream version/commit, license and asset/model hash, control dimensions/joint order/units/rate/action limits, sensors/frames/timestamps, task resets, seed, compute budget, exact commands, measured outcomes and provenance. Do not mark a script-only, upstream-trained, actor-only or skipped GPU test as the full run.

Source verification precedes final naming; use the maintained task ID on the frozen release, not guessed registry strings or a replacement generic robot. Each item can deliver its independent milestone, but every required row remains open in whole-program acceptance until its own proof exists.

## Early handoff reference — supplemental, not a rewrite of I01–I22

| ID | Reference | Required outcome | Priority / status |
| --- | --- | --- | --- |
| F01 | SO-101 native reaching + **state-based behavior cloning** | Reproducible native expert episodes, real NexuML optimizer updates, source-pinned disjoint data, complete policy clean export/reload, fresh closed-loop learned success vs expert/untrained, repeated externally initialized bounded Python fit, validated numeric-model + physical-schema/metrics handoff, narrow PR gate | **P0 — planned, unverified** (tasks 0.1–0.8) |

F01 is an **additional early integration proof**, not an assertion that I03 reaching PPO, I07 teleoperation/corrections, I08 ACT/Diffusion or the original 236-item historical roadmap requirements have passed. Do not change the historical completion inventory. NexuFL federation/gRPC/Kubernetes evidence is owned by its separate integration change.

## Core reference matrix

| ID | Embodiment or task | Backend | Mandatory training/evaluation reference | Priority | Current status / owner |
| --- | --- | --- | --- | --- | --- |
| I01 | Isaac Cartpole | Isaac Lab canonical | continuous-action PPO, correctness, checkpoint/export/replay | P0 | historical verified EA; canonical port open; tasks 4–5 |
| I02 | Franka Reach | Isaac Lab canonical | joint-position PPO with true 7-DoF control | P0 | historical verified EA; canonical port open; tasks 4–5 |
| I03 | SO-101 Joint Reach | Isaac Lab canonical | five arm joints, fixed gripper, reached-goal/held outcome | P0 | historical verified EA; canonical port open; tasks 4–5 |
| I04 | SO-101 physical-contact vial placement | IsaacLabTutorial, Isaac Lab | six relative joint/gripper controls, PPO grasp/lift/place/release, home-reset score | P0 | planned; tasks 5.4–5.8 |
| I05 | SO-101 wrist-camera student | IsaacLabTutorial, Isaac Lab | privileged state teacher -> camera + proprioception distillation with deployment-only sensors | P0 | planned; tasks 8.5, 21.1 |
| I06 | SO-101 camera PPO from scratch | IsaacLabTutorial, Isaac Lab | native visual PPO with asymmetric critic, matched home reset/held-out tests | P1 | new; task 21.2 |
| I07 | SO-101 demonstrations/corrections | Isaac Capture/LeRobot data | approved teleoperation and LeRobotDataset v3 ingest; DAgger correction loop | P0/P1 | planned + expanded; tasks 7, 20 |
| I08 | SO-101 imitation | LeRobot + Isaac Lab | ACT, Diffusion and optional small flow policy from matching episodes -> Isaac replay | P0/P1 | ACT/Diffusion planned, flow added; tasks 8, 20 |
| I09 | SO-101 pretrained VLA | LeRobot + Isaac Lab | SmolVLA; selected GR00T N1.7; resource-gated pi0.5, matched processor/control contracts | P1 | planned; tasks 9, 21.3 |
| I10 | SO-101 adaptation loop | Isaac + approved arm | offline corrections, frozen old/new-task suite, parity, operator-approved real trial/rollback | P1/manual | planned; tasks 16, 20 |
| I11 | OpenArm single and bimanual manipulation | Maintained Isaac Lab tasks | one real unimanual RL task and one distinct bimanual IL/policy training task | P1 | planned; tasks 13.2, 21.4 |
| I12 | Unitree G1 exact 29-DoF | Isaac Lab/official Unitree | trained locomotion velocity tracking; separate humanoid manipulation/Mimic experience | P1 | planned; tasks 13.3, 21.4 |
| I13 | Unitree Go2 | Isaac Lab/Unitree | trained rough terrain/velocity tracking, falls and domain shift | P1 | planned; tasks 13.4 |
| I14 | Crazyflie 2.1+ | verified matching simulator | learned hover/goal tracking, correct Crazyflie actuator semantics | P1 | planned; task 13.5 |
| I15 | Separate PX4 SITL airframe | PX4/Gazebo | exported policy offboard control, timed loss/failsafe, classical comparison | P1 | planned; task 13.10 |
| I16 | TurtleBot3 Waffle Pi | compatible Gazebo/Isaac + ROS | learned waypoint/navigation vs Nav2, retained Gazebo/ros2_control proof | P1 | planned; tasks 12, 13.6 |
| I17 | UR10e or Rizon industrial insertion | Isaac Lab maintained | gear insertion with explicit pre-grasp/visual/controller assumptions, learned updates | P1 | planned; task 13.7 |
| I18 | UR5e/URSim | ROS 2/URSim | separate industrial command/inference and bounded learning reference if available | P1 | planned; task 13.7 |
| I19 | LumenPnP/industrial data | NexuML ordinary datasets | time-aligned vision offset/error/anomaly detection, read-only OPC UA; simulator feasibility report | P1 | planned; tasks 13.8, 14.6 |
| I20 | G1 humanoid locomanipulation or Isaac Mimic | Isaac Lab maintained | actual demonstration-generation plus imitation/closed-loop evaluation | P1 | existing generalized requirement; task 21.4 |
| I21 | Dexterous Allegro/Shadow or maintained equivalent | Isaac/DexSuite | one exact actuated hand object-reorientation/contact-rich training reference | P2 | new opt-in extension; task 21.6 |
| I22 | GR-1 manipulation or equivalent native example | Isaac Mimic | source-traceable teleoperation/synthetic demonstration and closed-loop learned policy | P2 | optional; task 21.6 |

## Alternative simulation and benchmarks

| ID | Reference | Backend | Required outcome | Priority | Owner |
| --- | --- | --- | --- | --- | --- |
| M01 | Humanoid-v5 PPO, current declared 45/17 observations/actions | native Gymnasium MuJoCo | native controls, train/resume/export/replay; headless EGL and OSMesa | P0 | 18.1–18.7 |
| M02 | Reacher-v5 PPO, declared 10/2 | native Gymnasium MuJoCo | same complete workflow without claiming real hardware identity | P0 | 18.1–18.7 |
| M03 | Meta-World MT1 reach-v3 | maintained Meta-World compatible lane | reach end-effector+gripper task, validated success-only, complete 50-goal matched-seed benchmark vs random | P0 | 18.1–18.10 |
| M04 | MuJoCo Playground CartpoleBalance | maintained MJWarp GPU | native vector correctness, actual NexuML update/resume/export/evaluate or explicitly blocked native interop | P1 | 19.1–19.6 |
| M05 | MuJoCo Playground Go2/G1 or maintained articulated task | MJWarp GPU | locomotion/manipulation benchmark, independent task/asset action checks | P1 | 19.1–19.6 |
| M06 | MuJoCo Playground PandaPickCubeCartesian or maintained vision task | MJWarp GPU batch renderer | true image-observation policy training/evaluation and rendering/profile proof | P1 | 19.1–19.6 |
| B01 | Isaac Lab-Arena | compatible maintained suite | frozen task/variation evaluation against one NexuML policy | P1 | 11 |
| B02 | LIBERO / LIBERO-plus | maintained LeRobot evaluation | native matching benchmark, processor/interface parity, per-task results | P1 | 21.5 |
| B03 | RoboTwin 2.0 | maintained LeRobot/robot benchmark | pinned subset with task-success protocol, no fake sim compatibility | P2 | 21.5 |

## Comparison protocols

- **SO-101 canonical home-pose vial test:** same six-action control semantics and same frozen physical task for compatible PPO state teacher, camera PPO, distilled student, ACT, Diffusion, flow-policy and VLA variants. State-only vs visual/language policies are reported with input-privilege differences prominently, not ranked as controlled identical-observation experiments. Release comparison requires separate matched-observation condition or interprets availability advantage as part of method difference.
- **MuJoCo MT1 reach:** goal/reset/horizon/protocol precisely frozen; three independent training seeds plus random-action reference; any-step success; never replace the complete native goal set by sampled seed episodes without verified goal IDs.
- **Locomotion:** exact G1/Go2 model and control, held-out friction/terrain, falls/tracking/joint-limit violations; native/upstream baselines only compare after matching task/action contracts.
- **Industrial:** task success with real contact/release/hold, not shaped reward or proximity; UR10e/Rizon pregrasp is not a complete pick-and-place claim. LumenPnP does not need an invented Isaac dynamic model.
- **Deployment:** offline recorded action parity -> simulator closed-loop -> calibrated supervised hardware trial; RL on real actuators remains independently gated. Never infer physical safety from policy reward.
- **Resource and compatibility:** each protocol requires a stable upstream/API/licensing/compute feasibility gate. Optional heavyweight variants stay blocked, not substituted by a toy network.

## Upstream source registry (re-resolve before implementation)

- Isaac Lab release tutorial and actual SO-101 tasks: https://github.com/isaac-sim/IsaacLabTutorial and https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/isaaclab/tutorial.html
- Isaac Lab maintained environments and Isaac Mimic: https://github.com/isaac-sim/IsaacLab
- OpenArm native tasks and models: https://github.com/enactic/openarm_isaac_lab
- Unitree exact training tasks: https://github.com/unitreerobotics/unitree_rl_lab
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- LeRobot maintained release and intervention/reward methods: https://github.com/huggingface/lerobot and https://huggingface.co/blog/lerobot-release-v060
- Meta-World protocol: https://metaworld.farama.org/evaluation/evaluation/

Names are stable intent labels, NOT claims that all combinations or current pinned versions have been natively verified.
