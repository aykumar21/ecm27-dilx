# DIL-X: ECM 2027 sim track plan (28 Nov → 30 Dec 2026, 12:00 SGT)

**Goal:** the highest *average* score over 20 automatic episodes (max 13 each). Only our **last** submission counts, and we get 5 scored submissions.

**Approach:** a hybrid that the organisers have confirmed is allowed:
- A **scripted stage machine** for the 6 stages
- **Learned perception** to find objects from the cameras
- Motion targets taken from the demos, relative to each object

A plain imitation-learning policy (ACT or diffusion) runs as a parallel baseline. Wiping is 6 of the 13 points, and stages only count in order, so reliability on the early stages matters most.

## Workstreams

| # | Workstream | Steps | Done when |
|---|---|---|---|
| A | **Infrastructure** | 1. Get an RTX GPU server (5090/4090/A6000/L40S, driver ≥ 580). 2. Run `team/setup/setup_gpu_server.sh`, then `smoke_test_sim.sh`. 3. Build a **local scorer**: run K episodes with randomisation and log the 13-point breakdown per stage. 4. Do a dry run of the submission path: Docker image → private HF repo → give `sgeaichallenge` access. | The example policy runs in the sim, and the scorer prints per-stage averages |
| B | **Data** | 1. Download HDF5 demos from batches 1–3 (selectively; about 2.9 GB each). 2. Re-render batch 2 (it has no images) and part of batch 1 with the background scene (`run_rerender.sh`, CPU physics). 3. Export the training set: 3 cameras, proprioception, and actions converted from 20-D to 19-D. 4. **Offline only:** label object poses from the sim state for perception training (allowed, but never used as a policy input). | Training set and labels on disk, with a loader in `team/training/` |
| C | **Perception** | 1. Estimate the poses of the chamber or handle, old part, box, sponge, replacement part and chamber faces from the head and wrist cameras. 2. Train with the labels from B4; test with objects moved slightly, as in the official runs. | Pose error small enough for grasping (target < 1–2 cm), robust to both backgrounds |
| D | **Skills and stage machine** | 1. Drive the base from its randomised start to the work pose. 2. Open the chamber (2 pts). 3. Grasp and extract the old part (1). 4. Place it in the box (1). 5. Wipe the 5 faces with the sponge (6), then put the sponge back on the table. 6. Place the replacement part (2). 7. Close the chamber (1). Each skill takes its arm targets from the demos, relative to the detected object, and converts world-frame EEF poses to the robot root frame. Add retries and timeouts within the 900 s episode. | Each stage succeeds reliably in the local scorer |
| E | **Imitation-learning baseline** | 1. Train ACT or diffusion policy on the HDF5 demos (publicly released pretrained weights are allowed). 2. Use it whole, or only for the stages where it beats the scripts (for example, wiping). | Average score compared with D on the same seeds |
| F | **Submission** | Package the best policy in `team/docker/`, submit, and read the feedback. **Never submit a weaker policy later**, since the last one counts. | 4–5 submissions in total |

## Timeline

| Week | Dates | Milestone |
|---|---|---|
| 1 | 28 Nov – 4 Dec | A1–A3 done; B1–B3 started; replay demos to confirm the 20-D to 19-D action conversion; **submission 1** (any policy that scores stage 1, to validate the pipeline) |
| 2 | 5 – 11 Dec | Perception v1 (C); stages 1–3 of the stage machine (D); IL baseline training (E) |
| 3 | 12 – 18 Dec | Wiping and stages 5–6; local average ≥ 7/13; **submission 2** |
| 4 | 19 – 25 Dec | Robustness against object shifts and base randomisation; **submission 3** by about 22 Dec (results by about 29 Dec) |
| 5 | 26 – 30 Dec | Fix only what submission 3's feedback shows; **final submission by 28 Dec**, and only if it beats the previous one locally |

## Rules that constrain the design (organiser rulings, see `ecm27-discord-log.md`)
- **Runtime inputs:** cameras and proprioception only. EEF pose is world frame relative to the env origin, quaternion xyzw. Simulator object poses and object positions from the scene files are forbidden.
- **Allowed:** object shapes and sizes from the assets; scripted primitives, motion planning and learned skills; public pretrained weights.
- **Physics:** CPU in evaluation, so use CPU locally too.
- **Data:** HDF5 is authoritative; the LeRobot copies are deprecated.
