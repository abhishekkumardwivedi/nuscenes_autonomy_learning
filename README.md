# nuScenes Autonomy Learning Pipeline

A **teaching-first** autonomy project. It deliberately exposes every major transformation instead of hiding the pipeline behind one `train.py` command.

## The idea

Run from Stage 00 up to exactly the stage you are studying:

```text
00 Foundation
01 Dataset
02 Sensor preprocessing
03 Calibration / geometry
04 Camera encoder (ResNet-50 + FPN)
05 Camera -> BEV (lift / transform / splat)
06 Radar -> BEV
07 Camera + radar spatial fusion
08 Ego motion / localization
09 Temporal BEV
10 Detection targets + detection head
11 Tracking
12 Dynamic occupancy
13 Agent prediction
14 Map context
15 World model
16 Behavior planning
17 Motion planning / planning model
18 Vehicle control
19 Safety supervision
20 Closed-loop integration boundary
```

Stages 00-15 are primarily learned on **nuScenes Mini**. Stages 16-20 demonstrate the interfaces and transparent baselines; serious planning/closed-loop work should later move to **nuPlan / CARLA**.

## Why some stages show GT/baselines instead of neural predictions

A randomly initialized neural model can produce a tensor with the correct shape but meaningless content. This project therefore shows:

- **real nuScenes ground truth** where available,
- **deterministic, interpretable baselines** before training,
- and the **learnable model class** next to the baseline.

That makes it clear what is geometry/data processing and what still needs training.

## Install

nuScenes devkit v1.2.0 supports Python 3.9 and 3.12; Python 3.12 is the recommended clean environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If your RunPod already has a CUDA-enabled PyTorch build, keep it and install the remaining requirements rather than replacing it unnecessarily.

## Download nuScenes Mini

```bash
./download_nuscenes_mini.sh /workspace/data/nuscenes
```

The official mini archive is only for development/learning. Once the complete pipeline is understood, point the same code at `v1.0-trainval` for serious training.

## First commands

See the curriculum:

```bash
python main.py --list-stages
```

Only inspect dataset loading:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after 1 \
  --verbose 3
```

Go through camera tensors and geometry:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after geometry \
  --verbose 3
```

Reach the ResNet/FPN encoder:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after encoder \
  --verbose 2
```

Reach spatial camera+radar fusion:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after fusion
```

Reach temporal BEV:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after temporal \
  --history 4 \
  --temporal-model ema \
  --verbose 2
```

Study the ConvGRU architecture instead of the deterministic EMA output:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after temporal \
  --temporal-model convgru
```

Run through the world model:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after world_model
```

Run every offline stage:

```bash
python main.py \
  --dataroot /workspace/data/nuscenes \
  --stop-after all
```

## Output style

Every stage creates a directory such as:

```text
outputs/
  stage03_geometry/
    summary.json
    camera_geometry_topdown.png
  stage04_camera_encoder/
    summary.json
    current_p4_feature_montage.png
  stage05_camera_bev/
    summary.json
    current_camera_bev.png
  ...
  run_report.md
```

Study in this order:

1. read `stages/stageXX_*.py`,
2. execute through that stage with `--verbose 3`,
3. read its `summary.json`,
4. open its PNG output,
5. progress by one stage.

## Important limitations (intentional)

- Stage 05 uses **real nuScenes geometry** but a deterministic depth prior until a depth head is trained.
- Radar and spatial-fusion CNNs are instantiated but untrained; raw radar channels remain the trustworthy visual reference.
- Stage 10 uses GT boxes to teach detection targets; the detection head is present but untrained.
- Stage 11 uses nuScenes `instance_token` as an oracle identity so you can learn what tracking should recover.
- Stage 13 uses a constant-velocity prediction baseline and compares it with recorded future.
- Stage 14 uses semantic map context; nuScenes does not provide a complete mission route for our ego planner.
- Stage 16/17 provide transparent planning baselines plus a learnable trajectory head.
- nuScenes is recorded data, so Stage 20 cannot be causally closed loop. CARLA is needed for true action -> new observation feedback.

These are not shortcuts hidden from you; the logs call them out explicitly so that each later training step has a clear purpose.
