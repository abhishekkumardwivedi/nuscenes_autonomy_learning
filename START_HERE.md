# Start here

Do **not** run all 20 stages first. Learn by stopping at one boundary.

```bash
python main.py --list-stages
```

Recommended first pass:

```bash
# 1. Understand samples, timestamps and sensors.
python main.py --dataroot /workspace/data/nuscenes --stop-after 1 --verbose 3

# 2. Add real camera tensors + calibration geometry.
python main.py --dataroot /workspace/data/nuscenes --stop-after 3 --verbose 3

# 3. See ResNet-50/FPN feature maps.
python main.py --dataroot /workspace/data/nuscenes --stop-after 4 --verbose 2

# 4. See camera features become BEV.
python main.py --dataroot /workspace/data/nuscenes --stop-after 5 --verbose 2

# 5. Add radar and camera+radar fusion.
python main.py --dataroot /workspace/data/nuscenes --stop-after 7 --verbose 2

# 6. Align historical frames and build Temporal BEV.
python main.py --dataroot /workspace/data/nuscenes --stop-after 9 --verbose 3

# 7. Continue through detection, tracking, occupancy, prediction and world model.
python main.py --dataroot /workspace/data/nuscenes --stop-after 15 --verbose 2

# 8. Finally inspect planning, control and safety interfaces.
python main.py --dataroot /workspace/data/nuscenes --stop-after 20 --verbose 2
```

For each command, open `outputs/run_report.md`, then inspect that stage's `summary.json` and PNG images before moving on.
