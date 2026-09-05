from __future__ import annotations

import argparse
from pathlib import Path
import sys

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger

import importlib

# Metadata lives here so --list-stages works even before optional nuScenes/CARLA
# dependencies are installed. Stage modules are imported only when executed.
STAGES = [
    (0,  "foundation",      "Foundation / configuration",                         "stages.stage00_foundation"),
    (1,  "dataset",         "nuScenes sequence loading",                           "stages.stage01_dataset"),
    (2,  "preprocess",      "Sensor loading and camera preprocessing",              "stages.stage02_sensor_preprocess"),
    (3,  "geometry",        "Calibration and coordinate geometry",                  "stages.stage03_calibration_geometry"),
    (4,  "encoder",         "Camera encoder: ResNet-50 + feature pyramid",           "stages.stage04_camera_encoder"),
    (5,  "camera_bev",      "Camera features -> BEV (educational lift-splat)",       "stages.stage05_camera_bev"),
    (6,  "radar_bev",       "Radar points -> radar BEV encoder",                    "stages.stage06_radar_bev"),
    (7,  "spatial_fusion",  "Spatial BEV fusion: camera + radar",                   "stages.stage07_spatial_fusion"),
    (8,  "ego_motion",      "Localization / ego motion between temporal frames",    "stages.stage08_localization_ego_motion"),
    (9,  "temporal",        "Temporal BEV: ego-motion alignment + memory fusion",   "stages.stage09_temporal_bev"),
    (10, "detection",       "Object detection representation and targets",          "stages.stage10_detection"),
    (11, "tracking",        "Object tracking and track history",                    "stages.stage11_tracking"),
    (12, "occupancy",       "Dynamic occupancy / occupied-space raster",            "stages.stage12_occupancy"),
    (13, "prediction",      "Agent trajectory prediction",                          "stages.stage13_prediction"),
    (14, "map",             "HD-map / road context",                                "stages.stage14_map_context"),
    (15, "world_model",     "World-model assembly",                                 "stages.stage15_world_model"),
    (16, "behavior",        "Behavior planning",                                    "stages.stage16_behavior_planning"),
    (17, "motion_planning", "Motion planning / future ego trajectory",              "stages.stage17_motion_planning"),
    (18, "control",         "Vehicle control: trajectory -> steer / throttle / brake", "stages.stage18_vehicle_control"),
    (19, "safety",          "Runtime safety / fallback supervision",                "stages.stage19_safety_supervision"),
    (20, "closed_loop",     "Closed-loop integration boundary",                     "stages.stage20_closed_loop"),
]


ALIASES = {name: number for number, name, _title, _path in STAGES}
ALIASES.update({
    "bev": 5,
    "fusion": 7,
    "temporal_bev": 9,
    "planner": 17,
    "planning": 17,
    "all": 20,
})


def resolve_stage(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        key = value.strip().lower()
        if key not in ALIASES:
            raise argparse.ArgumentTypeError(
                f"Unknown stage '{value}'. Use --list-stages to see choices."
            )
        number = ALIASES[key]
    if not 0 <= number <= 20:
        raise argparse.ArgumentTypeError("Stage must be between 0 and 20.")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Teaching-oriented nuScenes autonomy pipeline. Run from Stage 00 up to "
            "the stage you want to study, with logs + plots at every boundary."
        )
    )
    p.add_argument("--dataroot", default="/data/sets/nuscenes", help="Root containing v1.0-mini, samples, sweeps and maps")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--stop-after", default="9", type=resolve_stage, help="Stage number/name to execute through, e.g. 4, encoder, temporal, all")
    p.add_argument("--list-stages", action="store_true")

    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--sample-index", type=int, default=-1, help="-1 chooses a safe middle sample")
    p.add_argument("--history", type=int, default=4, help="Number of keyframes including current")
    p.add_argument("--future", type=int, default=6, help="Future keyframes for prediction/planning supervision")

    p.add_argument("--image-height", type=int, default=256)
    p.add_argument("--image-width", type=int, default=448)
    p.add_argument("--bev-resolution", type=float, default=0.5)
    p.add_argument("--bev-range", type=float, default=50.0, help="Symmetric +/- meters in x and y")
    p.add_argument("--depth-bins", type=int, default=24)

    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--no-pretrained", action="store_true", help="Do not request torchvision pretrained ResNet weights")
    p.add_argument("--temporal-model", choices=["ema", "convgru"], default="ema")
    p.add_argument("--planner-mode", choices=["classical", "learned"], default="classical")
    p.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=2)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--backend", choices=["offline", "carla"], default="offline")
    p.add_argument("--carla-host", default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=2000)
    return p


def print_stage_list() -> None:
    print("\nAutonomy learning stages\n" + "-" * 76)
    for number, alias, title, _module_path in STAGES:
        print(f"{number:02d}  {alias:<18}  {title}")
    print()


def write_run_report(output_dir: Path, executed) -> None:
    lines = [
        "# Autonomy Learning Pipeline — Run Report",
        "",
        "This report is generated automatically. Open each stage folder to inspect its `summary.json` and images.",
        "",
        "| Stage | Folder | Visual outputs |",
        "|---:|---|---|",
    ]
    for number, module in executed:
        folder = output_dir / f"stage{number:02d}_{module.SHORT_NAME}"
        images = sorted(folder.glob("*.png"))
        image_names = ", ".join(f"`{p.name}`" for p in images) if images else "—"
        lines.append(f"| {number:02d} | `{folder.name}/` | {image_names} |")
    lines.extend([
        "",
        "## How to study a stage",
        "",
        "1. Read the corresponding `stages/stageXX_*.py` file.",
        "2. Run with `--stop-after XX --verbose 3`.",
        "3. Compare the console explanation with `summary.json`.",
        "4. Open the generated PNG(s) before progressing to the next stage.",
        "",
    ])
    (output_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.list_stages:
        print_stage_list()
        return 0

    bev_range = float(args.bev_range)
    cfg = PipelineConfig(
        dataroot=args.dataroot,
        version=args.version,
        output_dir=args.output_dir,
        scene_index=args.scene_index,
        sample_index=args.sample_index,
        history_frames=args.history,
        future_frames=args.future,
        image_height=args.image_height,
        image_width=args.image_width,
        bev_x_min=-bev_range,
        bev_x_max=bev_range,
        bev_y_min=-bev_range,
        bev_y_max=bev_range,
        bev_resolution=args.bev_resolution,
        depth_bins=args.depth_bins,
        pretrained_backbone=not args.no_pretrained,
        temporal_model=args.temporal_model,
        planner_mode=args.planner_mode,
        verbose=args.verbose,
        save_plots=not args.no_plots,
        device=args.device,
        seed=args.seed,
        backend=args.backend,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
    )
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    log = LessonLogger(cfg.verbose)
    ctx = PipelineContext()

    print("\nAUTONOMY LEARNING PIPELINE")
    print(f"nuScenes version : {cfg.version}")
    print(f"Stop after       : Stage {args.stop_after:02d}")
    print(f"Temporal mode    : {cfg.temporal_model}")
    print(f"Planner mode     : {cfg.planner_mode}")
    print(f"Output directory : {cfg.output_path.resolve()}")

    executed = []
    try:
        for number, _alias, title, module_path in STAGES:
            if number > args.stop_after:
                break
            module = importlib.import_module(module_path)
            module.run(ctx, cfg, log)
            executed.append((number, module))
            write_run_report(cfg.output_path, executed)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        print("\n" + "!" * 76)
        stage_title = getattr(module, "STAGE_NAME", title) if "module" in locals() else title
        print(f"PIPELINE STOPPED AT/BEFORE STAGE {number:02d}: {stage_title}")
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print("Tip: rerun with --verbose 3 after fixing the reported dependency/data issue.")
        print("!" * 76)
        if cfg.verbose >= 3:
            raise
        return 1

    print("\n" + "=" * 76)
    print(f"DONE: executed through Stage {args.stop_after:02d}")
    print(f"Study report: {cfg.output_path / 'run_report.md'}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
