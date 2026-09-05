from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 17
STAGE_NAME = "Motion planning / future ego trajectory"
SHORT_NAME = "motion_planning"


class LearnedTrajectoryPlanner(nn.Module):
    """Simple learned planner head: temporal BEV + ego speed -> N future xy points."""

    def __init__(self, bev_channels: int, future_steps: int):
        super().__init__()
        self.future_steps = future_steps
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(bev_channels + 1, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, future_steps * 2),
        )

    def forward(self, bev: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        # bev [B,C,H,W], speed [B,1]
        x = self.pool(bev).flatten(1)
        out = self.mlp(torch.cat([x, speed], dim=1))
        return out.view(-1, self.future_steps, 2)


def _sample_occupancy(trajectory_xy: np.ndarray, occupancy: np.ndarray, cfg: PipelineConfig) -> float:
    risk = 0.0
    H, W = occupancy.shape
    for x, y in trajectory_xy:
        r = int((x - cfg.bev_x_min) / cfg.bev_resolution)
        c = int((y - cfg.bev_y_min) / cfg.bev_resolution)
        if r < 0 or r >= H or c < 0 or c >= W:
            risk += 4.0
        else:
            # Penalize a small neighborhood, not only one point-sized cell.
            r0, r1 = max(0, r-2), min(H, r+3)
            c0, c1 = max(0, c-2), min(W, c+3)
            risk += float(occupancy[r0:r1, c0:c1].max()) * 10.0
    return risk


def _classical_candidates(behavior: str, speed: float, times: np.ndarray) -> List[np.ndarray]:
    v = max(speed, 1.0)
    candidates = []
    if behavior == "STOP":
        # Constant-deceleration teaching trajectory, clipped at stop.
        a = max(v / max(times[-1], 1e-3), 1.0)
        x = np.maximum(v * times - 0.5 * a * times**2, 0.0)
        x = np.maximum.accumulate(x)
        candidates.append(np.column_stack([x, np.zeros_like(x)]))
        return candidates

    if behavior == "FOLLOW":
        v *= 0.65
    x = v * times
    for lateral_target in [0.0, 1.5, -1.5]:
        alpha = np.linspace(0.0, 1.0, len(times))
        y = lateral_target * (3 * alpha**2 - 2 * alpha**3)  # smoothstep
        candidates.append(np.column_stack([x, y]))
    return candidates


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("behavior", "current_ego_speed_mps", "dynamic_occupancy", "temporal_bev")

    behavior = ctx.get("behavior")
    speed = float(ctx.get("current_ego_speed_mps"))
    occupancy = ctx.get("dynamic_occupancy")
    times = ctx.get("future_times_sec")
    if times is None or len(times) == 0:
        times = np.arange(1, cfg.future_frames + 1, dtype=np.float64) * 0.5

    log.substage(17, 1, "Generate classical candidate trajectories")
    candidates = _classical_candidates(behavior, speed, times)
    log.info(f"Generated {len(candidates)} candidate trajectory/trajectories")

    log.substage(17, 2, "Score candidates for collision risk + lateral deviation")
    scores = []
    for i, tr in enumerate(candidates):
        collision = _sample_occupancy(tr, occupancy, cfg)
        lateral_cost = float(np.mean(np.abs(tr[:, 1]))) * 0.2
        score = collision + lateral_cost
        scores.append(score)
        log.detail(f"candidate {i}: collision_cost={collision:.2f}, lateral_cost={lateral_cost:.2f}, total={score:.2f}")
    best_idx = int(np.argmin(scores))
    classical_selected = candidates[best_idx]

    log.substage(17, 3, "Expose the learned planning-model alternative")
    bev = ctx.get("temporal_bev")
    learned_model = LearnedTrajectoryPlanner(bev.shape[0], len(times)).to(cfg.torch_device).eval()
    speed_tensor = torch.tensor([[speed]], dtype=bev.dtype, device=bev.device)
    with torch.no_grad():
        learned_raw = learned_model(bev.unsqueeze(0), speed_tensor)[0].detach().cpu().numpy()
    log.tensor("untrained learned trajectory [N,2]", learned_raw)
    log.detail("This neural planner must be trained on future ego trajectories before its numbers are meaningful.")

    if cfg.planner_mode.lower() == "learned":
        selected = learned_raw
        selection_mode = "learned UNTRAINED demo"
        log.info("WARNING: --planner-mode learned selected an untrained head for architecture demonstration only.")
    else:
        selected = classical_selected
        selection_mode = "classical transparent baseline"
    log.info(f"Selected planner mode = {selection_mode}")

    log.substage(17, 4, "Visualize candidate and selected motion")
    if cfg.save_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, tr in enumerate(candidates):
            ax.plot(tr[:, 1], tr[:, 0], "--", linewidth=1, label=f"candidate {i} score={scores[i]:.1f}")
        ax.plot(selected[:, 1], selected[:, 0], linewidth=2.5, marker="o", label="SELECTED")
        ax.scatter(0, 0, marker="x", s=80)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, min(cfg.bev_x_max, max(20, float(np.max(selected[:, 0]) + 5))))
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title(f"Stage 17: behavior={behavior} -> motion trajectory")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(stage_dir / "motion_candidates.png", dpi=150)
        plt.close(fig)

    values = {
        "motion_candidates": candidates,
        "motion_candidate_scores": scores,
        "selected_ego_trajectory": selected,
        "planner_selection_mode": selection_mode,
        "learned_planner_model": learned_model,
        "untrained_learned_trajectory": learned_raw,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("A high-level behavior has become a concrete future ego trajectory that a controller can follow.")
