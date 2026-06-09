"""Flip-specialized G1 motion tracking environment.

This keeps the generic G1MotionTracking defaults backward-compatible while
providing a dedicated registry task for flip-focused datasets/profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg

from .tracking import (
    X2MotionTrackingCfg,
    X2MotionTrackingEnv,
    PoseRandomization,
    VelocityRandomization,
)


def _zero_pose_randomization() -> PoseRandomization:
    return PoseRandomization(
        x=(0.0, 0.0),
        y=(0.0, 0.0),
        z=(0.0, 0.0),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )


def _zero_velocity_randomization() -> VelocityRandomization:
    return VelocityRandomization(
        x=(0.0, 0.0),
        y=(0.0, 0.0),
        z=(0.0, 0.0),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )


@dataclass
class X2FlipTrackingCfg(X2MotionTrackingCfg):
    """Config profile for flip tracking clips."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "agibotx2" / "scene_flat.xml")
        )
    )
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "agibotx2" / "flip_360_001__A304.npz"
    )
    pose_randomization: PoseRandomization = field(default_factory=_zero_pose_randomization)
    velocity_randomization: VelocityRandomization = field(
        default_factory=_zero_velocity_randomization
    )
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"] = "start"
    terminate_on_undesired_contacts: bool = True
    # Some flip clips include large anchor orientation deviations.
    anchor_ori_threshold: float = 1e9


@registry.envcfg("G1FlipTracking")
@dataclass
class X2FlipTrackingEnvCfg(X2FlipTrackingCfg):
    """Registered configuration for G1 flip tracking."""

    pass


@registry.env("G1FlipTracking", sim_backend="mujoco")
@registry.env("G1FlipTracking", sim_backend="motrix")
class X2FlipTrackingEnv(X2MotionTrackingEnv):
    """G1 flip-tracking environment implementation."""

    _cfg: X2FlipTrackingCfg


@dataclass
class X2WallFlipTrackingCfg(X2FlipTrackingCfg):
    """Config profile for wall-assisted G1 flip tracking clips."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "agibotx2" / "scene_flat_with_wall.xml")
        )
    )
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "agibotx2" / "flip_from_wall_104__A304.npz"
    )
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"] = "adaptive"
    anchor_pos_z_threshold: float = 0.5
    ee_body_pos_z_threshold: float = 0.5


@registry.envcfg("X2WallFlipTracking")
@dataclass
class X2WallFlipTrackingEnvCfg(X2WallFlipTrackingCfg):
    """Registered configuration for G1 wall flip tracking."""

    pass


@registry.env("X2WallFlipTracking", sim_backend="mujoco")
@registry.env("X2WallFlipTracking", sim_backend="motrix")
class X2WallFlipTrackingEnv(X2MotionTrackingEnv):
    """G1 wall flip-tracking environment implementation."""

    _cfg: X2WallFlipTrackingCfg


@dataclass
class X2ClimbTrackingCfg(X2MotionTrackingCfg):
    """Config profile for the climb_20_z_scale_1 motion clip."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "agibotx2" / "scene_climb_20_z_scale_1.xml")
        )
    )
    motion_file: str | list[str] = str(
        ASSETS_ROOT_PATH / "motions" / "agibotx2" / "climb_20_z_scale_1.0.npz"
    )
    max_episode_seconds: float = 15.0
    anchor_pos_z_threshold: float = 0.5
    ee_body_pos_z_threshold: float = 0.5


@registry.envcfg("G1ClimbTracking")
@dataclass
class X2ClimbTrackingEnvCfg(X2ClimbTrackingCfg):
    """Registered configuration for G1 box-climb motion tracking."""

    pass


@registry.env("X2ClimbTracking", sim_backend="mujoco")
@registry.env("X2ClimbTracking", sim_backend="motrix")
class X2ClimbTrackingEnv(X2MotionTrackingEnv):
    """G1 climb-tracking environment implementation."""

    _cfg: X2ClimbTrackingCfg
