"""Convert polaris_real2sim TAMP HDF5 episodes to a MolmoAct2 LeRobot v3.0 dataset.

Self-contained: uses this repo's lerobot v3.0 API (``lerobot.datasets``) plus the
bundled ``polaris_hdf5`` reader; no dependency on the polaris_real2sim root or
IsaacLab. The depth / visual-trace "reasoning" annotations are a MolmoAct v1
concept and are NOT needed here.

    cd third_party/molmoact2/lerobot
    uv sync --frozen --extra dataset --extra training
    uv run examples/polaris/convert_hdf5_to_lerobot.py \\
        --hdf5-root /path/to/tamp_hdf5_dataset \\
        --repo-id local/my_molmoact2_dataset \\
        --output-root /path/to/out_molmoact2_dataset \\
        --action-space joint --compute-stats

Then finetune in this same env via
``lerobot-train --policy.type=molmoact2 --dataset.repo_id=... --dataset.root=...``.
``--action-space joint`` -> joint proprio/action; ``ee`` -> EE-pose proprio +
delta-EE action. The embodiment/action-space is conveyed to MolmoAct2 at train
time via ``--policy.setup_type`` / ``--policy.control_mode`` prompt strings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from polaris_hdf5 import (
    CameraMapping,
    Hdf5DatasetReader,
    Hdf5EpisodeReader,
    build_state_action,
)

# Repo root is examples/polaris/<file> -> ../../ ; the quantile-stats script
# lives under src/lerobot/scripts/.
_LEROBOT_ROOT = Path(__file__).resolve().parents[2]

_CAMERA_FEATURES: dict[str, str] = {
    "exterior_1": "observation.images.image",
    "wrist": "observation.images.wrist_image",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=None, help="output dir (default: $HF_LEROBOT_HOME/<repo-id>)"
    )
    parser.add_argument("--action-space", choices=("joint", "ee"), default="joint")
    parser.add_argument("--camera-map", default=None, help="JSON {role: hdf5_image_key} override")
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--raw-gripper", action="store_true")
    parser.add_argument(
        "--compute-stats", action="store_true", help="run augment_dataset_quantile_stats.py afterwards"
    )
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def _build_features(
    sample: Hdf5EpisodeReader, mapping: CameraMapping, state_dim: int, action_dim: int
) -> dict[str, dict]:
    features: dict[str, dict] = {}
    for role, feature_name in _CAMERA_FEATURES.items():
        height, width, channels = sample.read_video(mapping.key_for(role)).shape[1:]
        features[feature_name] = {
            "dtype": "video",
            "shape": (int(height), int(width), int(channels)),
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (int(state_dim),),
        "names": {"axes": ["state"]},
    }
    features["action"] = {"dtype": "float32", "shape": (int(action_dim),), "names": {"axes": ["action"]}}
    return features


def main() -> None:
    args = _parse_args()
    normalize = not args.raw_gripper
    mapping = CameraMapping.from_json(args.camera_map)

    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    episode_paths = Hdf5DatasetReader(args.hdf5_root).episode_paths()
    if not episode_paths:
        raise FileNotFoundError(f"no episodes under {args.hdf5_root}")

    sample = Hdf5EpisodeReader(episode_paths[0])
    fps = args.fps if args.fps is not None else sample.fps
    sample_state, sample_action = build_state_action(
        sample, args.action_space, normalize_gripper_value=normalize
    )
    features = _build_features(sample, mapping, sample_state.shape[1], sample_action.shape[1])

    output_root = args.output_root if args.output_root is not None else HF_LEROBOT_HOME / args.repo_id
    if Path(output_root).exists():
        raise FileExistsError(f"output already exists: {output_root}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        robot_type=args.robot_type,
        fps=int(fps),
        features=features,
    )

    for path in episode_paths:
        reader = Hdf5EpisodeReader(path)
        videos = {feat: reader.read_video(mapping.key_for(role)) for role, feat in _CAMERA_FEATURES.items()}
        state, action = build_state_action(reader, args.action_space, normalize_gripper_value=normalize)
        for t in range(reader.num_frames):
            frame = {feat: videos[feat][t] for feat in videos}
            frame["observation.state"] = np.asarray(state[t], dtype=np.float32)
            frame["action"] = np.asarray(action[t], dtype=np.float32)
            frame["task"] = reader.instruction
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"[molmoact2-convert] wrote episode {path.stem} ({reader.num_frames} frames)", flush=True)

    dataset.finalize()
    if args.push_to_hub:
        dataset.push_to_hub(tags=["polaris", args.robot_type], private=False)

    print(f"[molmoact2-convert] done: {len(episode_paths)} episodes -> {output_root}", flush=True)

    if args.compute_stats:
        stats_script = _LEROBOT_ROOT / "src" / "lerobot" / "scripts" / "augment_dataset_quantile_stats.py"
        subprocess.run(
            [sys.executable, str(stats_script), "--repo-id", args.repo_id, "--root", str(output_root)],
            check=True,
        )


if __name__ == "__main__":
    main()
