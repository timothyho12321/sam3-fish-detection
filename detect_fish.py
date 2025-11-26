#!/usr/bin/env python
"""One-shot SAM 3 video inference entry point for fish detection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import save_masklet_video

DEFAULT_PROMPT = "gold fish with black stripes"
DEFAULT_VIDEO = Path("videos/Top_View_Normal_5min_close_lens_4.mp4")
DEFAULT_OUTPUT = Path("outputs/detect_fish/detections.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM 3 text-prompt video inference")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Path to MP4 video or JPEG folder",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt describing the objects of interest",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to anchor the first prompt",
    )
    parser.add_argument(
        "--propagation-direction",
        choices=["both", "forward", "backward"],
        default="both",
        help="Direction to track the segmentation",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to track",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for the exported visualization",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Overlay strength when rendering masks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the rendered MP4 overlay",
    )
    return parser.parse_args()


def ensure_video_exists(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no frames: {video_path}")
    return frames


def run_video_inference(
    video_path: Path,
    prompt: str,
    frame_index: int,
    propagation_direction: str,
    max_frames: int | None,
) -> dict[int, dict]:
    predictor = build_sam3_video_predictor()
    session = predictor.handle_request(
        request={"type": "start_session", "resource_path": str(video_path)}
    )
    session_id = session["session_id"]
    frame_outputs: dict[int, dict] = {}
    try:
        initial = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": prompt,
            }
        )
        frame_outputs[initial["frame_index"]] = initial["outputs"]
        stream_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": frame_index,
            "max_frame_num_to_track": max_frames,
        }
        for update in predictor.handle_stream_request(request=stream_request):
            if update.get("outputs") is None:
                continue
            frame_outputs[update["frame_index"]] = update["outputs"]
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    return frame_outputs


def summarize_outputs(outputs: dict[int, dict]) -> str:
    frames = len(outputs)
    objects = sum(len(out.get("out_obj_ids", [])) for out in outputs.values())
    return f"Tracked {objects} objects across {frames} frames"


def main() -> None:
    args = parse_args()
    ensure_video_exists(args.video_path)
    video_frames = load_video_frames(args.video_path)
    if args.frame_index >= len(video_frames):
        raise ValueError(
            f"frame_index {args.frame_index} exceeds video length {len(video_frames)}"
        )
    device_msg = (
        f"Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        if torch.cuda.is_available()
        else "CUDA is not available; inference requires a GPU."
    )
    print(device_msg)
    outputs = run_video_inference(
        video_path=args.video_path,
        prompt=args.prompt,
        frame_index=args.frame_index,
        propagation_direction=args.propagation_direction,
        max_frames=args.max_frames,
    )
    print(summarize_outputs(outputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_masklet_video(
        video_frames=video_frames,
        outputs=outputs,
        out_path=str(args.output),
        alpha=args.alpha,
        fps=args.fps,
    )
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
