import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video from an image and audio using SadTalker."
    )
    parser.add_argument("--source-image", required=True, help="Path to a portrait image or source video")
    parser.add_argument("--audio", required=True, help="Path to the driving audio file")
    parser.add_argument("--output", required=True, help="Path for the generated .mp4 file")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "checkpoints"),
        help="Path to the downloaded SadTalker checkpoints",
    )
    parser.add_argument(
        "--preprocess",
        choices=["crop", "extcrop", "resize", "full", "extfull"],
        default="crop",
        help="Image preprocessing mode",
    )
    parser.add_argument("--still", action="store_true", help="Reduce head movement")
    parser.add_argument(
        "--enhancer",
        choices=["gfpgan", "RestoreFormer"],
        help="Optional face enhancement model",
    )
    parser.add_argument("--size", type=int, choices=[256, 512], default=256, help="Face render resolution")
    parser.add_argument("--batch-size", type=int, default=2, help="Rendering batch size")
    parser.add_argument("--pose-style", type=int, default=0, help="Pose style index from 0 to 45")
    parser.add_argument("--expression-scale", type=float, default=1.0, help="Expression intensity")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU even if CUDA is available")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    args = parser.parse_args()

    args.source_image = Path(args.source_image).expanduser().resolve()
    args.audio = Path(args.audio).expanduser().resolve()
    args.output = Path(args.output).expanduser().resolve()
    args.checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()

    for label, path in (("Source image/video", args.source_image), ("Audio", args.audio)):
        if not path.is_file():
            parser.error("{} file does not exist: {}".format(label, path))
    if not args.checkpoint_dir.is_dir():
        parser.error(
            "Checkpoint directory does not exist: {}. Download the SadTalker models first.".format(
                args.checkpoint_dir
            )
        )
    if args.output.suffix.lower() != ".mp4":
        parser.error("Output path must have an .mp4 extension.")
    if args.output.exists() and not args.overwrite:
        parser.error(
            "Output already exists: {}. Use --overwrite to replace it.".format(args.output)
        )
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if not 0 <= args.pose_style < 46:
        parser.error("--pose-style must be between 0 and 45.")

    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(REPO_ROOT / "inference.py"),
        "--source_image",
        str(args.source_image),
        "--driven_audio",
        str(args.audio),
        "--checkpoint_dir",
        str(args.checkpoint_dir),
        "--preprocess",
        args.preprocess,
        "--size",
        str(args.size),
        "--batch_size",
        str(args.batch_size),
        "--pose_style",
        str(args.pose_style),
        "--expression_scale",
        str(args.expression_scale),
    ]
    if args.still:
        command.append("--still")
    if args.enhancer:
        command.extend(["--enhancer", args.enhancer])
    if args.cpu:
        command.append("--cpu")

    with tempfile.TemporaryDirectory(prefix="sadtalker-") as result_dir:
        command.extend(["--result_dir", result_dir])
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)

        generated_videos = list(Path(result_dir).glob("*.mp4"))
        if len(generated_videos) != 1:
            raise RuntimeError(
                "SadTalker completed without producing exactly one MP4 in its temporary output directory."
            )
        shutil.copy2(str(generated_videos[0]), str(args.output))

    print("Generated video: {}".format(args.output))


if __name__ == "__main__":
    main()
