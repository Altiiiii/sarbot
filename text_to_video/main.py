"""Command line entry point for the text-to-video generator."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .pipeline import TextToVideoPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a simple AI-assisted video from text.")
    parser.add_argument("text", nargs="?", help="Text to convert into a video. If omitted, --input-file is required.")
    parser.add_argument("--input-file", type=Path, help="Path to a UTF-8 text file to read the script from.")
    parser.add_argument("--output", type=Path, default=Path("output.mp4"), help="Path for the generated video file.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Output video height in pixels.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the video.")
    parser.add_argument(
        "--seconds-per-segment",
        type=float,
        default=4.0,
        help="Duration of each generated scene in seconds.",
    )
    parser.add_argument("--font", type=str, default=None, help="Optional path to a .ttf font file for rendering text.")
    return parser


def read_text(text_argument: Optional[str], input_file: Optional[Path]) -> str:
    if text_argument:
        return text_argument
    if input_file and input_file.exists():
        return input_file.read_text(encoding="utf-8")
    raise SystemExit("Either text argument or --input-file must be provided.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    script_text = read_text(args.text, args.input_file)
    pipeline = TextToVideoPipeline(
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds_per_segment=args.seconds_per_segment,
        font_path=args.font,
    )
    pipeline.run(script_text, str(args.output))
    print(f"Video generated at {args.output.resolve()}")


if __name__ == "__main__":
    main()
