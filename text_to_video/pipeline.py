"""Core pipeline for generating short videos from text prompts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


@dataclass
class StoryboardSegment:
    """A single segment in the generated storyboard."""

    text: str
    duration: float = 4.0
    background_color: tuple[int, int, int] = field(default_factory=lambda: (18, 18, 18))
    text_color: tuple[int, int, int] = field(default_factory=lambda: (245, 245, 245))

    def normalized(self) -> "StoryboardSegment":
        duration = max(self.duration, 0.5)
        return StoryboardSegment(
            text=self.text.strip(),
            duration=duration,
            background_color=self.background_color,
            text_color=self.text_color,
        )


class StoryboardBuilder:
    """Split an input text into visually pleasing storyboard segments."""

    def __init__(self, seconds_per_segment: float = 4.0, max_chars: int = 150) -> None:
        self.seconds_per_segment = seconds_per_segment
        self.max_chars = max_chars

    def build(self, text: str) -> List[StoryboardSegment]:
        sentences = self._split_sentences(text)
        segments: List[StoryboardSegment] = []
        for sentence in sentences:
            if not sentence:
                continue
            color = self._color_from_text(sentence)
            segments.append(
                StoryboardSegment(
                    text=sentence,
                    duration=self.seconds_per_segment,
                    background_color=color,
                    text_color=self._ideal_text_color(color),
                )
            )
        return [segment.normalized() for segment in segments]

    def _split_sentences(self, text: str) -> List[str]:
        raw = [chunk.strip() for chunk in text.replace("\n", " ").split(".")]
        sentences: List[str] = []
        buffer = ""
        for chunk in raw:
            if not chunk:
                continue
            candidate = (buffer + " " + chunk).strip() if buffer else chunk
            if len(candidate) <= self.max_chars:
                buffer = candidate
            else:
                if buffer:
                    sentences.append(buffer)
                buffer = chunk
        if buffer:
            sentences.append(buffer)
        return [sentence + "." if not sentence.endswith(".") else sentence for sentence in sentences]

    def _color_from_text(self, text: str) -> tuple[int, int, int]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return tuple(int(digest[i]) for i in range(3))

    def _ideal_text_color(self, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        r, g, b = rgb
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (20, 20, 20) if luminance > 128 else (245, 245, 245)


class FrameRenderer:
    """Render storyboard segments into RGB frames."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        font_path: Optional[str] = None,
        font_size: int = 48,
        margin: int = 60,
    ) -> None:
        self.width = width
        self.height = height
        self.margin = margin
        self.font = self._load_font(font_path, font_size)

    def render(self, segment: StoryboardSegment) -> np.ndarray:
        img = Image.new("RGB", (self.width, self.height), color=segment.background_color)
        draw = ImageDraw.Draw(img)
        wrapped = self._wrap_text(segment.text)
        current_y = self.margin
        for line in wrapped:
            w, h = draw.textsize(line, font=self.font)
            x = (self.width - w) / 2
            draw.text((x, current_y), line, fill=segment.text_color, font=self.font)
            current_y += h + 10
        return np.array(img)

    def _wrap_text(self, text: str) -> List[str]:
        draw = ImageDraw.Draw(Image.new("RGB", (self.width, self.height)))
        words = text.split()
        lines: List[str] = []
        current_line: List[str] = []
        for word in words:
            test_line = " ".join(current_line + [word]).strip()
            w, _ = draw.textsize(test_line, font=self.font)
            if w <= self.width - self.margin * 2 or not current_line:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def _load_font(self, font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
        try:
            return ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            return ImageFont.load_default()


class VideoAssembler:
    """Convert frames to a playable MP4 video using OpenCV."""

    def __init__(self, output_path: str, fps: int, frame_size: tuple[int, int]) -> None:
        self.output_path = output_path
        self.fps = fps
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = frame_size
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError("Video writer could not be initialized. Ensure the correct codecs are installed.")

    def write_frame(self, frame: np.ndarray, duration: float) -> None:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames_to_write = max(1, int(math.ceil(duration * (self.fps or 1))))
        for _ in range(frames_to_write):
            self.writer.write(frame_bgr)

    def close(self) -> None:
        self.writer.release()


class TextToVideoPipeline:
    """High-level orchestrator that converts raw text into a short narrated video."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        seconds_per_segment: float = 4.0,
        font_path: Optional[str] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.storyboard_builder = StoryboardBuilder(seconds_per_segment=seconds_per_segment)
        self.renderer = FrameRenderer(width=width, height=height, font_path=font_path)

    def run(self, text: str, output_path: str) -> List[StoryboardSegment]:
        segments = self.storyboard_builder.build(text)
        assembler = VideoAssembler(output_path=output_path, fps=self.fps, frame_size=(self.width, self.height))
        try:
            for segment in segments:
                frame = self.renderer.render(segment)
                assembler.write_frame(frame, duration=segment.duration)
        finally:
            assembler.close()
        return segments
