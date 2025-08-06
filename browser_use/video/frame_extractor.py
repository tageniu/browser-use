"""Utility for extracting frames from videos."""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import AsyncGenerator

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class VideoFrame(BaseModel):
	"""Represents a single video frame."""

	model_config = ConfigDict(extra='forbid')

	timestamp: float = Field(description="Frame timestamp in seconds")
	frame_number: int = Field(description="Frame number in the video")
	image_base64: str = Field(description="Base64 encoded image")
	width: int = Field(description="Frame width in pixels")
	height: int = Field(description="Frame height in pixels")


class VideoFrameExtractor:
	"""Extracts frames from video files or URLs."""

	def __init__(self, max_frames: int = 10, sample_interval: float = 1.0):
		"""
		Initialize the frame extractor.
		
		Args:
			max_frames: Maximum number of frames to extract
			sample_interval: Interval between frames in seconds
		"""
		self.max_frames = max_frames
		self.sample_interval = sample_interval

	async def extract_frames_from_file(
		self, video_path: Path | str
	) -> AsyncGenerator[VideoFrame, None]:
		"""
		Extract frames from a video file.
		
		Args:
			video_path: Path to the video file
			
		Yields:
			VideoFrame objects
		"""
		video_path = Path(video_path)
		if not video_path.exists():
			raise FileNotFoundError(f"Video file not found: {video_path}")

		# Run frame extraction in executor to avoid blocking
		loop = asyncio.get_event_loop()
		frames = await loop.run_in_executor(None, self._extract_frames_sync, str(video_path))
		
		for frame in frames:
			yield frame

	def _extract_frames_sync(self, video_path: str) -> list[VideoFrame]:
		"""Synchronously extract frames from video."""
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise ValueError(f"Could not open video: {video_path}")

		frames = []
		fps = cap.get(cv2.CAP_PROP_FPS)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		duration = frame_count / fps if fps > 0 else 0

		# Calculate frame interval
		frame_interval = int(fps * self.sample_interval) if fps > 0 else 1
		
		# Ensure we don't exceed max frames
		total_samples = min(frame_count // frame_interval, self.max_frames)
		if total_samples == 0:
			total_samples = 1

		frame_number = 0
		sample_count = 0

		try:
			while sample_count < total_samples:
				ret, frame = cap.read()
				if not ret:
					break

				if frame_number % frame_interval == 0:
					# Convert frame to base64
					rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					pil_image = Image.fromarray(rgb_frame)
					
					# Resize if too large (max 1920x1080)
					max_width, max_height = 1920, 1080
					if pil_image.width > max_width or pil_image.height > max_height:
						pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
					
					# Convert to base64
					buffered = io.BytesIO()
					pil_image.save(buffered, format="PNG")
					img_base64 = base64.b64encode(buffered.getvalue()).decode()

					timestamp = frame_number / fps if fps > 0 else 0
					
					frames.append(VideoFrame(
						timestamp=timestamp,
						frame_number=frame_number,
						image_base64=img_base64,
						width=pil_image.width,
						height=pil_image.height
					))
					
					sample_count += 1

				frame_number += 1

		finally:
			cap.release()

		logger.info(f"Extracted {len(frames)} frames from video")
		return frames

	async def extract_key_frames(self, video_path: Path | str, max_frames: int = 5) -> list[VideoFrame]:
		"""
		Extract key frames evenly distributed throughout the video.
		
		Args:
			video_path: Path to the video file
			max_frames: Maximum number of frames to extract
			
		Returns:
			List of VideoFrame objects
		"""
		old_max = self.max_frames
		self.max_frames = max_frames
		
		frames = []
		async for frame in self.extract_frames_from_file(video_path):
			frames.append(frame)
		
		self.max_frames = old_max
		return frames