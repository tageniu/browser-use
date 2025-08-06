"""Video analysis coordinator for browser-use."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from browser_use.video.frame_extractor import VideoFrame, VideoFrameExtractor

logger = logging.getLogger(__name__)


class VideoAnalysisResult(BaseModel):
	"""Result of video analysis."""

	model_config = ConfigDict(extra='forbid')

	video_url: str | None = Field(default=None, description="URL of the analyzed video")
	video_path: str | None = Field(default=None, description="Path of the analyzed video")
	duration: float = Field(description="Video duration in seconds")
	frame_count: int = Field(description="Total number of frames analyzed")
	analysis: str = Field(description="Analysis result from the LLM")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional video metadata")


class VideoAnalyzer:
	"""Coordinates video analysis with frame extraction and LLM processing."""

	def __init__(self, frame_extractor: VideoFrameExtractor | None = None):
		"""
		Initialize the video analyzer.
		
		Args:
			frame_extractor: Frame extractor instance (creates default if None)
		"""
		self.frame_extractor = frame_extractor or VideoFrameExtractor()

	async def analyze_video_file(
		self,
		video_path: Path | str,
		analysis_prompt: str,
		llm_callback: Any
	) -> VideoAnalysisResult:
		"""
		Analyze a video file.
		
		Args:
			video_path: Path to the video file
			analysis_prompt: Prompt for the LLM analysis
			llm_callback: Async function that takes frames and prompt, returns analysis
			
		Returns:
			VideoAnalysisResult
		"""
		video_path = Path(video_path)
		
		# Extract frames
		frames = []
		async for frame in self.frame_extractor.extract_frames_from_file(video_path):
			frames.append(frame)

		if not frames:
			raise ValueError("No frames could be extracted from the video")

		# Prepare frame data for LLM
		frame_descriptions = []
		for frame in frames:
			frame_descriptions.append(f"Frame at {frame.timestamp:.1f}s (#{frame.frame_number})")

		# Call LLM with frames and prompt
		analysis = await llm_callback(frames, analysis_prompt, frame_descriptions)

		# Calculate video metadata
		duration = frames[-1].timestamp if frames else 0.0
		
		return VideoAnalysisResult(
			video_path=str(video_path),
			duration=duration,
			frame_count=len(frames),
			analysis=analysis,
			metadata={
				"sample_interval": self.frame_extractor.sample_interval,
				"max_frames": self.frame_extractor.max_frames
			}
		)

	async def analyze_video_url(
		self,
		video_url: str,
		analysis_prompt: str,
		llm_callback: Any,
		download_callback: Any
	) -> VideoAnalysisResult:
		"""
		Analyze a video from URL.
		
		Args:
			video_url: URL of the video
			analysis_prompt: Prompt for the LLM analysis
			llm_callback: Async function that takes frames and prompt, returns analysis
			download_callback: Async function that downloads video from URL
			
		Returns:
			VideoAnalysisResult
		"""
		# Download video to temp file
		with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
			tmp_path = Path(tmp_file.name)
			
		try:
			await download_callback(video_url, tmp_path)
			
			# Analyze the downloaded video
			result = await self.analyze_video_file(tmp_path, analysis_prompt, llm_callback)
			result.video_url = video_url
			result.video_path = None  # Don't expose temp path
			
			return result
			
		finally:
			# Clean up temp file
			if tmp_path.exists():
				tmp_path.unlink()

	def create_frame_summary(self, frames: list[VideoFrame]) -> str:
		"""
		Create a textual summary of the frames for context.
		
		Args:
			frames: List of VideoFrame objects
			
		Returns:
			Summary string
		"""
		if not frames:
			return "No frames available"

		summary_parts = [
			f"Video contains {len(frames)} analyzed frames",
			f"Duration: {frames[-1].timestamp:.1f} seconds",
			f"Frame timestamps: {', '.join(f'{f.timestamp:.1f}s' for f in frames)}"
		]
		
		return "\n".join(summary_parts)