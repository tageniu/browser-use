"""Tests for video analysis functionality."""

import asyncio
import base64
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from browser_use.video import VideoFrameExtractor, VideoAnalyzer
from browser_use.video.frame_extractor import VideoFrame


@pytest.fixture
async def sample_video_file():
	"""Create a simple test video file."""
	with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
		video_path = Path(tmp_file.name)
	
	# Create a simple video with colored frames
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(str(video_path), fourcc, 1.0, (640, 480))
	
	# Create 5 frames with different colors
	colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
	for color in colors:
		frame = np.zeros((480, 640, 3), dtype=np.uint8)
		frame[:, :] = color
		out.write(frame)
	
	out.release()
	
	yield video_path
	
	# Cleanup
	if video_path.exists():
		video_path.unlink()


@pytest.mark.asyncio
async def test_frame_extraction(sample_video_file):
	"""Test basic frame extraction from video."""
	extractor = VideoFrameExtractor(max_frames=3, sample_interval=1.0)
	
	frames = []
	async for frame in extractor.extract_frames_from_file(sample_video_file):
		frames.append(frame)
	
	assert len(frames) == 3
	assert all(isinstance(f, VideoFrame) for f in frames)
	assert all(f.width == 640 and f.height == 480 for f in frames)
	assert all(f.timestamp >= 0 for f in frames)
	assert all(f.image_base64 for f in frames)


@pytest.mark.asyncio
async def test_key_frame_extraction(sample_video_file):
	"""Test key frame extraction."""
	extractor = VideoFrameExtractor()
	frames = await extractor.extract_key_frames(sample_video_file, max_frames=2)
	
	assert len(frames) == 2
	assert frames[0].frame_number < frames[1].frame_number


@pytest.mark.asyncio
async def test_video_analyzer(sample_video_file):
	"""Test video analyzer integration."""
	analyzer = VideoAnalyzer()
	
	# Mock LLM callback
	async def mock_llm_callback(frames, prompt, frame_descriptions):
		return f"Analyzed {len(frames)} frames for: {prompt}"
	
	result = await analyzer.analyze_video_file(
		sample_video_file,
		"Count the number of different colors",
		mock_llm_callback
	)
	
	assert result.video_path == str(sample_video_file)
	assert result.frame_count > 0
	assert result.duration > 0
	assert "Analyzed" in result.analysis


@pytest.mark.asyncio
async def test_frame_extractor_nonexistent_file():
	"""Test frame extractor with non-existent file."""
	extractor = VideoFrameExtractor()
	
	with pytest.raises(FileNotFoundError):
		async for _ in extractor.extract_frames_from_file("nonexistent.mp4"):
			pass


@pytest.mark.asyncio
async def test_frame_base64_encoding(sample_video_file):
	"""Test that frames are properly encoded as base64."""
	extractor = VideoFrameExtractor(max_frames=1)
	
	frames = []
	async for frame in extractor.extract_frames_from_file(sample_video_file):
		frames.append(frame)
	
	assert len(frames) == 1
	frame = frames[0]
	
	# Decode base64 and verify it's a valid image
	img_data = base64.b64decode(frame.image_base64)
	img = Image.open(io.BytesIO(img_data))
	
	assert img.size == (frame.width, frame.height)
	assert img.format == 'PNG'


@pytest.mark.asyncio
async def test_video_analyzer_with_url():
	"""Test video analyzer with URL (mocked download)."""
	analyzer = VideoAnalyzer()
	
	# Mock download callback
	async def mock_download(url, path):
		# Create a simple video file at the path
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(str(path), fourcc, 1.0, (320, 240))
		frame = np.zeros((240, 320, 3), dtype=np.uint8)
		frame[:, :] = (0, 255, 0)  # Green frame
		out.write(frame)
		out.release()
	
	# Mock LLM callback
	async def mock_llm_callback(frames, prompt, frame_descriptions):
		return "Video shows a green screen"
	
	result = await analyzer.analyze_video_url(
		"https://example.com/video.mp4",
		"What color is shown?",
		mock_llm_callback,
		mock_download
	)
	
	assert result.video_url == "https://example.com/video.mp4"
	assert result.video_path is None  # Should not expose temp path
	assert "green" in result.analysis.lower()


def test_frame_summary_creation():
	"""Test frame summary creation."""
	analyzer = VideoAnalyzer()
	
	frames = [
		VideoFrame(timestamp=0.0, frame_number=0, image_base64="", width=640, height=480),
		VideoFrame(timestamp=1.0, frame_number=30, image_base64="", width=640, height=480),
		VideoFrame(timestamp=2.0, frame_number=60, image_base64="", width=640, height=480),
	]
	
	summary = analyzer.create_frame_summary(frames)
	
	assert "3 analyzed frames" in summary
	assert "Duration: 2.0 seconds" in summary
	assert "0.0s, 1.0s, 2.0s" in summary