#!/usr/bin/env python3
"""Advanced audio processing with chunk-based transcription to avoid missing text."""

import sys
import os
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_audio_to_text_chunked(audio_file_path: str) -> str:
	"""Convert audio file to text using chunk-based processing to avoid missing text."""
	recognizer = sr.Recognizer()
	
	try:
		# First check if file exists and has content
		if not os.path.exists(audio_file_path):
			return f"[Error: File not found at {audio_file_path}]"
		
		file_size = os.path.getsize(audio_file_path)
		if file_size == 0:
			return "[Error: Audio file is empty]"
		
		logger.info(f"Processing audio file: {audio_file_path}")
		logger.info(f"File size: {file_size} bytes")
		
		# Load audio file with pydub
		logger.info("Loading audio file...")
		audio = AudioSegment.from_file(audio_file_path)
		logger.info(f"Audio loaded: duration={len(audio)/1000:.1f}s")
		
		# Convert to mono and standard sample rate
		audio = audio.set_channels(1)
		audio = audio.set_frame_rate(16000)
		
		# Split audio on silence
		logger.info("Splitting audio on silence...")
		chunks = split_on_silence(
			audio,
			min_silence_len=500,  # 500ms of silence
			silence_thresh=audio.dBFS - 14,  # Silence threshold
			keep_silence=500  # Keep 500ms of silence at start/end of chunks
		)
		
		if not chunks:
			# If no chunks found, use the whole audio
			logger.info("No silence detected, processing as single chunk")
			chunks = [audio]
		else:
			logger.info(f"Split into {len(chunks)} chunks")
		
		# Process each chunk
		full_text = []
		for i, chunk in enumerate(chunks):
			logger.info(f"Processing chunk {i+1}/{len(chunks)} (duration: {len(chunk)/1000:.1f}s)")
			
			# Export chunk to temporary WAV file
			with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
				chunk.export(tmp_wav.name, format='wav')
				tmp_wav_path = tmp_wav.name
			
			try:
				# Process the chunk
				with sr.AudioFile(tmp_wav_path) as source:
					audio_data = recognizer.record(source)
				
				# Transcribe the chunk
				try:
					result = recognizer.recognize_google(audio_data, show_all=True)
					
					if isinstance(result, dict) and 'alternative' in result:
						text = result['alternative'][0]['transcript']
					elif isinstance(result, str):
						text = result
					else:
						text = ""
					
					if text:
						full_text.append(text)
						logger.info(f"  Chunk {i+1} text: {text[:50]}...")
					
				except sr.UnknownValueError:
					logger.warning(f"  Chunk {i+1}: Could not understand audio")
				except sr.RequestError as e:
					logger.error(f"  Chunk {i+1}: API error - {e}")
					
			finally:
				# Clean up temporary file
				if os.path.exists(tmp_wav_path):
					os.unlink(tmp_wav_path)
		
		# Combine all text
		final_text = " ".join(full_text)
		
		if final_text:
			logger.info("Transcription successful!")
			return final_text
		else:
			return "[Error: No text could be transcribed from the audio]"
			
	except Exception as e:
		logger.error(f"Failed to process audio file: {e}")
		import traceback
		traceback.print_exc()
		return f"[Error: {e}]"


def convert_audio_to_text_simple(audio_file_path: str) -> str:
	"""Simple conversion without chunking - process entire audio at once."""
	recognizer = sr.Recognizer()
	
	try:
		# Check if file exists
		if not os.path.exists(audio_file_path):
			return f"[Error: File not found at {audio_file_path}]"
		
		logger.info(f"Processing audio file (simple mode): {audio_file_path}")
		
		# Load and convert audio
		audio = AudioSegment.from_file(audio_file_path)
		audio = audio.set_channels(1).set_frame_rate(16000)
		
		# Add a small silence at the beginning to avoid cutting off
		silence = AudioSegment.silent(duration=200)  # 200ms silence
		audio = silence + audio
		
		# Export to temporary WAV
		with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
			audio.export(tmp_wav.name, format='wav')
			tmp_wav_path = tmp_wav.name
		
		try:
			# Process the WAV file
			with sr.AudioFile(tmp_wav_path) as source:
				# Record without ambient noise adjustment
				audio_data = recognizer.record(source)
			
			# Transcribe
			result = recognizer.recognize_google(audio_data, show_all=True)
			
			if isinstance(result, dict) and 'alternative' in result:
				text = result['alternative'][0]['transcript']
			elif isinstance(result, str):
				text = result
			else:
				text = str(result)
			
			return text
			
		finally:
			if os.path.exists(tmp_wav_path):
				os.unlink(tmp_wav_path)
				
	except Exception as e:
		logger.error(f"Failed to process audio file: {e}")
		return f"[Error: {e}]"


def main():
	"""Main function to test audio processing."""
	if len(sys.argv) < 2:
		print("Usage: python test_audio_processing_advanced.py <audio_file_path> [--chunked]")
		print("Example: python test_audio_processing_advanced.py /path/to/audio.mp3")
		print("         python test_audio_processing_advanced.py /path/to/audio.mp3 --chunked")
		sys.exit(1)
	
	audio_file_path = sys.argv[1]
	use_chunked = "--chunked" in sys.argv
	
	print(f"\n{'='*60}")
	print("Advanced Audio Processing Test")
	print(f"Mode: {'Chunked' if use_chunked else 'Simple'}")
	print(f"{'='*60}\n")
	
	# Check ffmpeg installation
	try:
		import subprocess
		result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
		if result.returncode == 0:
			print("✓ ffmpeg is installed")
		else:
			print("✗ ffmpeg is not properly installed")
	except FileNotFoundError:
		print("✗ ffmpeg is not installed - required for audio processing")
		print("  Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
		sys.exit(1)
	
	print(f"\nProcessing: {audio_file_path}\n")
	
	# Process the audio file
	if use_chunked:
		result = convert_audio_to_text_chunked(audio_file_path)
	else:
		result = convert_audio_to_text_simple(audio_file_path)
	
	print(f"\n{'='*60}")
	print("TRANSCRIPTION RESULT:")
	print(f"{'='*60}")
	print(result)
	print(f"{'='*60}\n")


if __name__ == "__main__":
	main()