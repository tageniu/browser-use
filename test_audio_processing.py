"""Test audio processing and transcription functionality."""

import sys
import os
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_audio_to_text(audio_file_path: str) -> str:
	"""Convert audio file to text using speech recognition."""
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
		
		# Check file extension
		file_ext = Path(audio_file_path).suffix.lower()
		logger.info(f"File extension: {file_ext}")
		
		# If not WAV, convert to WAV first using pydub
		if file_ext != '.wav':
			logger.info(f"Converting {file_ext} to WAV format...")
			
			try:
				# Load audio file with pydub - let it auto-detect format
				audio = AudioSegment.from_file(audio_file_path)
				logger.info(f"Audio loaded successfully:")
				logger.info(f"  Duration: {len(audio)/1000:.1f} seconds")
				logger.info(f"  Channels: {audio.channels}")
				logger.info(f"  Frame rate: {audio.frame_rate}")
				logger.info(f"  Sample width: {audio.sample_width}")
				
				# Create temporary WAV file
				with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
					tmp_wav_path = tmp_wav.name
					# Export as WAV with standard parameters for speech recognition
					audio = audio.set_channels(1)  # Convert to mono
					audio = audio.set_frame_rate(16000)  # Standard sample rate for speech
					audio.export(tmp_wav_path, format='wav')
					logger.info(f"Exported to temporary WAV: {tmp_wav_path}")
				
				# Use the temporary WAV file
				audio_file_path = tmp_wav_path
				cleanup_temp = True
			except Exception as e:
				logger.error(f"Failed to convert audio format: {e}")
				return f"[Error: Could not convert audio format - {e}. Make sure ffmpeg is installed.]"
		else:
			cleanup_temp = False
		
		# Now process the WAV file
		try:
			with sr.AudioFile(audio_file_path) as source:
				logger.info("Reading audio file...")
				# Don't adjust for ambient noise on the whole audio - it can cut off the beginning
				# Instead, record the entire audio
				audio_data = recognizer.record(source)
				logger.info(f"Audio data recorded:")
				logger.info(f"  Sample rate: {audio_data.sample_rate}")
				logger.info(f"  Sample width: {audio_data.sample_width}")
				logger.info(f"  Duration: {len(audio_data.frame_data) / audio_data.sample_rate / audio_data.sample_width:.1f} seconds")
		except Exception as e:
			if cleanup_temp:
				os.unlink(audio_file_path)
			logger.error(f"Failed to read audio file: {e}")
			return f"[Error: Could not read audio file - {e}]"
			
		# Clean up temporary file if created
		if cleanup_temp:
			os.unlink(audio_file_path)
			logger.info("Cleaned up temporary file")
			
		# Try Google Speech Recognition first (free, no API key needed)
		try:
			logger.info("Sending audio to Google Speech Recognition...")
			# Use show_all=True to get all possible transcriptions
			result = recognizer.recognize_google(audio_data, show_all=True)
			
			if isinstance(result, dict) and 'alternative' in result:
				# Get the best transcription
				best_transcript = result['alternative'][0]['transcript']
				logger.info("Transcription successful!")
				logger.info(f"Number of alternatives: {len(result['alternative'])}")
				return best_transcript
			elif isinstance(result, str):
				logger.info("Transcription successful!")
				return result
			else:
				logger.warning(f"Unexpected result format: {type(result)}")
				return str(result)
		except sr.UnknownValueError:
			logger.warning("Google Speech Recognition could not understand the audio")
			return "[Error: Could not understand the audio - the audio might be unclear or contain no speech]"
		except sr.RequestError as e:
			logger.error(f"Google Speech Recognition request error: {e}")
			return f"[Error: API request error - {e}]"
		except Exception as e:
			logger.error(f"Unexpected error during recognition: {e}")
			return f"[Error: {e}]"
			
	except Exception as e:
		logger.error(f"Failed to process audio file: {e}")
		import traceback
		traceback.print_exc()
		return f"[Error: {e}]"


def main():
	"""Main function to test audio processing."""
	if len(sys.argv) != 2:
		print("Usage: python test_audio_processing.py <audio_file_path>")
		print("Example: python test_audio_processing.py /path/to/audio.mp3")
		sys.exit(1)
	
	audio_file_path = sys.argv[1]
	
	print(f"\n{'='*60}")
	print("Audio Processing Test")
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
		print("✗ ffmpeg is not installed - this is required for non-WAV audio files")
		print("  Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
	
	print(f"\nProcessing: {audio_file_path}\n")
	
	# Process the audio file
	result = convert_audio_to_text(audio_file_path)
	
	print(f"\n{'='*60}")
	print("TRANSCRIPTION RESULT:")
	print(f"{'='*60}")
	print(result)
	print(f"{'='*60}\n")


if __name__ == "__main__":
	main()