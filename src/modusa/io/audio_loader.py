#!/usr/bin/env python3

from modusa.io.base import ModusaIO
from modusa.signals.audio_signal import AudioSignal
from modusa.decorators import validate_args_type
from pathlib import Path
import tempfile
import numpy as np

class AudioLoader(ModusaIO):
	"""
	Loads audio from various sources like filepath, YouTube, etc.
	
	Note
	----
	- All `from_` methods return :class:`~modusa.signals.AudioSignal` instance.
	
	"""
	
	#--------Meta Information----------
	_name = "Audio Loader"
	_description = "Loads audio from various sources."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-05"
	#----------------------------------
	
	def __init__(self):
		super().__init__()

	@staticmethod
	@validate_args_type()
	def from_youtube(url: str, sr: int | None = None) -> "AudioSignal":
		"""
		Loads audio from youtube url using :class:`~modusa.io.YoutubeDownloader`,
		:class:`~modusa.io.AudioConverter` and `librosa`.

		.. code-block:: python
			
			from modusa.io import AudioSignalLoader
		
			# From youtube
			audio_signal = AudioSignalLoader.from_youtube(
				url="https://www.youtube.com/watch?v=lIpw9-Y_N0g", 
				sr=None
			)

		PARAMETERS
		----------
		url: str
			Link to the YouTube video.
		sr: int
			Sampling rate to load the audio in.
		
		Returns
		-------
		
		AudioSignal:
			`Audio signal` instance with loaded audio content from YouTube.
		"""

		from modusa.tools.youtube_downloader import YoutubeDownloader
		from modusa.tools.audio_converter import AudioConverter
		import librosa
		
		# Download the audio in temp directory using tempfile module
		with tempfile.TemporaryDirectory() as tmpdir:
			audio_fp: Path = YoutubeDownloader.download(url=url, content_type="audio", output_dir=Path(tmpdir))
			
			# Convert the audio to ".wav" form for loading
			wav_audio_fp: Path = AudioConverter.convert(inp_audio_fp=audio_fp, output_audio_fp=audio_fp.with_suffix(".wav"))
			
			# Load the audio in memory and return that
			audio_data, audio_sr = librosa.load(wav_audio_fp, sr=sr)
		
		audio = AudioSignal(data=audio_data, sr=audio_sr, title=audio_fp.stem)

		return audio
	
	@staticmethod
	@validate_args_type()
	def from_fp(fp: str | Path, sr: int | None = None) -> AudioSignal:
		"""
		Loads audio from a filepath using `librosa`.

		.. code-block:: python
			
			from modusa.io import AudioSignalLoader
			
			# From file
			audio_signal = AudioSignalLoader.from_fp(
				fp="path/to/audio.wav", 
				sr=None
			)

		Parameters
		----------
		fp: str | Path
			Local filepath of the audio.
		sr: int | None
			Sampling rate to load the audio in.
		
		Returns
		-------
		AudioSignal
			`Audio signal` instance with loaded audio content from filepath.
		
		"""
		import librosa
		import warnings
		warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")
		
		fp = Path(fp)
		y, sr = librosa.load(fp, sr=sr)
		
		audio_signal = AudioSignal(data=y, sr=sr, title=fp.name)
		
		return audio_signal
		
		