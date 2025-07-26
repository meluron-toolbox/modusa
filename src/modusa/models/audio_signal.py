#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from .time_domain_signal import TimeDomainSignal
from modusa.tools.math_ops import MathOps
from typing import Self, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class AudioSignal(TimeDomainSignal):
	"""
	Represents a 1D audio signal within modusa framework.

	Note
	----
	- It is highly recommended to use  :class:`~modusa.io.AudioLoader` to instantiate an object of this class.
	- This class assumes audio is mono (1D numpy array).

	Parameters
	----------
	y : np.ndarray
		1D numpy array representing the audio signal.
	sr : int | None
		Sampling rate in Hz. Required if `t` is not provided.
	t0 : float, optional
		Starting time in seconds. Defaults to 0.0.
	title : str | None, optional
		Optional title for the signal. Defaults to `"Audio Signal"`.
	"""

	#--------Meta Information----------
	_name = "Audio Signal"
	_nickname = "Audio"
	_description = ""
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-04"
	#----------------------------------
	
	@validate_args_type()
	def __init__(
		self,
		data: np.ndarray | list | int | float,
		sr: float | int = 1.0,
		t0: float | int = 0.0,
		title: str = "Audio Signal"
	):

		super().__init__(data=data, data_label="Amplitude", sr=sr, t0=t0, time_label="Time (sec)", title=title) # Instantiating `TimeDomainSignal` class
		
	
	#-----------------------------------
	# Setter
	#-----------------------------------
		
	def set_meta_info(self, title: str | None = None) -> None:
		"""
		
		"""
		title = str(title) if title is not None else self.title
		sr = self.time_sax.sr
		t0 = self.time_sax.values[0]
		
		return self.__class__(data=self.data, sr=sr, t0=t0, title=title)
	
	#===================================
	
	#----------------------------------
	# Loaders
	#----------------------------------
	
	@classmethod
	def from_youtube(cls, url: str, sr: int | float = None):
		"""
		Loads audio from youtube at a given sr.
		The audio is deleted from the device
		after loading.
		
		.. code-block:: python
			
			import modusa as ms
			audio = ms.audio.from_youtube(
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
		import soundfile as sf
		from scipy.signal import resample
		import tempfile
		
		# Download the audio in temp directory using tempfile module
		with tempfile.TemporaryDirectory() as tmpdir:
			audio_fp: Path = YoutubeDownloader.download(url=url, content_type="audio", output_dir=Path(tmpdir))
			
			# Convert the audio to ".wav" form for loading
			wav_audio_fp: Path = AudioConverter.convert(inp_audio_fp=audio_fp, output_audio_fp=audio_fp.with_suffix(".wav"))
			
			# Load the audio in memory
			audio_data, audio_sr = sf.read(wav_audio_fp)
			
			# Convert to mono if it's multi-channel
			if audio_data.ndim > 1:
				audio_data = audio_data.mean(axis=1)
				
			# Resample if needed
			if sr is not None:
				if audio_sr != sr:
					n_samples = int(len(audio_data) * sr / audio_sr)
					audio_data = resample(audio_data, n_samples)
					audio_sr = sr
			
		audio = cls(data=audio_data, sr=audio_sr, title=audio_fp.stem)
		
		return audio
		
	@classmethod
	def from_filepath(cls, fp: str | Path, sr: int | float = None):
		import soundfile as sf
		from scipy.signal import resample
		from pathlib import Path
		
		fp = Path(fp)
		# Load the audio in memory
		audio_data, audio_sr = sf.read(fp)
		
		# Convert to mono if it's multi-channel
		if audio_data.ndim > 1:
			audio_data = audio_data.mean(axis=1)
			
		# Resample if needed
		if sr is not None:
			if audio_sr != sr:
				n_samples = int(len(audio_data) * sr / audio_sr)
				audio_data = resample(audio_data, n_samples)
				audio_sr = sr
		
		audio = cls(data=audio_data, sr=audio_sr, title=fp.stem)
		
		return audio
	
	#==================================	
	
	#----------------------------------
	# Methods
	#----------------------------------
			
	def play(self, regions: list[tuple[float, float], ...] | None = None, title: str | None = None):
		"""
		Play the audio signal inside a Jupyter Notebook.
	
		.. code-block:: python
	
			from modusa.generators import AudioSignalGenerator
			audio = AudioSignalGenerator.generate_example()
			audio.play(regions=[(0.0, 1.0), (2.0, 3.0)])
	
		Parameters
		----------
		regions : list[tuple[float, float, str] | None
			[(start_time, end_time, 'tag'), ...] pairs in seconds specifying the regions to play.
			If None, the entire signal is played.
		title : str or None, optional
			Optional title for the player interface. Defaults to the signal’s internal title.
	
		Returns
		-------
		IPython.display.Audio
			An interactive audio player widget for Jupyter environments.

		See Also
		--------
		:class:`~modusa.tools.audio_player.AudioPlayer`
		"""
		
		from modusa.tools.audio_player import AudioPlayer
		title = title or self.title
		audio_player = AudioPlayer.play(y=self.data, sr=self.sr, t0=self.t0, regions=regions, title=title)
		
		return audio_player