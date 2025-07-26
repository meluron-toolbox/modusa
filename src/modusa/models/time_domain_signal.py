#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from .signal1D import Signal1D
from modusa.tools.math_ops import MathOps
import modusa as ms
from typing import Self, Any, Callable
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt

class TimeDomainSignal(Signal1D):
	"""
	Space to represent time domain signals.
	
	Note
	----
	- We only allow uniform time domain signal.

	Parameters
	----------
	data: np.ndarray | list | int | float
		- Data array
		- We convert any other types (list | int | float) to `np.ndarray` in the constructor.
	sr: float | int
		- Sampling rate of the signal.
		- Default is 1.0
		- We convert it into float.
	data_label: str
		- What does the data represent? 
		- e.g. "Amplitude (dB)".
		- We use this as ylabel for plots.
	time_label: str
		- Label for the time axis.
		- Default is "Time (sec)
	title: str
		- What does the signal represent?
		- e.g. "MySignal"
		- Default to "Time Domain Signal"
		- This is used as the title for plot.
	"""
	
	#--------Meta Information----------
	_name = "Time Domain Signal"
	_nickname = "TDS" # This is to be used in repr/str methods
	_description = "Space to represent uniform time domain signal."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-20"
	#----------------------------------
	
	@validate_args_type()
	def __init__(
		self,
		data: np.ndarray,
		sr: float,
		t0: float,
		data_label: str,
		time_label: str,
		title: str
	):
		
		assert data.ndim == 1
		sax = ms.sax.linear(n_points=data.shape[0], sr=sr, start=t0, label=time_label)
		sax = (sax, )
		super().__init__(data=data, data_label=data_label, sax=sax, title=title) # Instantiating `Signal1D` class
		self.__sr = sr
		
		assert self._x._is_uniform, "Only support Uniform Signal" # Can be removed later on for saving some extra computation
		
	#--------------------------------------
	# Properties (User Facing)
	#--------------------------------------
	
	#=======================================
	
	#----------------------
	# Properties (Hidden)
	#----------------------
	
	@immutable_property("Read only property.")
	def _t(self) -> np.ndarray:
		"""Timestamp array."""
		return self._x # Comes from 'Signal1D' properties

	@immutable_property("Read only property.")
	def _sr(self) -> float:
		"""Sampling rate of the signal."""
		return self.__sr
	
	@immutable_property("Create a new object instead.")
	def _t0(self) -> int:
		"""Start timestamp of the signal."""
		return self._t._values[0]
	
	@immutable_property("Create a new object instead.")
	def _duration(self) -> int:
		"""Duration of the signal."""
		return self._t._values[-1]
	
	#=======================================
	
	#-----------------------------------
	# Setter
	#-----------------------------------
	
	def set_meta_info(self, data_label: str | None = None, time_label: str | None = None, title: str | None = None) -> None:
		"""
		Set meta info about the signals.

		Parameters
		----------
		data_label: str
			- Label for the data. (y_label)
			- e.g. "Amplitude (dB)"
		time_label: str
			- Label for the time axis. (x_label)
			- e.g. "Time (sec)"
		title: str
			- Title for the signal
			- e.g. "Speech Signal"
		"""
		y = self._y
		y_label = str(data_label) if data_label is not None else self._y._label
		t_label = str(time_label) if time_label is not None else self._t._label
		title = str(title) if title is not None else self._title
		sr = self._sr
		t0 = self._t0
		
		return self.__class__(data=y, data_label=y_label, sr=sr, t0=t0, time_label=t_label, title=title)
	
	#===================================
	
	#-------------------------------
	# Tools
	#-------------------------------
		
	@validate_args_type()
	def translate_t(self, by_sample: int):
		"""
		Translates the signal along time axis.
		
		Note
		----
		- Negative indexing is allowed but just note that you might end up getting time < 0
		- For the time being, we are not putting checks on if the time is below 0


		.. code-block:: python
			
			import modusa as ms
			s1 = ms.tds([1, 2, 4, 4, 5, 3, 2, 1])
			ms.plot(s1, s1.translate_t(-1), s1.translate_t(3))
		
		Parameters
		----------
		by_sample: int
			By how many sample you would like to translate the signal.
		
		Returns
		-------
		AudioSignal
			Translated audio signal
		"""
		from modusa.models.audio_signal import AudioSignal
		assert self.__class__ in [TimeDomainSignal, AudioSignal]
		
		t0 = self._t0
		sr = self._sr
		y, y_label = self._y, self._y_label
		t, t_label = self._t._values, self._t._label
		title = self._title
		
		t0_new = t0 + (by_sample / sr)
		
		if self.__class__ == TimeDomainSignal:
			return self.__class__(data=y, data_label=y_label, sr=sr, t0=t0_new, time_label=t_label, title=title)
		elif self.__class__ == AudioSignal:
			return self.__class__(data=y, sr=sr, t0=t0_new, title=title)
		
	
	
	def crop(self, t_min: int | float | None = None, t_max: int | float | None = None) -> "AudioSignal":
		"""
		Crop the signal to a time range [t_min, t_max].

		.. code-block:: python

			import modusa as ms
			import numpy as np
			s1 = ms.tds(np.random.random(1000), sr=10)
			ms.plot(s1, s1.crop(5, 40), s1.crop(20), s1.crop(60, 80))

	
		Parameters
		----------
		t_min : float or None
			Inclusive lower time bound in second (other units). If None, no lower bound.
		t_max : float or None
			Exclusive upper time bound in second (other units). If None, no upper bound.
	
		Returns
		-------
		AudioSignal
			Cropped audio signal.
		"""
		from modusa.models.audio_signal import AudioSignal
		assert self.__class__ in [TimeDomainSignal, AudioSignal]
		
		
		# Get the time axis 
		t = self._t
		# Create a mask based on the t_min and t_max
		mask = (t._values > t_min) & (t._values < t_max)
		# Apply mask on the signal
		y = self._y
		y_cropped = y[mask]
		t_cropped = t[mask]
		# We need to find out the new t0 for the cropped signal
		new_t0 = t_cropped._values[0]
		
		if self.__class__ == TimeDomainSignal:
			sr = self._sr
			y_label = self._y_label
			t_label = self._t._label
			title = self._title
			return self.__class__(data=y_cropped, sr=sr, t0=new_t0, data_label=y_label, time_label=t_label, title=title)
		elif self.__class__ == AudioSignal:
			sr = self._sr
			title = self._title
			return self.__class__(data=y_cropped, sr=sr, t0=new_t0, title=title)
	
	@validate_args_type()
	def pad(self, right: np.ndarray | list | None = None, left: np.ndarray | list | None = None) -> Self:
		"""
		Pad the signal by prepending `left` and appending `right` sample arrays.
		If `left` or `right` is None, no padding is applied on that side.
		"""
		raise NotImplementedError
		
		data = self.data
		t0 = self.t0
		sr = self.sr
		
		if left is not None:
			left_arr = np.asarray(left)
			data = np.concatenate([left_arr, data])
			t0 = t0 - len(left_arr) / sr
			
		if right is not None:
			right_arr = np.asarray(right)
			data = np.concatenate([data, right_arr])
			
		return self.__class__(data=data, data_label=self.data_label, sr=sr, t0=t0, title=self.title)
	

	def apply_window(self, window: Self) -> Self:
		"""
		
		"""
		raise NotImplementedError
		
		assert window.sr == self.sr, "Sampling rates must match"
		assert np.any(np.isclose(self.t, window.t0)), "window.t0 must align with a sample in self.t"
		
		# Crop the original signal to the window size and position
		start_idx = int(np.where(np.isclose(self.t, window.t0))[0][0])
		end_idx = start_idx + len(window)
		signal = self[start_idx:end_idx]
		
		assert signal.shape == window.shape, "Window and signal shapes must match"
		assert np.allclose(signal.t, window.t), "Time axes must match for windowing"
		
		# Product of the signal with the window
		signal_windowed = signal * window
		
		return signal_windowed
	
	#===================================
	
	#----------------------------
	# To different signals
	#----------------------------
	def to_audio_signal(self) -> "AudioSignal":
		"""
		Moves TimeDomainSignal to AudioSignal
		"""
		raise NotImplementedError
		from modusa.signals.audio_signal import AudioSignal
		
		return AudioSignal(data=self.data, sr=self.sr, t0=self.t0, title=self.title)
	
	def to_spectrogram(
		self,
		n_fft: int = 2048,
		hop_length: int = 512,
		win_length: int | None = None,
		window: str = "hann"
	) -> "Spectrogram":
		"""
		Compute the Short-Time Fourier Transform (STFT) and return a Spectrogram object.
		
		Parameters
		----------
		n_fft : int
			FFT size.
		win_length : int or None
			Window length. Defaults to `n_fft` if None.
		hop_length : int
			Hop length between frames.
		window : str
			Type of window function to use (e.g., 'hann', 'hamming').
		
		Returns
		-------
		Spectrogram
			Spectrogram object containing S (complex STFT), t (time bins), and f (frequency bins).
		"""
		raise NotImplementedError
		import warnings
		warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.intervals")
		
		from modusa.signals.feature_time_domain_signal import FeatureTimeDomainSignal
		import librosa
		
		S = librosa.stft(self.data, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
		f = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
		t = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length)
		frame_rate = self.sr / hop_length
		spec = FeatureTimeDomainSignal(data=S, feature=f, feature_label="Freq (Hz)", frame_rate=frame_rate, t0=self.t0, time_label="Time (sec)", title=self.title)
		if self.title != self._name: # Means title of the audio was reset so we pass that info to spec
			spec = spec.set_meta_info(title=self.title)
		
		return spec
	#=====================================
	
	#=====================================
	
	#--------------------------
	# Other signal ops
	#--------------------------
	
#	def interpolate(self, to: TimeDomainSignal, kind: str = "linear", fill_value: str | float = "extrapolate") -> TimeDomainSignal:
#		"""
#		Interpolate the current signal to match the time axis of `to`.
#	
#		Parameters:
#			to (TimeDomainSignal): The signal whose time axis will be used.
#			kind (str): Interpolation method ('linear', 'nearest', etc.)
#			fill_value (str or float): Value used to fill out-of-bounds.
#	
#		Returns:
#			TimeDomainSignal: A new signal with values interpolated at `to.t`.
#		"""
#		assert self.y.ndim == 1, "Only 1D signals supported for interpolation"
#		
#		interpolator = interp1d(
#			self.t,
#			self.y,
#			kind=kind,
#			fill_value=fill_value,
#			bounds_error=False,
#			assume_sorted=True
#		)
#		
#		y_interp = interpolator(to.y)
	
#		return self.__class__(y=y_interp, sr=to.sr, t0=to.t0, title=f"{self.title} → interpolated")
	
	def autocorr(self) -> Self:
		"""
		
		"""
		raise NotImplementedError
		r = np.correlate(self.data, self.data, mode="full")
		r = r[self.data.shape[0] - 1:]
		r_signal = self.__class__(data=r, sr=self.sr, t0=self.t0, title=self.title + " [Autocorr]")
		return r_signal
		
	#======================================
	# Concatenation
	#======================================
		
	def __or__(self, other: int | float | np.generic | list | np.ndarray | Self) -> Self:
		"""
		Concatenate another signal.
		"""
		
		assert isinstance(other, (int, float, np.generic, list, np.ndarray, self.__class__))
		assert self._sr == other._sr
		
		if isinstance(other, (int, float, np.generic, list, np.ndarray)): other_data = np.asarray(other)
		elif isinstance(other, TimeDomainSignal): y_other = other._y
		
		# Getting data from the self object
		y = self._y
		y_label = self._y_label
		sr = self._sr
		t0 = self._t0
		t_label = self._t._label
		
		data_cat = np.concatenate([y, y_other])
		new_title = f"{self._title} | {other._title}"
		
		return self.__class__(data=data_cat, sr=sr, t0=t0, data_label=y_label, time_label=t_label, title=new_title)
	
	#======================================
	
	
	#-----------------------------------
	# Information
	#-----------------------------------
	
	def print_info(self) -> None:
		"""Prints info about the audio."""
		print("-" * 50)
		print(f"{'Title'}: {self._title}")
		print("-" * 50)
		print(f"{'Type':<20}: {self.__class__.__name__}")
		print(f"{'Shape':<20}: {self.shape}")
		print(f"{'Duration':<20}: {self._duration:.2f} sec")
		print(f"{'Sampling Rate':<20}: {self._sr} Hz")
		print(f"{'Sampling Period':<20}: {(1 / self._sr * 1000):.2f} ms")
		
		# Inheritance chain
		cls_chain = " → ".join(cls.__name__ for cls in reversed(self.__class__.__mro__[:-1]))
		print(f"{'Inheritance':<20}: {cls_chain}")
		print("=" * 50)
	
	#======================================