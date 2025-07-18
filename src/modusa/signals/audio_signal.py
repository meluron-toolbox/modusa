#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from modusa.signals.base import ModusaSignal
from modusa.tools.math_ops import MathOps
from typing import Self, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class AudioSignal(ModusaSignal):
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
	_description = ""
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-04"
	#----------------------------------
	
	@validate_args_type()
	def __init__(self, y: np.ndarray, sr: int, t0: float = 0.0, title: str | None = None):
		"""
		Loads the audio signal.
		"""
		if y.ndim != 1: # Mono signal only
			raise excp.InputValueError(f"`y` must have 1 dimension, not {y.ndim}.")
			
		self._y = y
		self._sr = sr
		self._t0 = t0
		self.title = title or self._name
			
	#----------------------
	# Properties
	#----------------------
	@immutable_property("Create a new object instead.")
	def y(self) -> np.ndarray:
		"""Returns audio data."""
		return self._y
	
	@immutable_property("Create a new object instead.")
	def sr(self) -> np.ndarray:
		"""Returns sampling rate of the audio."""
		return self._sr
	
	@immutable_property("Create a new object instead.")
	def t0(self) -> np.ndarray:
		"""Returns start timestamp of the audio."""
		return self._t0
	
	#----------------------
	# Derived Properties
	#----------------------
	@immutable_property("Create a new object instead.")
	def t(self) -> np.ndarray:
		"""Timestamp array of the audio."""
		return self.t0 + np.arange(len(self.y)) / self.sr 
	
	@immutable_property("Mutation not allowed.")
	def Ts(self) -> float:
		"""Sampling Period of the audio."""
		return 1. / self.sr

	@immutable_property("Mutation not allowed.")
	def duration(self) -> float:
		"""Duration of the audio."""
		return len(self.y) / self.sr
	
	@immutable_property("Mutation not allowed.")
	def shape(self) -> tuple:
		"""Shape of the audio signal."""
		return self.y.shape
	
	@immutable_property("Mutation not allowed.")
	def ndim(self) -> int:
		"""Dimension of the audio."""
		return self.y.ndim
	
	@immutable_property("Mutation not allowed.")
	def __len__(self) -> int:
		"""Dimension of the audio."""
		return len(self.y)
	
	#----------------------
	# Methods
	#----------------------
	
	def print_info(self) -> None:
		"""Prints info about the audio."""
		print("-" * 50)
		print(f"{'Title':<20}: {self.title}")
		print(f"{'Type':<20}: {self._name}")
		print(f"{'Duration':<20}: {self.duration:.2f} sec")
		print(f"{'Sampling Rate':<20}: {self.sr} Hz")
		print(f"{'Sampling Period':<20}: {(self.Ts*1000) :.4f} ms")
		print("-" * 50)
	
	def __getitem__(self, key):
		sliced_y = self.y[key]
		
		# If key is a single integer, return just the sample value
		if isinstance(key, int):
			return sliced_y
	
		# Otherwise, slicing: use self.t[key][0] as new t0
		new_t0 = self.t[key][0]
	
		return self.__class__(
			y=sliced_y,
			sr=self.sr,
			t0=new_t0,
			title=f"{self.title}[{key}]"
		)
			
	@validate_args_type()
	def crop(self, t_min: int | float | None = None, t_max: int | float | None = None) -> "AudioSignal":
		"""
		Crop the audio signal to a time range [t_min, t_max].

		.. code-block:: python

			from modusa.generators import AudioSignalGenerator
			audio_example = AudioSignalGenerator.generate_example()
			cropped_audio = audio_example.crop(1.5, 2)
	
		Parameters
		----------
		t_min : float or None
			Inclusive lower time bound. If None, no lower bound.
		t_max : float or None
			Exclusive upper time bound. If None, no upper bound.
	
		Returns
		-------
		AudioSignal
			Cropped audio signal.
		"""
		y = self.y
		t = self.t

		mask = np.ones_like(t, dtype=bool)
		if t_min is not None:
			mask &= (t >= t_min)
		if t_max is not None:
			mask &= (t < t_max)
			
		cropped_y = y[mask]
		new_t0 = t[mask][0] if np.any(mask) else self.t0  # fallback to original t0 if mask is empty

		return self.__class__(y=cropped_y, sr=self.sr, t0=new_t0, title=self.title)
	
	
	@validate_args_type()
	def plot(
		self,
		ax: plt.Axes | None = None,
		fmt: str = "k-",
		title: str | None = None,
		ylabel: str | None = "Amplitude",
		xlabel: str | None = "Time (sec)",
		ylim: tuple[float, float] | None = None,
		xlim: tuple[float, float] | None = None,
		highlight_regions: list[tuple[float, float, str]] | None = None,
		vlines: list[float] | None = None,
		hlines: list[float] | None = None,
		legend: str | tuple[str, str] | None = None,
		show_grid: bool = False,
		show_stem: bool = False,
	) -> plt.Figure | None:
		"""
		Plot the audio waveform using matplotlib.
		
		.. code-block:: python
		
			from modusa.generators import AudioSignalGenerator
			audio_example = AudioSignalGenerator.generate_example()
			audio_example.plot(color="orange", title="Example Audio")
		
		Parameters
		----------
		ax : matplotlib.axes.Axes | None
			Pre-existing axes to plot into. If None, a new figure and axes are created.
		fmt : str | None
			Format of the plot as per matplotlib standards (Eg. "k-" or "blue--o)
		title : str | None
			Plot title. Defaults to the signal’s title.
		ylabel : str | None
			Label for the y-axis. Defaults to `"Amplitude"`.
		xlabel : str | None
			Label for the x-axis. Defaults to `"Time (sec)"`.
		ylim : tuple[float, float] | None
			Limits for the y-axis.
		xlim : tuple[float, float] | None
		highlight_regions : list[tuple[float, float, str]] | None
			List of time intervals to highlight on the plot, each as (start, end, 'tag').
		vlines: list[float]
			List of x values to draw vertical lines. (Eg. [10, 13.5])
		hlines: list[float]
			List of y values to draw horizontal lines. (Eg. [10, 13.5])
		show_grid: bool
			If true, shows grid.
		show_stem : bool
			If True, use a stem plot instead of a continuous line. Autorejects if signal is too large.
		legend : str | tuple[str, str] | None
			If provided, adds a legend at the specified location (e.g., "signal", ["signal", "upper right"]).
			Limits for the x-axis.
		
		Returns
		-------
		matplotlib.figure.Figure | None
			The figure object containing the plot or None in case an axis is provided.
		"""
		
		from modusa.tools.plotter import Plotter
		
		if title is None:
			title = self.title
		
		fig: plt.Figure | None = Plotter.plot_signal(y=self.y, x=self.t, ax=ax, fmt=fmt, title=title, ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim, highlight_regions=highlight_regions, vlines=vlines, hlines=hlines, show_grid=show_grid, show_stem=show_stem, legend=legend)
		
		return fig
	
	def play(self, regions: list[tuple[float, float], ...] | None = None, title: str | None = None):
		"""
		Play the audio signal inside a Jupyter Notebook.
	
		.. code-block:: python
	
			from modusa.generators import AudioSignalGenerator
			audio = AudioSignalGenerator.generate_example()
			audio.play(regions=[(0.0, 1.0), (2.0, 3.0)])
	
		Parameters
		----------
		regions : list of tuple of float, optional
			List of (start_time, end_time) pairs in seconds specifying the regions to play.
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
		audio_player = AudioPlayer.play(y=self.y, sr=self.sr, regions=regions, title=title)
		
		return audio_player
	
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
		import warnings
		warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.intervals")
		
		from modusa.signals.spectrogram import Spectrogram
		import librosa
		
		S = librosa.stft(self.y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
		f = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
		t = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length)
		frame_rate = self.sr / hop_length
		spec = Spectrogram(S=S, f=f, frame_rate=frame_rate, t0=self.t0)
		if self.title != self._name: # Means title of the audio was reset so we pass that info to spec
			spec.title = self.title
			
		return spec
	
	
	#----------------------------
	# Math ops
	#----------------------------
	
	def __array__(self, dtype=None):
		return np.asarray(self.y, dtype=dtype)
	
	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		if method == "__call__":
			input_arrays = [x.y if isinstance(x, self.__class__) else x for x in inputs]
			result = ufunc(*input_arrays, **kwargs)
			return self.__class__(y=result, sr=self.sr, title=f"{self.title}")
		return NotImplemented
	
	def __add__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.add(self.y, other_data)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __radd__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.add(other_data, self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __sub__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.subtract(self.y, other_data)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __rsub__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.subtract(other_data, self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __mul__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.multiply(self.y, other_data)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __rmul__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.multiply(other_data, self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __truediv__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.divide(self.y, other_data)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __rtruediv__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.divide(other_data, self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __floordiv__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.floor_divide(self.y, other_data)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __rfloordiv__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.floor_divide(other_data, self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __pow__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.power(self.y, other_data)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __rpow__(self, other):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.power(other_data, self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __abs__(self):
		other_data = other.y if isinstance(other, self.__class__) else other
		result = MathOps.abs(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def __or__(self, other):
		if not isinstance(other, self.__class__):
			raise excp.InputTypeError(f"Can only concatenate with another {self.__class__.__name__}")
			
		if self.sr != other.sr:
			raise excp.InputValueError(f"Cannot concatenate: Sampling rates differ ({self.sr} vs {other.sr})")
			
		# Concatenate raw audio data
		y_cat = np.concatenate([self.y, other.y])
	
		# Preserve t0 of the first signal
		new_title = f"{self.title} | {other.title}"
		return self.__class__(y=y_cat, sr=self.sr, t0=self.t0, title=new_title)
	
	
	#--------------------------
	# Other signal ops
	#--------------------------
	def abs(self) -> Self:
		"""Compute the element-wise abs of the signal data."""
		result = MathOps.abs(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)
	
	def sin(self) -> Self:
		"""Compute the element-wise sine of the signal data."""
		result = MathOps.sin(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def cos(self) -> Self:
		"""Compute the element-wise cosine of the signal data."""
		result = MathOps.cos(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def exp(self) -> Self:
		"""Compute the element-wise exponential of the signal data."""
		result = MathOps.exp(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def tanh(self) -> Self:
		"""Compute the element-wise hyperbolic tangent of the signal data."""
		result = MathOps.tanh(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def log(self) -> Self:
		"""Compute the element-wise natural logarithm of the signal data."""
		result = MathOps.log(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def log1p(self) -> Self:
		"""Compute the element-wise natural logarithm of (1 + signal data)."""
		result = MathOps.log1p(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def log10(self) -> Self:
		"""Compute the element-wise base-10 logarithm of the signal data."""
		result = MathOps.log10(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)

	def log2(self) -> Self:
		"""Compute the element-wise base-2 logarithm of the signal data."""
		result = MathOps.log2(self.y)
		return self.__class__(y=result, sr=self.sr, t0=self.t0, title=self.title)
	

	#--------------------------
	# Aggregation signal ops
	#--------------------------
	def mean(self) -> "np.generic":
		"""Compute the mean of the signal data."""
		return MathOps.mean(self.y)
	
	def std(self) -> "np.generic":
		"""Compute the standard deviation of the signal data."""
		return MathOps.std(self.y)
	
	def min(self) -> "np.generic":
		"""Compute the minimum value in the signal data."""
		return MathOps.min(self.y)
	
	def max(self) -> "np.generic":
		"""Compute the maximum value in the signal data."""
		return MathOps.max(self.y)
	
	def sum(self) -> "np.generic":
		"""Compute the sum of the signal data."""
		return MathOps.sum(self.y)
	
	#-----------------------------------
	# Repr
	#-----------------------------------
	
	def __str__(self):
		cls = self.__class__.__name__
		data = self.y
		
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=50,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
			formatter={'float_kind': lambda x: f"{x:.4g}"}
		)
		
		return f"Signal({arr_str}, shape={data.shape}, type={cls})"
	
	def __repr__(self):
		cls = self.__class__.__name__
		data = self.y
		
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=50,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
			formatter={'float_kind': lambda x: f"{x:.4g}"}
		)
		
		return f"Signal({arr_str}, shape={data.shape}, type={cls})"