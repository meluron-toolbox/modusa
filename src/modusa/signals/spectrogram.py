#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from modusa.signals.base import ModusaSignal
from modusa.tools.math_ops import MathOps
from typing import Self, Any
import numpy as np
import matplotlib.pyplot as plt

class Spectrogram(ModusaSignal):
	"""
	A 2D time–frequency representation of a signal.

	Parameters
	----------
	S : np.ndarray
		2D matrix representing the spectrogram (shape: [n_freqs, n_frames]).
	f : np.ndarray
		Frequency axis corresponding to the rows of `S` (shape: [n_freqs]).
	t : np.ndarray
		Time axis corresponding to the columns of `S` (shape: [n_frames]).
	title : str, optional
		Optional title for the spectrogram (e.g., used in plotting).
	"""
	
	#--------Meta Information----------
	_name = "Spectrogram"
	_description = ""
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-07"
	#----------------------------------
	
	@validate_args_type()
	def __init__(self, S: np.ndarray, f: np.ndarray, frame_rate: float, t0: float = 0.0, title: str | None = None):
		super().__init__()
		
		if S.ndim != 2:
			raise excp.InputValueError(f"`S` must have 2 dimensions, got {S.ndim}.")
		if f.ndim != 1:
			raise excp.InputValueError(f"`f` must have 1 dimension, got {f.ndim}.")
		if f.shape[0] != S.shape[0]:
			raise excp.InputValueError(
				f"Shape mismatch between `S` and `f`: expected {S.shape[0]}, got {f.shape[0]}"
			)
		if S.shape[1] == 0:
			raise excp.InputValueError("`S` must have at least one time frame.")
			
		self._S = S
		self._f = f
		self._frame_rate = float(frame_rate)
		self._t0 = float(t0)
		self.title = title or self._name

	#----------------------
	# Properties
	#----------------------
	
	@immutable_property("Create a new object instead.")
	def S(self) -> np.ndarray:
		"""Spectrogram matrix (freq × time)."""
		return self._S
	
	@immutable_property("Create a new object instead.")
	def f(self) -> np.ndarray:
		"""Frequency axis."""
		return self._f
	
	@immutable_property("Create a new object instead.")
	def frame_rate(self) -> np.ndarray:
		"""Frequency axis."""
		return self._frame_rate
	
	@immutable_property("Create a new object instead.")
	def t0(self) -> np.ndarray:
		"""Frequency axis."""
		return self._t0
	
	#----------------------
	# Derived Properties
	#----------------------
	
	@immutable_property("Create a new object instead.")
	def t(self) -> np.ndarray:
		"""Time axis."""
		n_frames = self._S.shape[1]
		return np.arange(n_frames) / self.frame_rate + self.t0
	
	@immutable_property("Read only property.")
	def shape(self) -> np.ndarray:
		"""Shape of the spectrogram (freqs, frames)."""
		return self.S.shape
	
	@immutable_property("Read only property.")
	def ndim(self) -> np.ndarray:
		"""Number of dimensions (always 2)."""
		return self.S.ndim
	
	@property
	def magnitude(self) -> "Spectrogram":
		"""Return a new Spectrogram with magnitude values."""
		mag = np.abs(self.S)
		return self.__class__(S=mag, f=self.f, frame_rate=self.frame_rate, title=self.title)

	@property
	def power(self) -> "Spectrogram":
		"""Return a new Spectrogram with power (magnitude squared)."""
		power = np.abs(self.S) ** 2
		return self.__class__(S=power, f=self.f, frame_rate=self.frame_rate, title=self.title)

	@property
	def angle(self) -> "Spectrogram":
		"""Return a new Spectrogram with phase angle (in radians)."""
		angle = np.angle(self.S)
		return self.__class__(S=angle, f=self.f, frame_rate=self.frame_rate, title=self.title)

	@property
	def real(self) -> "Spectrogram":
		"""Return a new Spectrogram with real part."""
		return self.__class__(S=self.S.real, f=self.f, frame_rate=self.frame_rate, title=self.title)

	@property
	def imag(self) -> "Spectrogram":
		"""Return a new Spectrogram with imaginary part."""
		return self.__class__(S=self.S.imag, f=self.f, frame_rate=self.frame_rate, title=self.title)

	@property
	def phase(self) -> "Spectrogram":
		"""Return a new Spectrogram with normalized phase."""
		phase = self.S / (np.abs(self.S) + 1e-10)  # Avoid division by zero
		return self.__class__(S=phase, f=self.f, frame_rate=self.frame_rate, title=self.title)
	
	
		
	#------------------------
	# Useful tools
	#------------------------
	
	def print_info(self) -> None:
		"""Print key information about the spectrogram signal."""
		time_resolution = 1.0 / self.frame_rate
		n_freq_bins = self.S.shape[0]
		
		# Estimate NFFT size
		nfft = (n_freq_bins - 1) * 2
		
		print("-"*50)
		print(f"{'Title':<20}: {self.title}")
		print(f"{'Kind':<20}: {self._name}")
		print(f"{'Shape':<20}: {self.S.shape} (freq bins × time frames)")
		print(f"{'Frame Rate':<20}: {self.frame_rate} (frames / sec)")
		print(f"{'Time resolution':<20}: {time_resolution:.4f} sec ({time_resolution * 1000:.2f} ms)")
		print(f"{'Freq resolution':<20}: {(self.f[1] - self.f[0]):.2f} Hz")
		print("-"*50)

	def __getitem__(self, key: tuple[int | slice, int | slice]) -> "Spectrogram | FrequencyDomainSignal | TimeDomainSignal":
		"""
		Enable 2D indexing: signal[f_idx, t_idx]
	
		Returns:
		- Spectrogram when both f and t are slices
		- FrequencyDomainSignal when t is int (i.e., fixed time)
		- TimeDomainSignal when f is int (i.e., fixed frequency)
		"""
		from modusa.signals.time_domain_signal import TimeDomainSignal
		from modusa.signals.frequency_domain_signal import FrequencyDomainSignal
		
		if isinstance(key, tuple) and len(key) == 2:
			f_key, t_key = key
		
			sliced_data = self.S[f_key, t_key]
			sliced_f = self.f[f_key]
			sliced_t = self.t[t_key]
		
			# Case 1: Scalar value → return plain numpy scalar
			if np.isscalar(sliced_data):
				return np.array(sliced_data)
		
			# Case 2: frequency slice at a single time (→ FrequencyDomainSignal)
			elif isinstance(t_key, int):
				if not isinstance(f_key, int):  # already handled scalar case
					sliced_data = np.asarray(sliced_data).flatten()
					sliced_f = np.asarray(sliced_f)
					t0 = float(self.t[t_key])
					return FrequencyDomainSignal(spectrum=sliced_data, f=sliced_f, t0=t0, title=self.title + f" [t = {t0:.2f} sec]")
		
			# Case 3: time slice at a single frequency (→ TimeDomainSignal)
			elif isinstance(f_key, int):
				sliced_data = np.asarray(sliced_data).flatten()
				sliced_t = np.asarray(sliced_t)
				sr = 1.0 / np.mean(np.diff(self.t))  # assume uniform time axis
				t0 = float(self.t[0])
				f_val = float(self.f[f_key])
				return TimeDomainSignal(y=sliced_data, sr=sr, t0=t0, title=self.title + f" [f = {f_val:.2f} Hz]")
		
			# Case 4: 2D slice → Spectrogram
			else:
				return self.__class__(S=sliced_data, f=sliced_f, frame_rate=self.frame_rate, t0=sliced_t[0], title=self.title)
		
		raise TypeError("Expected 2D indexing: signal[f_idx, t_idx]")
	
	def crop(
		self,
		f_min: float | None = None,
		f_max: float | None = None,
		t_min: float | None = None,
		t_max: float | None = None
	) -> "Spectrogram":
		"""
		Crop the spectrogram to a rectangular region in frequency-time space.
	
		.. code-block:: python

			cropped = spec.crop(f_min=100, f_max=1000, t_min=5.0, t_max=10.0)
	
		Parameters
		----------
		f_min : float or None
			Inclusive lower frequency bound. If None, no lower bound.
		f_max : float or None
			Exclusive upper frequency bound. If None, no upper bound.
		t_min : float or None
			Inclusive lower time bound. If None, no lower bound.
		t_max : float or None
			Exclusive upper time bound. If None, no upper bound.
	
		Returns
		-------
		Spectrogram
			Cropped spectrogram.
		"""
		S = self.S
		f = self.f
		t = self.t
		
		f_mask = (f >= f_min) if f_min is not None else np.ones_like(f, dtype=bool)
		f_mask &= (f < f_max) if f_max is not None else f_mask
		
		t_mask = (t >= t_min) if t_min is not None else np.ones_like(t, dtype=bool)
		t_mask &= (t < t_max) if t_max is not None else t_mask
		
		cropped_S = S[np.ix_(f_mask, t_mask)]
		cropped_f = f[f_mask]
		cropped_t = t[t_mask]
		
		return self.__class__(S=cropped_S, f=cropped_f, frame_rate=self.frame_rate, t0=cropped_t[0], title=self.title)
	
	
	def plot(
		self,
		ax: plt.Axes | None = None,
		cmap: str = "gray_r",
		title: str | None = None,
		Mlabel: str | None = None,
		ylabel: str | None = "Frequency (hz)",
		xlabel: str | None = "Time (sec)",
		ylim: tuple[float, float] | None = None,
		xlim: tuple[float, float] | None = None,
		highlight: list[tuple[float, float, float, float]] | None = None,
		vlines: list | None = None,
		hlines: list | None = None,
		origin: str = "lower",  # or "lower"
		gamma: int | float | None = None,
		show_colorbar: bool = True,
		cax: plt.Axes | None = None,
		show_grid: bool = True,
		tick_mode: str = "center",  # "center" or "edge"
		n_ticks: tuple[int, int] | None = None,
	) -> plt.Figure:
		"""
		Plot the spectrogram using Matplotlib.
	
		.. code-block:: python
			
			fig = spec.plot(log_compression_factor=10, title="Log-scaled Spectrogram")
	
		Parameters
		----------
		log_compression_factor : float or int, optional
			If specified, apply log-compression using log(1 + S * factor).
		ax : matplotlib.axes.Axes, optional
			Axes to draw on. If None, a new figure and axes are created.
		cmap : str, default "gray_r"
			Colormap used for the image.
		title : str, optional
			Title to use for the plot. Defaults to the signal's title.
		Mlabel : str, optional
			Label for the colorbar (e.g., "Magnitude", "dB").
		ylabel : str, optional
			Label for the y-axis. Default is "Frequency (hz)".
		xlabel : str, optional
			Label for the x-axis. Default is "Time (sec)".
		ylim : tuple of float, optional
			Limits for the y-axis (frequency).
		xlim : tuple of float, optional
			Limits for the x-axis (time).
		highlight : list of (x, y, w, h), optional
			Rectangular regions to highlight, specified in data coordinates.
		origin : {"lower", "upper"}, default "lower"
			Origin position for the image (for flipping vertical axis).
		show_colorbar : bool, default True
			Whether to display the colorbar.
		cax : matplotlib.axes.Axes, optional
			Axis to draw the colorbar on. If None, uses default placement.
		show_grid : bool, default True
			Whether to show the major gridlines.
		tick_mode : {"center", "edge"}, default "center"
			Whether to place ticks at bin centers or edges.
		n_ticks : tuple of int, optional
			Number of ticks (y_ticks, x_ticks) to display on each axis.
	
		Returns
		-------
		matplotlib.figure.Figure
			The figure object containing the plot.
		"""
		from modusa.tools.plotter import Plotter
		
		title = title or self.title
	
		fig = Plotter.plot_matrix(
			M=self.S,
			r=self.f,
			c=self.t,
			ax=ax,
			cmap=cmap,
			title=title,
			Mlabel=Mlabel,
			rlabel=ylabel,
			clabel=xlabel,
			rlim=ylim,
			clim=xlim,
			highlight=highlight,
			vlines=vlines,
			hlines=hlines,
			origin=origin,
			gamma=gamma,
			show_colorbar=show_colorbar,
			cax=cax,
			show_grid=show_grid,
			tick_mode=tick_mode,
			n_ticks=n_ticks	
		)
		
		return fig
	
	#----------------------------
	# Math ops
	#----------------------------
	
	def __array__(self, dtype=None):
		return np.asarray(self._S, dtype=dtype)
	
	def __add__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = MathOps.add(self.S, other_data)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __radd__(self, other):
		result = MathOps.add(other, self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __sub__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = MathOps.subtract(self.S, other_data)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __rsub__(self, other):
		result = MathOps.subtract(other, self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __mul__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = MathOps.multiply(self.S, other_data)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __rmul__(self, other):
		result = MathOps.multiply(other, self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __truediv__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = MathOps.true_divide(self.S, other_data)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __rtruediv__(self, other):
		result = MathOps.true_divide(other, self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __floordiv__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = MathOps.floor_divide(self.S, other_data)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __rfloordiv__(self, other):
		result = MathOps.floor_divide(other, self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __pow__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = MathOps.power(self.S, other_data)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __rpow__(self, other):
		result = MathOps.power(other, self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def __abs__(self):
		result = MathOps.abs(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	#--------------------------
	# Other signal ops
	#--------------------------
	
	def sin(self):
		"""Element-wise sine of the spectrogram."""
		result = MathOps.sin(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def cos(self):
		"""Element-wise cosine of the spectrogram."""
		result = MathOps.cos(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def exp(self):
		"""Element-wise exponential of the spectrogram."""
		result = MathOps.exp(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def tanh(self):
		"""Element-wise hyperbolic tangent of the spectrogram."""
		result = MathOps.tanh(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def log(self):
		"""Element-wise natural logarithm of the spectrogram."""
		result = MathOps.log(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def log1p(self):
		"""Element-wise log(1 + M) of the spectrogram."""
		result = MathOps.log1p(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def log10(self):
		"""Element-wise base-10 logarithm of the spectrogram."""
		result = MathOps.log10(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	def log2(self):
		"""Element-wise base-2 logarithm of the spectrogram."""
		result = MathOps.log2(self.S)
		return self.__class__(S=result, f=self.f, frame_rate=self.frame_rate, t0=self.t0, title=self.title)
	
	#--------------------------
	# Aggregation signal ops
	#--------------------------
	
	def mean(self, axis: int | None = None) -> float:
		"""Return the mean of the spectrogram values."""
		from modusa.signals.time_domain_signal import TimeDomainSignal
		from modusa.signals.frequency_domain_signal import FrequencyDomainSignal
		result = MathOps.mean(self.S, axis=axis)
		if axis == 0: # Aggregating across rows
			return TimeDomainSignal(y=result, sr=self.frame_rate, t0=self.t0, title=self.title)
		elif axis in [1, -1]:
			return FrequencyDomainSignal(spectrum=result, f=self.f, t0=self.t0, title=self.title)
		elif axis is None:
			return result
		else:
			raise excp.InputValueError("Can't perform mean operation")
	
	def std(self, axis: int | None = None) -> float:
		"""Return the standard deviation of the spectrogram values."""
		from modusa.signals.time_domain_signal import TimeDomainSignal
		from modusa.signals.frequency_domain_signal import FrequencyDomainSignal
		result = MathOps.std(self.S, axis=axis)
		if axis == 0: # Aggregating across rows
			return TimeDomainSignal(y=result, sr=self.frame_rate, t0=self.t0, title=self.title)
		elif axis in [1, -1]:
			return FrequencyDomainSignal(spectrum=result, f=self.f, t0=self.t0, title=self.title)
		elif axis is None:
			return result
		else:
			raise excp.InputValueError("Can't perform std operation")
	
	def min(self, axis: int | None = None) -> float:
		"""Return the minimum value in the spectrogram."""
		from modusa.signals.time_domain_signal import TimeDomainSignal
		from modusa.signals.frequency_domain_signal import FrequencyDomainSignal
		result = MathOps.min(self.S, axis=axis)
		if axis == 0: # Aggregating across rows
			return TimeDomainSignal(y=result, sr=self.frame_rate, t0=self.t0, title=self.title)
		elif axis in [1, -1]:
			return FrequencyDomainSignal(spectrum=result, f=self.f, t0=self.t0, title=self.title)
		elif axis is None:
			return result
		else:
			raise excp.InputValueError("Can't perform min operation")
	
	def max(self, axis: int | None = None) -> float:
		"""Return the maximum value in the spectrogram."""
		from modusa.signals.time_domain_signal import TimeDomainSignal
		from modusa.signals.frequency_domain_signal import FrequencyDomainSignal
		result = MathOps.max(self.S, axis=axis)
		if axis == 0: # Aggregating across rows
			return TimeDomainSignal(y=result, sr=self.frame_rate, t0=self.t0, title=self.title)
		elif axis in [1, -1]:
			return FrequencyDomainSignal(spectrum=result, f=self.f, t0=self.t0, title=self.title)
		elif axis is None:
			return result
		else:
			raise excp.InputValueError("Can't perform max operation")
	
	def sum(self, axis: int | None = None) -> float:
		"""Return the sum of the spectrogram values."""
		from modusa.signals.time_domain_signal import TimeDomainSignal
		from modusa.signals.frequency_domain_signal import FrequencyDomainSignal
		result = MathOps.sum(self.S, axis=axis)
		if axis == 0: # Aggregating across rows
			return TimeDomainSignal(y=result, sr=self.frame_rate, t0=self.t0, title=self.title)
		elif axis in [1, -1]:
			return FrequencyDomainSignal(spectrum=result, f=self.f, t0=self.t0, title=self.title)
		elif axis is None:
			return result
		else:
			raise excp.InputValueError("Can't perform sum operation")
	
	#-----------------------------------
	# Repr
	#-----------------------------------
	
	def __str__(self):
		cls = self.__class__.__name__
		data = self.S
		
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=50,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
			formatter={'float_kind': lambda x: f"{x:.4g}"}
		)
		
		return f"Signal({arr_str}, shape={data.shape}, kind={cls})"
	
	def __repr__(self):
		cls = self.__class__.__name__
		data = self.S
		
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=50,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
			formatter={'float_kind': lambda x: f"{x:.4g}"}
		)
		
		return f"Signal({arr_str}, shape={data.shape}, kind={cls})"