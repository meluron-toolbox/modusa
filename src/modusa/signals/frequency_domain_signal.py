#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from modusa.signals.base import ModusaSignal
from typing import Self, Any
import numpy as np
import matplotlib.pyplot as plt

class FrequencyDomainSignal(ModusaSignal):
	"""
	Represents a 1D signal in the frequency domain.
	
	Note
	----
	- The class is not intended to be instantiated directly.

	Parameters
	----------
	spectrum : np.ndarray
		The frequency-domain representation of the signal (real or complex-valued).
	f : np.ndarray
		The frequency axis corresponding to the spectrum values. Must match the shape of `spectrum`.
	t0 : float, optional
		The time (in seconds) corresponding to the origin of this spectral slice. Defaults to 0.0.
	title : str, optional
		An optional title for display or plotting purposes.
	"""
	#--------Meta Information----------
	_name = "Frequency Domain Signal"
	_description = "Represents Frequency Domain Signal"
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-09"
	#----------------------------------
	
	@validate_args_type()
	def __init__(self, spectrum: np.ndarray, f: np.ndarray, t0: float | int = 0.0, title: str | None = None):
		super().__init__() # Instantiating `ModusaSignal` class
		
		if spectrum.shape != f.shape:
			raise excp.InputValueError(f"`spectrum` and `f` shape must match, got {spectrum.shape} and {f.shape}")
		
		self._spectrum = spectrum
		self._f = f
		self._t0 = float(t0)
	
		self.title = title or self._name # This title will be used as plot title by default
		
	
	#----------------------
	# Properties
	#----------------------
	
	@immutable_property("Create a new object instead.")
	def spectrum(self) -> np.ndarray:
		"""Complex valued spectrum data."""
		return self._spectrum
	
	@immutable_property("Create a new object instead.")
	def f(self) -> np.ndarray:
		"""frequency array of the spectrum."""
		return self._f
	
	@immutable_property("Create a new object instead.")
	def t0(self) -> np.ndarray:
		"""Time origin (in seconds) of this spectral slice, e.g., from a spectrogram frame."""
		return self._t0
	
	#----------------------
	# Derived Properties
	#----------------------
	@immutable_property("Create a new object instead.")
	def __len__(self) -> int:
		return len(self.spectrum)
	
	@immutable_property("Create a new object instead.")
	def ndim(self) -> int:
		return self.spectrum.ndim
	
	@immutable_property("Create a new object instead.")
	def shape(self) -> tuple:
		return self.spectrum.shape
	
	#----------------------
	# Methods
	#----------------------
	
	def print_info(self) -> None:
		"""Prints info about the audio."""
		print("-" * 50)
		print(f"{'Title':<20}: {self.title}")
		print(f"{'Type':<20}: {self._name}")
		print(f"{'Frequency Range':<20}: ({self.f[0]:.2f}, {self.f[-1]:.2f}) Hz")
		print("-" * 50)
	
	def __getitem__(self, key: slice) -> Self:
		sliced_spectrum = self._spectrum[key]
		sliced_f = self._f[key]
		return self.__class__(spectrum=sliced_spectrum, f=sliced_f, t0=self.t0, title=self.title)
	
	@validate_args_type()
	def plot(
		self,
		ax: plt.Axes | None = None,
		fmt: str = "k-",
		title: str | None = None,
		ylabel: str | None = "Strength",
		xlabel: str | None = "Frequency (Hz)",
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
			
		fig: plt.Figure | None = Plotter.plot_signal(y=self.spectrum, x=self.f, ax=ax, fmt=fmt, title=title, ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim, highlight_regions=highlight_regions, vlines=vlines, hlines=hlines, show_grid=show_grid, show_stem=show_stem, legend=legend)
		
		return fig
	
	#----------------------------
	# Math ops
	#----------------------------
	
	def __array__(self, dtype=None):
		return np.asarray(self.spectrum, dtype=dtype)
	
	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		if method == "__call__":
			input_arrays = [x.spectrum if isinstance(x, self.__class__) else x for x in inputs]
			result = ufunc(*input_arrays, **kwargs)
			return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
		return NotImplemented
	
	def __add__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.add(self.spectrum, other_data)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __radd__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.add(other_data, self.spectrum)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __sub__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.subtract(self.spectrum, other_data)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __rsub__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.subtract(other_data, self.spectrum)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __mul__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.multiply(self.spectrum, other_data)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __rmul__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.multiply(other_data, self.spectrum)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __truediv__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.divide(self.spectrum, other_data)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __rtruediv__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.divide(other_data, self.spectrum)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __floordiv__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.floor_divide(self.spectrum, other_data)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __rfloordiv__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.floor_divide(other_data, self.spectrum)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __pow__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.power(self.spectrum, other_data)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __rpow__(self, other):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.power(other_data, self.spectrum)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def __abs__(self):
		other_data = other.spectrum if isinstance(other, self.__class__) else other
		result = MathOps.abs(self.spectrum)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)	
	
	#--------------------------
	# Other signal ops
	#--------------------------
	def abs(self) -> Self:
		"""Compute the element-wise abs of the signal data."""
		result = MathOps.abs(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def sin(self) -> Self:
		"""Compute the element-wise sine of the signal data."""
		result = MathOps.sin(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def cos(self) -> Self:
		"""Compute the element-wise cosine of the signal data."""
		result = MathOps.cos(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def exp(self) -> Self:
		"""Compute the element-wise exponential of the signal data."""
		result = MathOps.exp(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def tanh(self) -> Self:
		"""Compute the element-wise hyperbolic tangent of the signal data."""
		result = MathOps.tanh(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def log(self) -> Self:
		"""Compute the element-wise natural logarithm of the signal data."""
		result = MathOps.log(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def log1p(self) -> Self:
		"""Compute the element-wise natural logarithm of (1 + signal data)."""
		result = MathOps.log1p(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def log10(self) -> Self:
		"""Compute the element-wise base-10 logarithm of the signal data."""
		result = MathOps.log10(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	def log2(self) -> Self:
		"""Compute the element-wise base-2 logarithm of the signal data."""
		result = MathOps.log2(self.y)
		return self.__class__(spectrum=result, f=self.f, t0=self.t0, title=self.title)
	
	
	#--------------------------
	# Aggregation signal ops
	#--------------------------
	def mean(self) -> "np.generic":
		"""Compute the mean of the signal data."""
		return MathOps.mean(self.spectrum)
	
	def std(self) -> "np.generic":
		"""Compute the standard deviation of the signal data."""
		return MathOps.std(self.spectrum)
	
	def min(self) -> "np.generic":
		"""Compute the minimum value in the signal data."""
		return MathOps.min(self.spectrum)
	
	def max(self) -> "np.generic":
		"""Compute the maximum value in the signal data."""
		return MathOps.max(self.spectrum)
	
	def sum(self) -> "np.generic":
		"""Compute the sum of the signal data."""
		return MathOps.sum(self.spectrum)
	
	#-----------------------------------
	# Repr
	#-----------------------------------
	
	def __str__(self):
		cls = self.__class__.__name__
		data = self.spectrum
		
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
		data = self.spectrum
		
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=50,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
			formatter={'float_kind': lambda x: f"{x:.4g}"}
		)
		
		return f"Signal({arr_str}, shape={data.shape}, type={cls})"
	