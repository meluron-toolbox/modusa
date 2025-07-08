#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from modusa.signals.base import ModusaSignal
from typing import Self, Any
import numpy as np
import matplotlib.pyplot as plt

class Spectrogram(ModusaSignal):
	"""

	"""
	
	#--------Meta Information----------
	_name = "Spectrogram"
	_description = ""
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-07"
	#----------------------------------
	
	@validate_args_type()
	def __init__(self, S: np.ndarray, f: np.ndarray, t: np.ndarray, title: str | None = None):
		super().__init__() # Instantiating `ModusaSignal` class
		
		if S.ndim != 2:
			raise excp.InputValueError(f"`S` must have 2 dimension, got {S.ndim}.")
		if f.ndim != 1:
			raise excp.InputValueError(f"`f` must have 1 dimension, got {f.ndim}.")
		if t.ndim != 1:
			raise excp.InputValueError(f"`t` must have 1 dimension, got {t.ndim}.")
		
		if t.shape[0] != S.shape[1] or f.shape[0] != S.shape[0]:
			raise excp.InputValueError(f"`f` and `t` shape do not match with `M` {S.shape}, got {(f.shape[0], t.shape[0])}")
		
		if S.shape[1] < 2:
			raise excp.InputValueError(f"`S` must not have time dimension shape < 2, got {S.shape[1]}")
		dts = np.diff(t)
		if not np.allclose(dts, dts[0]):
			raise excp.InputValueError("`t` must be equally spaced")
		
		self._S = S
		self._f = f
		self._t = t
		self.title = title or self._name
		
	#----------------------
	# Properties
	#----------------------
	@immutable_property("Create a new object instead.")
	def S(self) -> np.ndarray:
		return self._S
	
	@immutable_property("Create a new object instead.")
	def f(self) -> np.ndarray:
		return self._f
	
	@immutable_property("Create a new object instead.")
	def t(self) -> np.ndarray:
		return self._t
	
	@immutable_property("Read only property.")
	def shape(self) -> np.ndarray:
		return self.S.shape
	
	@immutable_property("Read only property.")
	def ndim(self) -> np.ndarray:
		return self.S.ndim
	
	@immutable_property("Mutation not allowed.")
	def info(self) -> None:
		"""Print key information about the spectrogram signal."""
		time_resolution = self.t[1] - self.t[0]
		n_freq_bins = self.S.shape[0]
	
		# Estimate NFFT size
		nfft = (n_freq_bins - 1) * 2
		
		print("-"*50)
		print(f"{'Title':<20}: {self.title}")
		print(f"{'Kind':<20}: {self._name}")
		print(f"{'Shape':<20}: {self.S.shape} (freq bins × time frames)")
		print(f"{'Time resolution':<20}: {time_resolution:.4f} sec ({time_resolution * 1000:.2f} ms)")
		print(f"{'Freq resolution':<20}: {(self.f[1] - self.f[0]):.2f} Hz")
		print("-"*50)
	#------------------------
	
		
	#------------------------
	# Useful tools
	#------------------------
	def __getitem__(self, key: tuple[int, int]) -> "Spectrogram":
		"""
		Enable 2D indexing: signal[f_idx, t_idx]
	
		Returns a new Spectrogram object with sliced data and corresponding frequency/time axes.
		"""
		if isinstance(key, tuple) and len(key) == 2:
			f_key, t_key = key
			
			# Slice data
			sliced_data = self.S[f_key, t_key]
			
			# Slice frequency and time axes
			sliced_f = self.f[f_key]
			sliced_t = self.t[t_key]
			
			# Normalize shapes
			if np.isscalar(sliced_data):
				sliced_data = np.array([[sliced_data]])
				sliced_f = np.array([sliced_f])
				sliced_t = np.array([sliced_t])
			elif sliced_data.ndim == 1:
				if isinstance(f_key, int):
					sliced_data = np.expand_dims(sliced_data, axis=0)
					sliced_f = np.array([sliced_f])
				if isinstance(t_key, int):
					sliced_data = np.expand_dims(sliced_data, axis=1)
					sliced_t = np.array([sliced_t])
					
			return self.__class__(S=sliced_data, f=sliced_f, t=sliced_t, title=self.title)
		
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
		
		return self.__class__(S=cropped_S, f=cropped_f, t=cropped_t, title=self.title)
	
	
	def plot(
		self,
		log_compression_factor: int | float | None = None,
		ax: plt.Axes | None = None,
		cmap: str = "gray_r",
		title: str | None = None,
		Mlabel: str | None = None,
		ylabel: str | None = "Frequency (hz)",
		xlabel: str | None = "Time (sec)",
		ylim: tuple[float, float] | None = None,
		xlim: tuple[float, float] | None = None,
		highlight: list[tuple[float, float, float, float]] | None = None,
		origin: str = "lower",  # or "lower"
		show_colorbar: bool = True,
		cax: plt.Axes | None = None,
		show_grid: bool = True,
		tick_mode: str = "center",  # "center" or "edge"
		n_ticks: tuple[int, int] | None = None,
	) -> plt.Figure:
		
		from modusa.io import Plotter
		
		title = title or self.title
	
		fig = Plotter.plot_matrix(
			M=self.S,
			r=self.f,
			c=self.t,
			log_compression_factor=log_compression_factor,
			ax=ax,
			cmap=cmap,
			title=title,
			Mlabel=Mlabel,
			rlabel=ylabel,
			clabel=xlabel,
			rlim=ylim,
			clim=xlim,
			highlight=highlight,
			origin=origin,
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
	
	#----------------------------
	# Math ops
	#----------------------------
	def __add__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = np.add(self.S, other_data)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __radd__(self, other):
		result = np.add(other, self.S)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __sub__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = np.subtract(self.S, other_data)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __rsub__(self, other):
		result = np.subtract(other, self.S)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __mul__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = np.multiply(self.S, other_data)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __rmul__(self, other):
		result = np.multiply(other, self.S)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __truediv__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = np.true_divide(self.S, other_data)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __rtruediv__(self, other):
		result = np.true_divide(other, self.S)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __floordiv__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = np.floor_divide(self.S, other_data)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __rfloordiv__(self, other):
		result = np.floor_divide(other, self.S)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __pow__(self, other):
		other_data = other.S if isinstance(other, self.__class__) else other
		result = np.power(self.S, other_data)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __rpow__(self, other):
		result = np.power(other, self.S)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def __abs__(self):
		result = np.abs(self.S)
		return self.__class__(S=result, f=self.f, t=self.t, title=self.title)
	
	def sin(self):
		"""Element-wise sine of the spectrogram."""
		return self.__class__(S=np.sin(self.S), f=self.f, t=self.t, title=self.title)
	
	def cos(self):
		"""Element-wise cosine of the spectrogram."""
		return self.__class__(S=np.cos(self.S), f=self.f, t=self.t, title=self.title)
	
	def exp(self):
		"""Element-wise exponential of the spectrogram."""
		return self.__class__(S=np.exp(self.S), f=self.f, t=self.t, title=self.title)
	
	def tanh(self):
		"""Element-wise hyperbolic tangent of the spectrogram."""
		return self.__class__(S=np.tanh(self.S), f=self.f, t=self.t, title=self.title)
	
	def log(self):
		"""Element-wise natural logarithm of the spectrogram."""
		return self.__class__(S=np.log(self.S), f=self.f, t=self.t, title=self.title)
	
	def log1p(self):
		"""Element-wise log(1 + M) of the spectrogram."""
		return self.__class__(S=np.log1p(self.S), f=self.f, t=self.t, title=self.title)
	
	def log10(self):
		"""Element-wise base-10 logarithm of the spectrogram."""
		return self.__class__(S=np.log10(self.S), f=self.f, t=self.t, title=self.title)
	
	def log2(self):
		"""Element-wise base-2 logarithm of the spectrogram."""
		return self.__class__(S=np.log2(self.S), f=self.f, t=self.t, title=self.title)
	
	
	def mean(self) -> float:
		"""Return the mean of the spectrogram values."""
		return float(np.mean(self.S))
	
	def std(self) -> float:
		"""Return the standard deviation of the spectrogram values."""
		return float(np.std(self.S))
	
	def min(self) -> float:
		"""Return the minimum value in the spectrogram."""
		return float(np.min(self.S))
	
	def max(self) -> float:
		"""Return the maximum value in the spectrogram."""
		return float(np.max(self.S))
	
	def sum(self) -> float:
		"""Return the sum of the spectrogram values."""
		return float(np.sum(self.S))
	
	