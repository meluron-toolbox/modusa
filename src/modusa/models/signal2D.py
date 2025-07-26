#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from .base import ModusaSignal
from .s_ax import SAx
from modusa.tools.math_ops import MathOps
from typing import Self, Any, Callable
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
import copy

class Signal2D(ModusaSignal):
	"""
	Space to represent 2D signal.

	Note
	----
	- The signal can have uniform/non-uniform axes.

	.. code:: python
		
		import modusa as ms
		import numpy as np
		signal_ax_0 = ms.sax(np.arange(100), label="My row axis (My axis unit)")
		signal_ax_1 = ms.sax(np.arange(20), label="My column Axis (My axis unit)")
		sax = (signal_ax_0, signal_ax_1)
		signal = ms.signal1D(data=np.random.random(100), data_label="My data (My data unit)", sax=sax, title="My Signal")
		signal_ax.print_info()
		signal.print_info()
	"""
	
	#--------Meta Information----------
	_name = "Signal 2D"
	_nickname = "Signal2D" # This is to be used in repr/str methods
	_description = "Space to represent 2D signal."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-20"
	#----------------------------------
	
	@validate_args_type()
	def __init__(
		self,
		data: np.ndarray,
		sax: tuple[SAx, SAx] | None,
		data_label: str,
		title: str
	):
		
		super().__init__() # Instantiating `ModusaSignal` class
		
		assert data.ndim == 2
		assert len(sax) == 2 # For 2 axes
		assert isinstance(sax[0], SAx) and isinstance(sax[1], SAx) # Both should be SAx instance
		
		assert data.shape[0] == sax[0].shape[0] and data.shape[1] == sax[1].shape[0], f"data and axis shape must match"
		
		self.__data = data
		self.__sax = sax
		self.__data_label = data_label
		self.__title = title
		
	#--------------------------------------
	# Properties (Hidden)
	#--------------------------------------
		
	@immutable_property("Use .set_meta_info method.")
	def _M(self) -> np.ndarray:
		"""Data array."""
		return self.__data
	
	@immutable_property("Use .set_meta_info method.")
	def _y(self) -> SAx:
		"""Axis corresponding to y."""
		return self.__sax[0]
	
	@immutable_property("Use .set_meta_info method.")
	def _x(self) -> np.ndarray:
		"""Axis corresponding to x"""
		return self.__sax[1]
	
	@immutable_property("Use .set_meta_info method.")
	def _M_label(self) -> str:
		"""Data array."""
		return self.__data_label
	
	@immutable_property("Use .set_meta_info method.")
	def _title(self) -> str:
		"""Data array."""
		return self.__title
		
	#===================================	
	
	#-----------------------------------
	# Properties (User facing)
	#-----------------------------------
	
	@immutable_property("Read only.")
	def shape(self) -> tuple:
		"""Shape of the data array."""
		return self._M.shape
	
	@immutable_property("Read only.")
	def ndim(self) -> tuple:
		"""Dimension of the data array. (2)"""
		return self._M.ndim # Should be 2
	
	#===================================
		
	#-----------------------------------
	# Setter
	#-----------------------------------
	
	@validate_args_type()
	def set_meta_info(self, data_label: str | None = None, title: str | None = None) -> None:
		"""
		Set meta info about the signal.

		Parameters
		----------
		data_label: str
			- Label for the data.
			- e.g. "Amplitude"
		title: str
			- Title for the signal
			- e.g. "Speech Signal"
		"""
		
		data_label = data_label if data_label is not None else self._M_label
		title = title if title is not None else self._title
		sax = (self._y, self._x)
		
		return self.__class__(data=self._M, data_label=data_label, sax=sax, title=title)
	
	#===================================
		
		
		
	#-----------------------------------
	# Visualisation
	#-----------------------------------
	
	def plot(
		self,
		ax: plt.Axes | None = None,
		cmap: str = "gray_r",
		title: str | None = None,
		y_lim: tuple[float, float] | None = None,
		x_lim: tuple[float, float] | None = None,
		highlight_regions: list[tuple[float, float, str], ...] | None = None,
		vlines: list | None = None,
		hlines: list | None = None,
		origin: str = "lower",  # or "lower"
		gamma: int | float | None = None,
		show_colorbar: bool = False,
		cax: plt.Axes | None = None,
		show_grid: bool = True,
		tick_mode: str = "center",  # "center" or "edge"
		n_ticks: tuple[int, int] | None = None,
	) -> "plt.Figure":
		"""
		Plot the 2DSignal using Matplotlib.
	
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
		ylim : tuple of float, optional
			Limits for the y-axis (frequency).
		xlim : tuple of float, optional
			Limits for the x-axis (time).
		highlight_regions : list[tuple[float, float, str]...] | None
			Regions to higlight (e.g. [(0, 10, 'tag')]
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
		import matplotlib.pyplot as plt
		
		M = self._M
		r = self._y._values
		c = self._x._values
		M_label = self._M_label
		r_label = self._y._label
		c_label = self._x._label
		title = title or self._title
		
		fig = Plotter.plot_matrix(M=M, r=r, c=c, ax=ax, cmap=cmap, title=title, M_label=M_label, r_label=r_label, c_label=c_label, r_lim=y_lim, c_lim=x_lim,
		highlight_regions=highlight_regions, vlines=vlines, hlines=hlines, origin=origin, gamma=gamma, show_colorbar=show_colorbar, cax=cax, show_grid=show_grid,
		tick_mode=tick_mode, n_ticks=n_ticks)
		
		return fig
		
	#===================================
	
	#-----------------------------------
	# Utility Methods
	#-----------------------------------
	
	def _is_same_as(self, other: Self) -> bool:
		"""
		Check if two `Signal1D` instances are same.
		"""
		assert isinstance(other, self.__class__)
		if not self._has_same_axis_as(other):
			return False
		if np.allclose(self.data, other.data):
			return True
		
		return False
	
	def _has_same_axis_as(self, other: Self) -> bool:
		"""
		Check if two 'Signal2D' instances have same
		axis (both axes). Many operations need to satify this.
		"""
		assert isinstance(other, self.__class__)
		result = self._y._is_same_as(other._y) and self._x._is_same_as(other._x)
		return result
	
	#-----------------------------------
	# Indexing support
	#-----------------------------------
	
	def __getitem__(self, key: slice | int | np.ndarray | Self) -> Self:
		raise NotImplementedError
		
	def __setitem__(self, key, value):
		raise NotImplementedError
		
	#======================================
		
	
	#-----------------------------------
	# Comparison operators return 
	# boolean masks
	#-----------------------------------
	
	def _perform_comparison_ops(self, other: int | float | np.generic, op: Callable):
		"""
		Perform comparison operations on 2D signals.
		"""
		from modusa.models.feature_time_domain_signal import FeatureTimeDomainSignal
		
		assert self.__class__ in [Signal2D, FeatureTimeDomainSignal]
		assert isinstance(other, (int, float, np.generic))
		
		if self.__class__ in [Signal2D]:
			M = self._M
			sax = copy.deepcopy((self._y, self._x)) # Making sure we pass a new copy of sax
			title = self._title
			
			if isinstance(other, (int, float)):
				mask = op(M, other)
				return self.__class__(data=mask, sax=sax, data_label="Mask", title=title) # sax is unchanges as we return mask only
			elif other.__class__ in [Signal2D]:
				assert self._has_same_axis_as(other)
				M_other = other._M
				mask = op(M, M_other)
				return self.__class__(data=mask, sax=sax, data_label="Mask", title=title)
			else:
				raise TypeError
				
		elif self.__class__ in [FeatureTimeDomainSignal]:
			M = self._M
			sr = self._sr
			t0 = self._t0
			t_label = self._t._label
			f = self._f._values
			f_label = self._f._label
			title = self._title
			
			if isinstance(other, (int, float)):
				mask = op(M, other)
				return FeatureTimeDomainSignal(data=mask, feature=f, frame_rate=sr, t0=t0, data_label="Mask", feature_label=f_label, time_label=t_label, title=title)
			elif other.__class__ in [TimeDomainSignal]:
				assert self._has_same_axis_as(other)
				M_other = other._M
				mask = op(M, M_other)
				return FeatureTimeDomainSignal(data=mask, feature=f, frame_rate=sr, t0=t0, data_label="Mask", feature_label=f_label, time_label=t_label, title=title)
			else:
				raise TypeError
		
		else:
			raise TypeError
			
			
			
	def __lt__(self, other: int | float | np.generic):
		return self._perform_comparison_ops(other, MathOps.lt)
	
	def __le__(self, other: int | float | np.generic):
		return self._perform_comparison_ops(other, MathOps.le)
	
	def __gt__(self, other: int | float | np.generic):
		return self._perform_comparison_ops(other, MathOps.gt)
	
	def __ge__(self, other: int | float | np.generic):
		return self._perform_comparison_ops(other, MathOps.ge)
	
	def __eq__(self, other: int | float | np.generic):
		return self._perform_comparison_ops(other, MathOps.eq)
	
	def __ne__(self, other: int | float | np.generic):
		return self._perform_comparison_ops(other, MathOps.ne)
	
	#======================================
	
	#-----------------------------------
	# Basic Math Operations
	#-----------------------------------
	
	def _perform_binary_ops(self, other, op: Callable, reverse=False):
		from modusa.models.feature_time_domain_signal import FeatureTimeDomainSignal
		
		assert self.__class__ in [Signal2D, FeatureTimeDomainSignal]
		assert isinstance(other, (int, float, np.generic, complex)) or other.__class__ in [Signal2D, FeatureTimeDomainSignal]
		
		if self.__class__ in [Signal2D]:
			assert isinstance(other, (int, float, np.generic, complex)) or other.__class__ in [Signal2D]
			M = self._M
			sax = copy.deepcopy((self._y, self._x))
			M_label = self._M_label
			title = self._title
			
			if isinstance(other, (int, float, np.generic, complex)):
				result = op(M, other) if not reverse else op(other, M)
				return self.__class__(data=result, sax=sax, data_label=M_label, title=title)
			elif other.__class__ in [Signal2D]:
				assert self._has_same_axis_as(other)
				M_other = other._M
				result = op(M, M_other) if not reverse else op(M_other, M)
				return self.__class__(data=result, sax=sax, data_label=M_label, title=title)
			else:
				raise TypeError
				
		elif self.__class__ in [FeatureTimeDomainSignal]:
			assert isinstance(other, (int, float, np.generic, complex)) or other.__class__ in [FeatureTimeDomainSignal]
			M = self._M
			f = self._f
			sr = self._sr
			t0 = self._t0
			M_label = self._M_label
			f_label = self._f._label
			t_label = self._t._label
			title = self._title
			
			if isinstance(other, (int, float, np.generic)):
				result = op(M, other) if not reverse else op(other, M)
				return self.__class__(data=result, feature=f, sr=sr, t0=t0, data_label=y_label, feature_label=f_label, time_label=t_label, title=title)
			elif other.__class__ in [FeatureTimeDomainSignal]:
				assert self._has_same_axis_as(other)
				M_other = other._M
				result = op(M, M_other) if not reverse else op(other, M_other)
				return self.__class__(data=result, feature=f, sr=sr, t0=t0, data_label=y_label, feature_label=f_label, time_label=t_label, title=title)
			else:
				raise TypeError
			
			
	def __add__(self, other):
		return self._perform_binary_ops(other, MathOps.add) 
	
	def __radd__(self, other):
		return self._perform_binary_ops(other, MathOps.add, reverse=True) 
	
	def __sub__(self, other):
		return self._perform_binary_ops(other, MathOps.subtract)
	
	def __rsub__(self, other):
		return self._perform_binary_ops(other, MathOps.subtract, reverse=True)
	
	def __mul__(self, other):
		return self._perform_binary_ops(other, MathOps.multiply)
	
	def __rmul__(self, other):
		return self._perform_binary_ops(other, MathOps.multiply, reverse=True)
	
	def __truediv__(self, other):
		return self._perform_binary_ops(other, MathOps.divide)
	
	def __rtruediv__(self, other):
		return self._perform_binary_ops(other, MathOps.divide, reverse=True)
	
	def __floordiv__(self, other):
		return self._perform_binary_ops(other, MathOps.floor_divide)
	
	def __rfloordiv__(self, other):
		return self._perform_binary_ops(other, MathOps.floor_divide, reverse=True)
	
	def __pow__(self, other):
		return self._perform_binary_ops(other, MathOps.power)
	
	def __rpow__(self, other):
		return self._perform_binary_ops(other, MathOps.power, reverse=True)
	
	#======================================
	
	#-----------------------------------
	# Unary Operations (Transformations)
	#-----------------------------------
	
	def _perform_unary_ops(self, op: Callable) -> Self:
		from modusa.models.feature_time_domain_signal import FeatureTimeDomainSignal
		
		assert self.__class__ in [Signal2D, FeatureTimeDomainSignal]
		
		if self.__class__ in [Signal2D]:
			M = self._M
			M_label = self._M_label
			sax = copy.deepcopy((self._y, self._x)) # Making sure we pass a new copy of sax
			title = self._title
			
			result = op(M)
			return self.__class__(data=result, sax=sax, data_label=M_label, title=title)
		
		elif self.__class__ in [FeatureTimeDomainSignal]:
			M = self._M
			f = self._f
			sr = self._sr
			t0 = self._t0
			M_label = self._M_label
			t_label = self._t._label
			f_label = self._f._label
			title = self._title
			
			result = op(M)
			return TimeDomainSignal(data=result, feature=f, sr=sr, t0=t0, data_label=M_label, feature_label=f_label, time_label=t_label, title=title)
		
		else:
			raise TypeError
			
	def abs(self) -> Self:
		"""Compute the element-wise abs of the signal data."""
		return self._perform_unary_ops(MathOps.abs)
	
	def sin(self) -> Self:
		"""Compute the element-wise sine of the signal data."""
		return self._perform_unary_ops(MathOps.sin)
	
	def cos(self) -> Self:
		"""Compute the element-wise cosine of the signal data."""
		return self._perform_unary_ops(MathOps.cos)
	
	def exp(self) -> Self:
		"""Compute the element-wise exponential of the signal data."""
		return self._perform_unary_ops(MathOps.exp)
	
	def tanh(self) -> Self:
		"""Compute the element-wise hyperbolic tangent of the signal data."""
		return self._perform_unary_ops(MathOps.tanh)
	
	def log(self) -> Self:
		"""Compute the element-wise natural logarithm of the signal data."""
		return self._perform_unary_ops(MathOps.log)
	
	def log1p(self) -> Self:
		"""Compute the element-wise natural logarithm of (1 + signal data)."""
		return self._perform_unary_ops(self, MathOps.log1p)
	
	def log10(self) -> Self:
		"""Compute the element-wise base-10 logarithm of the signal data."""
		return self._perform_unary_ops(MathOps.log10)
	
	def log2(self) -> Self:
		"""Compute the element-wise base-2 logarithm of the signal data."""
		return self._perform_unary_ops(MathOps.log2)
	
	def floor(self) -> Self:
		"""Apply np.floor to the signal data."""
		return self._perform_unary_ops(MathOps.floor)
	
	def ceil(self) -> Self:
		"""Apply np.ceil to the signal data."""
		return self._perform_unary_ops(MathOps.ceil)
	
	def round(self) -> Self:
		"""Apply np.round to the signal data."""
		return self._perform_unary_ops(MathOps.round)
	
	def real(self) -> Self:
		"""Apply np.real to the signal data."""
		return self._perform_unary_ops(MathOps.real)
	
	def imag(self) -> Self:
		"""Apply np.imag to the signal data."""
		return self._perform_unary_ops(MathOps.imag)
	
	def angle(self) -> Self:
		"""Apply np.angle to the signal data."""
		return self._perform_unary_ops(MathOps.angle)
	
	def __abs__(self):
		return self._perform_unary_ops(MathOps.abs)
	
	#===================================
	
	#-----------------------------------
	# Unary Operations (Aggregation)
	#-----------------------------------
	
	def _perform_aggregation_ops(self, op: Callable, axis: int | None = None):
		"""
		Perform various aggregation operations along an axis
		or element wise.
		
		Parameters
		----------
		axis:
			Axis along which aggregation is to be performed.
		Returns
		-------
		Signal1D | TimeDomainSignal | np.generic
		
		"""
		from modusa.models.signal1D import Signal1D
		from modusa.models.time_domain_signal import TimeDomainSignal
		from modusa.models.feature_time_domain_signal import FeatureTimeDomainSignal
		
		assert isinstance(axis, int) or axis is None, "Invalid axis"
		assert axis in [0, 1, -1, None], "Axis out of bound"
		assert isinstance(self, (Signal2D, FeatureTimeDomainSignal))
		
		M = self._M # This is a numpy array (2D)
		result = op(M, axis=axis)
		
		if axis is None: # We aggregated all the elements and got a signal value
			return result
		
		# Case 1: We are aggregating Signal2D object
		if self.__class__ == Signal2D: # After aggreation, we should return Signal1D
			y = self._y
			x = self._x
			y_label = self._y._label
			x_label = self._x._label
			title = self._title
			if axis == 0:
				return Signal1D(data=result, data_label=y_label, sax=(x, ), title=title)
			elif axis in [1, -1]:
				return Signal1D(data=result, data_label=x_label, sax=(y, ), title=title)
			else:
				raise excp.InputValueError()
		
		# Case 2: We are aggregating Signal2D object
		elif self.__class__ == FeatureTimeDomainSignal: # After aggreation, we should return either Signal1D or TimeDomainSignal based on axis
			y = self._y
			sr = self._sr
			t0 = self._t0
			t_label = self._t._label
			title = self._title
			if axis == 0: # We aggregated along axis 0 (row) => we should return TimeDomainSignal
				return TimeDomainSignal(data=result, data_label=y_label, sr=sr, t0=t0, time_label=t_label, title=title)
			elif axis in [1, -1]:
				return Signal1D(data=result, data_label=t_label, sax=(y, ), title=title)
			else:
				raise excp.InputValueError()
				
		
	
	def mean(self, axis: int | None = None):
		"""
		Compute the mean of the signal data
		along a given axis if passed else for all elements.
		"""
		return self._perform_aggregation_ops(op=MathOps.mean, axis=axis)
	
	def std(self, axis: int | None = None):
		"""
		Compute the standard deviation of the signal data
		along a given axis if passed else for all elements.
		"""
		return self._perform_aggregation_ops(op=MathOps.std, axis=axis)
	
	def min(self, axis: int | None = None):
		"""
		Compute the minimum value in the signal data
		along a given axis if passed else for all elements.
		"""
		return self._perform_aggregation_ops(op=MathOps.min, axis=axis)
	
	def max(self, axis: int | None = None):
		"""
		Compute the maximum value in the signal data
		along a given axis if passed else for all elements.
		"""
		return self._perform_aggregation_ops(op=MathOps.max, axis=axis)
	
	def sum(self, axis: int | None = None):
		"""
		Compute the sum of the signal data
		along a given axis if passed else for all elements.
		"""
		return self._perform_aggregation_ops(op=MathOps.sum, axis=axis)
	
	#===================================
	
	#-----------------------------------
	# Information
	#-----------------------------------
	
	def print_info(self) -> None:
		"""Print key information about the spectrogram signal."""
		
		print("-"*50)
		print(f"{'Title':<20}: {self.title}")
		print("-"*50)
		print(f"{'Type':<20}: {self.__class__.__name__}")
		print(f"{'Shape':<20}: {self.shape} (freq bins × time frames)")
		
		# Inheritance chain
		cls_chain = " → ".join(cls.__name__ for cls in reversed(self.__class__.__mro__[:-1]))
		print(f"{'Inheritance':<20}: {cls_chain}")
		print("=" * 50)
		
	def __str__(self):
		M = self._M
		shape = self.shape
		label = self._M_label
		y = self._y
		x = self._x
		
		arr_str = np.array2string(
			M,
			separator=", ",
			threshold=20,       # limit number of elements shown
			edgeitems=2,          # show first/last 2 rows and columns
			max_line_width=120,   # avoid wrapping
		)
		return "=======\n" + f"{self._nickname}({arr_str}, shape={shape}, label={label})\n-------\n{y}\n{x})" + "\n======="
	
	def __repr__(self):
		M = self._M
		shape = self.shape
		label = self._M_label
		y = self._y
		x = self._x
		
		arr_str = np.array2string(
			M,
			separator=", ",
			threshold=20,       # limit number of elements shown
			edgeitems=2,          # show first/last 2 rows and columns
			max_line_width=120,   # avoid wrapping
		)
		return "=======\n" + f"{self._nickname}({arr_str}, shape={shape}, label={label})\n-------\n{y}\n{x})" + "\n======="
	
	#===================================