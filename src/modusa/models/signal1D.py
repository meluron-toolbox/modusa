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

class Signal1D(ModusaSignal):
	"""
	Space to represent any 1D Signal.
	
	Note
	----
	- The signal can have uniform/non-uniform axis.


	.. code:: python
		
		import modusa as ms
		import numpy as np
		signal_ax = ms.sax(np.arange(100), label="My Axis (My axis unit)")
		signal = ms.signal1D(data=np.random.random(100), data_label="My data (My data unit)", title="My Signal")
		signal_ax.print_info()
		signal.print_info()

	Parameters
	----------
	data: np.ndarray | list | int | float
		- Data array
		- We convert any other types (list | int | float) to `np.ndarray` in the constructor.
	sax: SAx | tuple(Sax) | None
		- Axis for the signal.
		- You can also pass tuple containing 1 element. We anyway convert it into a tuple.
		- If not passed, we create a `SAx` instance with integer indices.
	data_label: str
		- What does the data represent? 
		- e.g. "Amplitude (dB)".
		- We use this as ylabel for plots.
	title: str
		- What does the signal represent?
		- e.g. "MySignal"
		- This is used as the title for plot.
	"""
	
	#--------Meta Information----------
	_name = "Signal 1D"
	_nickname = "Signal1D" # This is to be used in repr/str methods
	_description = "Space to represent any 1D Signal."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-20"
	#----------------------------------
	
	@validate_args_type()
	def __init__(
		self,
		data: np.ndarray,
		sax: tuple[SAx],
		data_label: str,
		title: str
	):
		super().__init__() # Instantiating `ModusaSignal` class
		
		assert data.ndim == 1
		assert len(sax) == 1 and isinstance(sax[0], SAx) # It should be tuple of length 1 with SAx instance

		assert data.shape == sax[0].shape, f"Data and axis shape must match"
		
		# All these are private and we do not expose it to users directly.
		self.__data = data
		self.__sax = sax
		self.__data_label = data_label
		self.__title = title
		
	
	#--------------------------------------
	# Properties (Hidden)
	#--------------------------------------
	
	@immutable_property("Use .set_meta_info method.")
	def _y(self) -> str:
		"""Data array."""
		return self.__data
	
	@immutable_property("Use .set_meta_info method.")
	def _y_label(self) -> str:
		"""Data array."""
		return self.__data_label
	
	@immutable_property("Use .set_meta_info method.")
	def _x(self) -> str:
		"""Data array."""
		return self.__sax[0]
	
	@immutable_property("Use .set_meta_info method.")
	def _title(self) -> str:
		"""Data array."""
		return self.__title
	
	#-----------------------------------
	# Properties (User Facing)
	#-----------------------------------
	
	@immutable_property("Read only.")
	def shape(self) -> tuple:
		"""Shape of the data array."""
		return self._y.shape
	
	@immutable_property("Read only.")
	def ndim(self) -> tuple:
		"""Dimension of the data array. (1)"""
		return self._y.ndim # Should be 1
	
	#===================================
	
	#-----------------------------------
	# Setter
	#-----------------------------------
	
	def set_meta_info(self, data_label: str | None = None, title: str | None = None) -> None:
		"""
		Set meta info about the signals.

		Parameters
		----------
		data_label: str
			- Label for the data.
			- e.g. "Amplitude"
		title: str
			- Title for the signal
			- e.g. "Speech Signal"
		"""
		data_label = str(data_label) if data_label is not None else self._y_label
		title = str(title) if title is not None else self._title
		
		return self.__class__(data=self.data, data_label=data_label, sax=self.sax, title=title)
	
	#===================================
	
	#-----------------------------------
	# Utility Methods
	#-----------------------------------
	
	def _is_same_as(self, other_signal: Self) -> bool:
		"""
		Check if two `Signal1D` instances are equal.
		"""
		raise NotImplementedError
		assert isinstance(other_signal, self.__class__)
		if not self._sax[0].is_same_as(other_signal._sax[0]):
			return False
		if np.allclose(self.data, other_signal.data):
			return True
		
		return False
	
	def _has_same_axis_as(self, other: Self) -> bool:
		"""
		Check if two 'Signal1D' instances have same
		axis. Many operations need to satify this.
		"""
		assert isinstance(other, self.__class__)
		return self._sax[0]._is_same_as(other._sax[0])
	
	#===================================
	
	
	#-----------------------------------
	# Tools
	#-----------------------------------
	
	def plot(
		self,
		ax: plt.Axes | None = None,
		fmt: str = "k-",
		title: str | None = None,
		y_label: str | None = None,
		x_label: str | None = None,
		y_lim: tuple[float, float] | None = None,
		x_lim: tuple[float, float] | None = None,
		highlight_regions: list[tuple[float, float, str]] | None = None,
		vlines: list[float] | None = None,
		hlines: list[float] | None = None,
		legend: str | tuple[str, str] | None = None,
		show_grid: bool = False,
		show_stem: bool = False,
	) -> plt.Figure | None:
		"""
		Plot the signal.
		
		.. code-block:: python
		
			import modusa as ms
			import numpy as np
			signal = ms.signal1D(data=np.random.random(100), data_label="My data (unit)", title="My Random Signal")
			display(signal.plot())
		
		Parameters
		----------
		ax : matplotlib.axes.Axes | None
			- If you want to plot the signal on a given matplotlib axes, you can pass the ax here. We do not return any figure in this case.
			- If not passed, we create a new figure, plots the signal on it and then return the figure.
		fmt : str
			- Format of the plot as per matplotlib standards (Eg. "k-" or "blue--o)
			- Default is "k-"
		title : str | None
			- Title for the plot.
			- If not passed, we use the default set during signal instantiation.
		y_label : str | None
			- Label for the y-axis.
			- If not passed, we use the default set during signal instantiation.
		x_label : str | None
			- Label for the x-axis.
			- If not passed, we use the default set during signal instantiation.
		y_lim : tuple[float, float] | None
			- Limits for the y-axis.
		x_lim : tuple[float, float] | None
			- Limits for the x-axis.
		highlight_regions : list[tuple[float, float, str]] | None
			- List of time intervals to highlight on the plot.
			- [(start, end, 'tag')]
		vlines: list[float]
			- List of x values to draw vertical lines.
			- e.g. [10, 13.5]
		hlines: list[float]
			- List of data values to draw horizontal lines.
			- e.g. [10, 13.5]
		show_grid: bool
			- If true, shows grid.
		show_stem : bool
			- If True, use a stem plot instead of a continuous line. 
			- Autorejects if signal is too large.
		legend : str | tuple[str, str] | None
			- If provided, adds a legend at the specified location.
			- e.g., "signal" -> gets converted into ("signal", "best")
			- e.g. ("signal", "upper right")
		
		Returns
		-------
		matplotlib.figure.Figure | None
			- The figure object containing the plot.
			- None in case an axis is provided.
		"""
		
		from modusa.tools.plotter import Plotter
		
		y = self._y
		if y_label is None: y_label = self._y_label
		x = self._x._values
		if x_label is None: x_label = self._x._label
		if title is None: title = self._title
		
		
		fig: plt.Figure | None = Plotter.plot_signal(y=y, x=x, ax=ax, fmt=fmt, title=title, y_label=y_label, x_label=x_label, y_lim=y_lim, x_lim=x_lim, highlight_regions=highlight_regions, vlines=vlines, hlines=hlines, show_grid=show_grid, show_stem=show_stem, legend=legend)
		
		return fig
	
	#===================================
	
	
	#-----------------------------------
	# Indexing
	#-----------------------------------
	
	def __getitem__(self, key: slice | int | np.ndarray | Self) -> Self:
		from modusa.models.time_domain_signal import TimeDomainSignal
		
		# Since, we need to make sure that this method is valid for all the subclasses, we make strict checking at the start
		assert self.__class__ in [Signal1D, TimeDomainSignal] # We do this explicitely rather that using inheritance logic to make sure that the implementation covers other subclasses
		assert key.__class__ in [slice, int, np.ndarray, Signal1D, TimeDomainSignal] or isintance(key, np.generic), "Invalid key"
		
		y = self._y
		y_label = self._y_label
		x = self._x
		x_label = self._x._label
		title = self._title
		
		y_label = "mask" # Resetting y_label
		
		# Case 1: self is Signal1D so we must return that
		if self.__class__ == Signal1D:
			# If key is int
			if key.__class__ in [int, slice]:
				sliced_data = y[key]
				sliced_sax = x[key]
				return Signal1D(data=sliced_data, sax=(sliced_sax, ), data_label=y_label, title=title)
		
			# If key is "Signal1D" with boolean mask
			elif key.__class__ in [Signal1D]:
				raise NotImplementedError
			
			# If key is np.array with boolean mask
			elif key.__class__ in [np.ndarray]:
				raise NotImplementedError
			
		# Case 2: self is TimeDomainSignal so we must return that
		if self.__class__ == TimeDomainSignal:
			sr = self._sr
			t0 = self._t0
			t = self._t
			# If key is int
			if key.__class__ in [int, slice]:
				sliced_data = y[key]
				sliced_t = t[key]
				new_t0 = sliced_t._values[0]
				return TimeDomainSignal(data=sliced_data, sr=sr, t0=new_t0, data_label=y_label, time_label=x_label, title=title)
			
			# If key is "TimeDomainSignal" with boolean mask
			elif key.__class__ in [TimeDomainSignal]:
				raise NotImplementedError
			
			# If key is np.array with boolean mask
			elif key.__class__ in [np.ndarray]:
				raise NotImplementedError
		
	def __setitem__(self, key, value):
		raise NotImplementedError
		self.__data[key] = value # TODO
	
	#===================================
	
	#-----------------------------------
	# Comparison operators return 
	# boolean masks
	#-----------------------------------
	
	def _perform_comparison_ops(self, other: int | float | np.generic, op: Callable):
		"""
		Performs comparison operations like >, <, >=, <=, ==, !=
		"""
		from modusa.models.time_domain_signal import TimeDomainSignal
		
		assert self.__class__ in [Signal1D, TimeDomainSignal]
		assert isinstance(other, (int, float, np.generic))
		
		if self.__class__ in [Signal1D]:
			y = self._y
			sax = copy.deepcopy(self._sax) # Making sure we pass a new copy of sax
			title = self._title
		
			if isinstance(other, (int, float)):
				mask = op(y, other)
				return self.__class__(data=mask, sax=sax, data_label="Mask", title=title) # sax is unchanges as we return mask only
			elif other.__class__ in [Signal1D]:
				assert self._has_same_axis_as(other)
				y_other = other._y
				mask = op(y, y_other)
				return self.__class__(data=mask, sax=sax, data_label="Mask", title=title)
			else:
				raise TypeError
				
		elif self.__class__ in [TimeDomainSignal]:
			y = self._y
			sr = self._sr
			t0 = self._t0
			t_label = self._t._label
			title = self._title
			
			if isinstance(other, (int, float)):
				mask = op(y, other)
				return TimeDomainSignal(data=mask, sr=sr, t0=t0, data_label="Mask", time_label=t_label, title=title)
			elif other.__class__ in [TimeDomainSignal]:
				assert self._has_same_axis_as(other)
				y_other = other._y
				mask = op(y, y_other)
				return TimeDomainSignal(data=mask, sr=sr, t0=t0, data_label="Mask", time_label=t_label, title=title)
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
		"""
		Perform binary operations like +, -, *, ...
		"""
		from modusa.models.time_domain_signal import TimeDomainSignal
		
		assert self.__class__ in [Signal1D, TimeDomainSignal]
		assert isinstance(other, (int, float, np.generic)) or other.__class__ in [Signal1D, TimeDomainSignal]
		
		if self.__class__ in [Signal1D]:
			assert isinstance(other, (int, float, np.generic)) or other.__class__ in [Signal1D]
			y = self._y
			sax = copy.deepcopy(self._sax)
			y_label = self._y_label
			x_label = self._x_label
			title = self._title
			
			if isinstance(other, (int, float, np.generic)):
				result = op(y, other) if not reverse else op(other, y)
				return self.__class__(data=result, sax=sax, data_label=y_label, title=title)
			elif other.__class__ in [Signal1D]:
				assert self._has_same_axis_as(other)
				other = other._y
				result = op(y, other) if not reverse else op(other, y)
				return self.__class__(data=result, sax=sax, data_label=y_label, title=title)
			else:
				raise TypeError
			
		elif self.__class__ in [TimeDomainSignal]:
			assert isinstance(other, (int, float, np.generic)) or other.__class__ in [TimeDomainSignal]
			y = self._y
			sr = self._sr
			t0 = self._t0
			y_label = self._y_label
			t_label = self._t._label
			title = self._title
			
			if isinstance(other, (int, float, np.generic)):
				result = op(y, other) if not reverse else op(other, y)
				return self.__class__(data=result, sr=sr, t0=t0, data_label=y_label, time_label=t_label, title=title)
			elif other.__class__ in [TimeDomainSignal]:
				assert self._has_same_axis_as(other)
				other = other._y
				result = op(y, other) if not reverse else op(other, y)
				return self.__class__(data=result, sr=sr, t0=t0, data_label=y_label, time_label=t_label, title=title)
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
		"""
		Perform unary operations like sin, cos, ...
		"""
		from modusa.models.time_domain_signal import TimeDomainSignal
		
		assert self.__class__ in [Signal1D, TimeDomainSignal]
		
		if self.__class__ in [Signal1D]:
			y = self._y
			sax = copy.deepcopy(self._sax) # Making sure we pass a new copy of sax
			title = self._title

			result = op(y)
			return self.__class__(data=result, sax=sax, data_label="Mask", title=title)
		
		elif self.__class__ in [TimeDomainSignal]:
			y = self._y
			sr = self._sr
			t0 = self._t0
			t_label = self._t._label
			title = self._title
			
			result = op(y)
			return TimeDomainSignal(data=result, sr=sr, t0=t0, data_label="Mask", time_label=t_label, title=title)
		
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
	# We return np.generic type so that
	# we can smoothly perform other 
	# operations with them like max norm.
	#-----------------------------------
	
	def mean(self) -> "np.generic":
		"""Mean of the data array."""
		return MathOps.mean(self._y)
	
	def std(self) -> "np.generic":
		"""Standard deviation of the data array."""
		return MathOps.std(self._y)
	
	def min(self) -> "np.generic":
		"""Minimum value in the data array."""
		return MathOps.min(self._y)
	
	def max(self) -> "np.generic":
		"""Maximum value in the data array."""
		return MathOps.max(self._y)
	
	def sum(self) -> "np.generic":
		"""Sum of the data array."""
		return MathOps.sum(self._y)
	
	#===================================
	
	#----------------------------------
	# Information
	#----------------------------------
	
	def print_info(self) -> None:
		"""Prints info about the audio."""
		print("-" * 50)
		print(f"{'Title'}: {self._title}")
		print("-" * 50)
		print(f"{'Type':<20}: {self.__class__.__name__}")
		print(f"{'Shape':<20}: {self.shape}")
		
		# Inheritance chain
		cls_chain = " → ".join(cls.__name__ for cls in reversed(self.__class__.__mro__[:-1]))
		print(f"{'Inheritance':<20}: {cls_chain}")
		print("=" * 50)
	
	def __str__(self):
		y = self._y
		y_label = self._y_label
		shape = self.shape
		x = self._x
		
		arr_str = np.array2string(
			y,
			separator=", ",
			threshold=30,       # limit number of elements shown
			edgeitems=2,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
		)
		
		return "=======\n" + f"{self._nickname}({arr_str}, shape={shape}, label={y_label})\n-------\n{x})" + "\n======="
		
	def __repr__(self):
		y = self._y
		y_label = self._y_label
		shape = self.shape
		x = self._x
		
		arr_str = np.array2string(
			y,
			separator=", ",
			threshold=30,       # limit number of elements shown
			edgeitems=2,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
		)
		return "=======\n" + f"{self._nickname}({arr_str}, shape={shape}, label={y_label})\n-------\n{x})" + "\n======="
	#===================================