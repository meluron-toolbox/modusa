#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from .base import ModusaSignalAxis
from modusa.tools.math_ops import MathOps
import modusa as ms
from typing import Self, Any, Callable
import numpy as np
import matplotlib.pyplot as plt

class SAx(ModusaSignalAxis):
	"""
	Space to represent signal's axis.

	Note
	----
	- All the signal class needs to have `SAx` instance to store a meaningful axis.
	- It acts as a middle layer between indexes and the signal.
	- 	SIGNAL: x1, x2, x3, ..., xn \n
		SAx:	v1, v2, v3, ..., vn [Can be any values that has some meaning for different signal] \n
		Index:	1, 	2,  3,  ..., n
	"""
	
	#--------Meta Information----------
	_name = "Signal Axis"
	_nickname = "SAx" # This is to be used in repr/str methods
	_description = "Axis for different types of signals."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-20"
	#----------------------------------
	
	@validate_args_type()
	def __init__(self, values: np.ndarray, label: str):
		
		super().__init__() # Instantiating `ModusaSignal` class
		
		assert values.ndim == 1
		assert isinstance(label, str)
		
		self.__values = values
		self.__label = label
		
	#------------------------------------
	# Properties (Hidden)
	#------------------------------------
	@immutable_property("Read only property")
	def _values(self) -> np.ndarray:
		return self.__values
	
	@immutable_property("Use .set_meta_info method.")
	def _label(self) -> str:
		return self.__label
	
	@immutable_property("Read only property")
	def _is_uniform(self) -> bool:
		values = self._values
		if len(values) == 1:
			return True
		elif len(values) > 1:
			diffs = np.diff(values)
			if np.allclose(diffs, diffs[0]): # Maybe we can later change it to look for any instead of all to save some compute
				return True
		return False
	
	#====================================
	
	#-----------------------------------
	# Properties (User Facing)
	#-----------------------------------
	
	@immutable_property("Read only.")
	def shape(self) -> tuple:
		"""Shape of the axis."""
		return self._values.shape
	
	@immutable_property("Read only.")
	def ndim(self) -> tuple:
		"""Dimension of the axis array (=1)"""
		return self._values.ndim # Should be 1
	
	#====================================
	
	#------------------------------------
	# Setter
	#------------------------------------
	def set_meta_info(self, label: str = None) -> Self:
		"""
		Set meta info about the axis.

		Parameters
		----------
		label: str
			Label for the axis (e.g. "Time (sec)")
		Returns
		-------
		SAx
			SAx instance with new label
		"""
		label = str(label) if label is not None else self._label
		values = self._values
		
		return self.__class__(values=values, label=label)
	
	#====================================
	
	
	#------------------------------------
	# Visualisation
	#------------------------------------
	
	def plot(self, ax: plt.Axes | None = None, color: str = "b") -> plt.Figure | None:
		"""
		Plot vertical lines showing the axis.

		Parameters
		----------
		ax: plt.Axes | None
			- Incase, you want to plot it on your defined matplot ax.
			- If not provided, set to None, meaning we create a new figure and return that.
		color: str
			- Color of the vertival lines.
			- Useful while plotting multiple SAx instances on the same plot.

		Returns
		-------
		plt.Figure | None
			- Figure if ax is None.
			- None is ax is not None.
		"""
		from modusa.tools.plotter import Plotter
		values = self._values
		label = self._label
		
		fig: plt.Figure | None = Plotter.plot_event(event=values, ax=ax, title=label, y_label="", x_label=label, color=color)
		
		return fig
	#====================================

	
	#------------------------------------
	# Utility methods
	#------------------------------------
	
	def _is_same_as(self, other_sax: Self) -> bool:
		"""
		Check if two axes are same.
		"""
		assert isinstance(other_sax, self.__class__)
		if self.shape != other_sax.shape:
			return False
		if np.allclose(self._values, other_sax._values):
			return True
		return False
	
	#====================================
	
	#-----------------------------------
	# Indexing
	#-----------------------------------
	
	def __getitem__(self, key: slice | int | np.ndarray) -> Self:
		from modusa.models.signal1D import Signal1D
		
		assert key.__class__ in [slice, int, np.ndarray, Signal1D], "Invalid key"
		
		values = self._values
		label = self._label
	
		if key.__class__ in [slice, int, np.ndarray]:
			sliced_values = values[key]
		elif key.__class__ in [Signal1D]:
			mask = key._y
			sliced_values = values[mask]
		
		if isinstance(sliced_values, (int, float, np.generic)): sliced_values = [sliced_values]
		sliced_values = np.asarray(sliced_values)
		
		return self.__class__(values=sliced_values, label=label)
	
	#====================================
	
	#-----------------------------------
	# Comparison operators return 
	# boolean masks
	#-----------------------------------
	
	def _perform_comparison_ops(self, other: int | float | np.generic, op: Callable):
		
		assert self.__class__ in [SAx]
		assert other.__class__ in [int, float] or isintance(other, np.generic)
		
		label = self._label
		mask = op(self._values, other)
		
		# We should return it as signal1D with data as mask and sax
		from modusa.models.signal1D import Signal1D
		sax = self # We pass on the entire SAx while creating the boolean signal
		
		return Signal1D(data=mask, data_label="Mask", sax=(sax, ), title="Boolean Mask")
		
	
	def __lt__(self, other: int | float | np.generic) -> "Signal1D":
		return self._perform_comparison_ops(other=other, op=MathOps.lt)
	
	def __le__(self, other: int | float | np.generic) -> "Signal1D":
		return self._perform_comparison_ops(other=other, op=MathOps.le)
	
	def __gt__(self, other: int | float | np.generic) -> "Signal1D":
		return self._perform_comparison_ops(other=other, op=MathOps.gt)
	
	def __ge__(self, other: int | float | np.generic) -> "Signal1D":
		return self._perform_comparison_ops(other=other, op=MathOps.ge)
	
	def __eq__(self, other: int | float | np.generic) -> "Signal1D":
		return self._perform_comparison_ops(other=other, op=MathOps.eq)
	
	def __ne__(self, other: int | float | np.generic) -> "Signal1D":
		return self._perform_comparison_ops(other=other, op=MathOps.ne)
	
	#======================================
	
	
	#-----------------------------------
	# Basic Math Operations
	#-----------------------------------
	
	def _perform_binary_ops(self, other: int | float, op: Callable, reverse=False):
		from modusa.models.signal1D import Signal1D
		
		assert self.__class__ in [SAx]
		assert other.__class__ in [int, float, complex]
		
		values = self._values
		label = self._label
		
		new_sax = self.__class__(values=values, label=label) # We make sure that we create a new instance rather than self
		
		if isinstance(other, (int, float, complex)):
			result = op(values, other) if not reverse else op(other, values)
		else:
			raise excp.InputTypeError("Unsupported operand type")

		return Signal1D(data=result, sax=(new_sax, ), data_label="Y", title="Signal 1D")
			
			
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
		from modusa.models.signal1D import Signal1D
		assert self.__class__ in [SAx]
		
		values = self._values
		label = self._label
		
		new_sax = self.__class__(values=values, label=label)
		result = op(values)
		
		return Signal1D(data=result, data_label="Y", sax=(new_sax, ), title="Signal 1D")

			
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
	
	def __abs__(self):
		return self._perform_unary_ops(MathOps.abs)
	
	#===================================
	
	#----------------------------------
	# Information
	#----------------------------------
	
	def print_info(self) -> None:
		"""Prints info about the audio."""
		print("-" * 50)
		print("Axis Info")
		print("-" * 50)
		print(f"{'Label':<20}: {self._label}")
		print(f"{'Shape':<20}: {self._values.shape}")
		print(f"{'Start Value':<20}: {self._values[0]:.2f}")
		print(f"{'End Value':<20}: {self._values[-1]:.2f}")
		# Inheritance chain
		cls_chain = " → ".join(cls.__name__ for cls in reversed(self.__class__.__mro__[:-1]))
		print(f"{'Inheritance':<20}: {cls_chain}")
		print("=" * 50)
	
	def __str__(self):
		data = self._values
		label = self._label
		shape = self.shape
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=30,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
		)
		
		return f"{self._nickname}({arr_str}, shape={shape}, label={label})"
	
	def __repr__(self):
		data = self._values
		label = self._label
		shape = self.shape
		
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=30,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
		)
		
		
		return f"{self._nickname}({arr_str}, shape={shape}, label={label})"
	#===================================