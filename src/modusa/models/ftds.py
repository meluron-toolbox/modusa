#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from .s2d import S2D
from .s_ax import SAx
from .t_ax import TAx
from .data import Data
from modusa.tools.math_ops import MathOps
from typing import Self, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
import modusa as ms

class FTDS(S2D):
	"""
	Space to represent feature time domain signal (2D).

	Note
	----
	- Use :class:`~modusa.generators.ftds.FTDSGen` API to instantiate this class.
	- The signal must have uniform time axis thus `TAx`.

	Parameters
	----------
	M: Data
		- Data object holding the main 2D array.
	f: SAx
		- Feature-axis of the 2D signal.
	t: TAx
		- Time-axis of the 2D signal.
	title: str
		- What does the signal represent?
		- e.g. "MySignal"
		- This is used as the title while plotting.
	"""
	
	#--------Meta Information----------
	_name = "Feature Time Domain Signal"
	_nickname = "FTDS" # This is to be used in repr/str methods
	_description = "Space to represent feature time domain signal (2D)."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-21"
	#----------------------------------
	
	def __init__(self, M, f, t, title = None):
		
		if not (isinstance(M, Data) and isinstance(f, SAx), isinstance(t, TAx)):
			raise TypeError(f"`M` must be `Data` instance, `f` and `x` must be `SAx` and `TAx` instances, got {type(M)}, {type(f)} and {type(t)}")
		
		super().__init__(M=M, y=f, x=t, title=title) # Instantiating `ModusaSignal` class
	
	#--------------------------------------
	# Properties
	#--------------------------------------
		
	@property
	def M(self) -> Data:
		return self._M
	
	@property
	def f(self) -> SAx:
		return self._y
	
	@property
	def t(self) -> SAx:
		return self._x
	
	@property
	def title(self) -> str:
		return self._title
	
	@property
	def shape(self) -> tuple:
		return self.M.shape
	
	@property
	def ndim(self) -> tuple:
		return self.M.ndim # Should be 2
	
	#===================================
	
		
	#-------------------------------
	# NumPy Protocol
	#-------------------------------
	
	def __array__(self, dtype=None) -> np.ndarray:
		return np.asarray(self.M.values, dtype=dtype)
	
	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		"""
		Supports NumPy universal functions on the Signal1D object.
		"""
		from .data import Data  # Ensure this is the same Data class you're using
		from modusa.utils import np_func_cat as nfc
		
		raw_inputs = [
			np.asarray(obj.M) if isinstance(obj, type(self)) else obj
			for obj in inputs
		]
		
		result = getattr(ufunc, method)(*raw_inputs, **kwargs)
		
		result = Data(values=result, label=None)
		f = self.f.copy()
		t = self.t.copy()
		
		if result.shape[0] != f.shape[0] or result.shape[1] != t.shape[0]:
			raise ValueError(f"`{ufunc.__name__}` caused shape mismatch between data and axis, please create a github issue")
			
		return self.__class__(M=result, f=f, t=t, title=self.title)
	
	def __array_function__(self, func, types, args, kwargs):
		"""
		Additional numpy function support for modusa signals.
		Handles reduction and ufunc-like behavior.
		"""
		from .data import Data
		from modusa.utils import np_func_cat as nfc
		
		if not all(issubclass(t, type(self)) for t in types):
			return NotImplemented
		
		if func in nfc.CONCAT_FUNCS:
			raise NotImplementedError(f"`{func.__name__}` is not yet tested on modusa signal, please create a GitHub issue.")
			
		obj = args[0]
		M_arr = np.asarray(obj.M)
		
		axis = kwargs.get("axis", None)
		keepdims = kwargs.get("keepdims", False)
		
		result = func(M_arr, **kwargs)
		result = Data(values=result, label=None)
		
		if func in nfc.REDUCTION_FUNCS:
			# Case 1: Full reduction → scalar or 0D result
			if axis is None or result.ndim == 0:
				return result
			
			# Case 2: Reduced to 1D → return S1D with correct axis
			elif result.ndim == 1:
				from .s1d import S1D
				from .s2d import TDS
				
				if axis == 0:
					return TDS(y=result, t=self.t, title=None)
				elif axis == 1:
					return S1D(y=result, x=self.f, title=None)
				else:
					raise ValueError(f"Unsupported axis={axis} for reduction on shape {signal_array.shape}")
			
			# Case 3: Reduction keeps both axes (unlikely)
			else:
				raise NotImplementedError(f"{func.__name__} result shape={result.shape} not handled for modusa signal")
				
		elif func in nfc.X_NEEDS_ADJUSTMENT_FUNCS:
			raise NotImplementedError(f"{func.__name__} requires x-axis adjustment logic.")
			
		else:
			raise NotImplementedError(f"`{func.__name__}` is not yet tested on modusa signal, please create a GitHub issue.")
			
			
	#================================
	
	#-------------------------------
	# Indexing
	#-------------------------------
			
	def __getitem__(self, key):
		"""
		Return a sliced or indexed view of the data.
		
		Parameters
		----------
		key : int | slice | S2D
			- Index to apply to the values.
		
		Returns
		-------
		S2D | S1D | Data
			- S2D object if slicing results in 2D array
			- S1D object if slicing results in 1D array
			- Data if slicing results in scalar
		"""
		from .s1d import S1D
		from .tds import TDS
		
		if isinstance(key, S1D):
			raise TypeError(f"Applying `S1D` mask on `S2D` is not allowed.")
			
		# We slice the data
		sliced_M = self.M[key]
		
		# Case 1: Row indexing only — return a horizontal slice (1D view across columns)
		if isinstance(key, int):
			sliced_f = self.f[key]
			sliced_t = self.t
			return TDS(y=sliced_M, x=sliced_t, title=self.title)
		
		# Case 2: Column indexing only — return a vertical slice (1D view across rows)
		elif isinstance(key, slice):
			sliced_f = self.f[key]
			sliced_t = self.t
			return self.__class__(M=sliced_M, f=sliced_f, t=sliced_t, title=self.title)
		
		# Case 3: 2D slicing
		elif isinstance(key, tuple) and len(key) == 2:
			row_key, col_key = key
			if isinstance(row_key, int) and isinstance(col_key, int):
				# Single value extraction → shape = (1, 1)
				# Will return data object
				return Data(values=sliced_M, title=self.title)
			
			elif isinstance(row_key, int) and isinstance(col_key, slice):
				# Row vector → return S1D
				sliced_f = self.f[row_key] # Scalar
				sliced_t = self.t[col_key]
				
				return TDS(y=sliced_M, t=sliced_t, title=self.title)
			
			elif isinstance(row_key, slice) and isinstance(col_key, int):
				# Column vector → return S1D
				sliced_f = self.f[row_key]
				sliced_t = self.t[col_key] # Scalar
				
				return S1D(y=sliced_M, x=sliced_f, title=self.title)
			
			elif isinstance(row_key, slice) and isinstance(col_key, slice):
				# 2D slice → return same class
				sliced_f = self.f[row_key]
				sliced_t = self.t[col_key] 
				
				return self.__class__(M=sliced_M, f=sliced_f, t=sliced_t, title=self.title)
		
		# Case 4: Boolean masking signal
		elif isinstance(key, type(self)):
			sliced_f = self.f
			sliced_t = self.t
			
			return self.__class__(M=sliced_M, f=sliced_f, t=sliced_t, title=self.title)
		
		else:
			raise TypeError(f"Unsupported index type: {type(key)}")
			
	def __setitem__(self, key, value):
		"""
		Set values at the specified index.
	
		Parameters
		----------
		key : int | slice | array-like | boolean array | S1D
			Index to apply to the values.
		value : int | float | array-like
			Value(s) to set.
		"""
		
		self.M[key] = value  # In-place assignment
		
	#===================================
		
	#-------------------------------
	# Basic arithmetic operations
	#-------------------------------
	def __add__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.add(self, other) 
	
	def __radd__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.add(other, self)
	
	def __sub__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.subtract(self, other)
	
	def __rsub__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.subtract(other, self)
	
	def __mul__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.multiply(self, other) 
	
	def __rmul__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.multiply(other, self)
	
	def __truediv__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.divide(self, other) 
	
	def __rtruediv__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.divide(other, self)
	
	def __floordiv__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.floor_divide(self, other) 
	
	def __rfloordiv__(self, other):
		if not self.has_same_axis_as(other):
			raise ValueError("Axes are not aligned for the operation.")
		return np.floor_divide(other, self)
	
	def __pow__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.power(self, other) 
	
	def __rpow__(self, other):
		if isinstance(other, type(self)):
			if not self.has_same_axis_as(other):
				raise ValueError("Axes are not aligned for the operation.")
		return np.power(other, self)
	
	#===============================
	
	
	#-------------------------------
	# Basic comparison operations
	#-------------------------------
	def __eq__(self, other):
		return np.equal(self, other)
	
	def __ne__(self, other):
		return np.not_equal(self, other)
	
	def __lt__(self, other):
		return np.less(self, other)
	
	def __le__(self, other):
		return np.less_equal(self, other)
	
	def __gt__(self, other):
		return np.greater(self, other)
	
	def __ge__(self, other):
		return np.greater_equal(self, other)
	
	#===============================
	
	#-----------------------------------
	# Info
	#-----------------------------------
	
	def print_info(self) -> None:
		"""Print key information about the spectrogram signal."""
		t = self._t
		sr = t._sr
		duration = t._values[-1]
		shape = self.shape
		title = self._title
		
		print("-"*50)
		print(f"{'Title':<20}: {title}")
		print("-"*50)
		print(f"{'Type':<20}: {self.__class__.__name__}")
		print(f"{'Shape':<20}: {shape} (freq bins × time frames)")
		print(f"{'Duration':<20}: {duration}")
		print(f"{'Frame Rate':<20}: {sr} (frames / sec)")
		print(f"{'Frame Duration':<20}: {1 / sr:.4f} sec ({(1 / sr) * 1000:.2f} ms)")
		
		# Inheritance chain
		cls_chain = " → ".join(cls.__name__ for cls in reversed(self.__class__.__mro__[:-1]))
		print(f"{'Inheritance':<20}: {cls_chain}")
		print("=" * 50)
		
		
	#===================================