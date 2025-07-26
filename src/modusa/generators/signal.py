#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import validate_args_type
from modusa.generators.base import ModusaGenerator

from modusa.models.s_ax import SAx

from modusa.models.signal1D import Signal1D
from modusa.models.signal2D import Signal2D

from modusa.models.time_domain_signal import TimeDomainSignal
from modusa.models.feature_time_domain_signal import FeatureTimeDomainSignal

from modusa.models.audio_signal import AudioSignal

import numpy as np



class SignalGenerator(ModusaGenerator):
	"""
	Provides APIs to generate instances of different `ModusaSignal`
	subclasses directly from user friendly data structures
	(like np.ndarray, list, int, float, np.generic).
	"""
	
	#--------Meta Information----------
	_name = "SignalGenerator"
	_description = "Interface to generate different signals"
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-25"
	#----------------------------------
	
	def __init__(self):
		super().__init__()
		self.s1d = self.S1D()
		self.tds = self.TDS()
		self.s2d = self.S2D()
		self.ftds = self.FTDS()
	
	#-------------------------------
	# 1D Signals Below
	#-------------------------------
	
	class S1D:
		"""
		Provides user friendly APIs to generate instances of different `Signal1D`
		instances.
		"""
		@staticmethod
		def from_array(
			y: np.ndarray | list | float | int | np.generic,
			x: np.ndarray | list | float | int | np.generic | None = None,
			y_label: str = "Y",
			x_label: str = "X",
			title: str = "1D Signal"
		) -> Signal1D:
			"""
			Create `Signal1D` instance from basic data structures.
			"""
			assert isinstance(y, (np.ndarray, list, float, int, np.generic))
			assert isinstance(x, (np.ndarray, list, float, int, np.generic)) or x is None
			assert isinstance(y_label, str) and isinstance(x_label, str) and isinstance(title, str)
			
			if isinstance(y, (float, int, np.generic)): y = [y] # Convert to list of 1 element
			if isinstance(x, (float, int, np.generic)): x = [x] # Convert to list of 1 element
			
			y = np.asarray(y)
			if x is None: x = np.arange(y.shape[0])
			else: x = np.asarray(x)
			
			assert y.ndim ==1 and x.ndim == 1, "Signal1D must have only one dimension"
			assert y.shape == x.shape, "Shape mismatch"
			
			sax = SAx(values=x, label=x_label) # Creating a signal axis instance
			sax = (sax, )
			
			return Signal1D(data=y, sax=sax, data_label=y_label, title=title)
		
		@classmethod
		def zeros(cls, shape: int | tuple[int, int]) -> Signal1D:
			"""
			Create `Signal1D` instance with all zeros.
			"""
			assert isinstance(shape, (int, tuple))
			y = np.zeros(shape)
			
			return cls.from_array(y=y, title="Zeros")
		
		@classmethod
		def zeros_like(cls, signal: Signal1D) -> Signal1D:
			"""
			Create `Signal1D` instance similar to `signal`
			but with all entries being zeros.
			"""
			
			assert signal.__class__ in [Signal1D]
			
			y = np.zeros(signal.shape)
			y_label = signal._y_label
			x = signal._x
			x_label = signal._x_label
			title = signal._title
			
			return cls.from_array(y=y, x=x, y_label=y_label, x_label=x_label, title=title)
		
		
		@classmethod
		def ones(cls, shape: int | tuple[int, int]) -> Signal1D:
			"""
			Create `Signal1D` instance with all ones.
			"""
			assert isinstance(shape, (int, tuple))
			y = np.ones(shape)
			
			return cls.from_array(y=y, title="Ones")
		
		@classmethod
		def ones_like(cls, signal: Signal1D) -> Signal1D:
			"""
			Create `Signal1D` instance similar to `signal`
			but with all entries being ones.
			"""
			
			assert signal.__class__ in [Signal1D]
			
			y = np.ones(signal.shape)
			y_label = signal._y_label
			x = signal._x
			x_label = signal._x_label
			title = signal._title
			
			return cls.from_array(y=y, x=x, y_label=y_label, x_label=x_label, title=title)
		
		@classmethod
		def random(cls, shape: int | tuple[int, int]) -> Signal1D:
			"""
			Create `Signal1D` instance with random entries.
			"""
			assert isinstance(shape, (int, tuple))
			y = np.random.random(shape)
			
			return cls.from_array(y=y, title="Random")
		
		@classmethod
		def random_like(cls, signal: Signal1D) -> Signal1D:
			"""
			Create `Signal1D` instance similar to `signal`
			but with all entries being ones.
			"""
			
			assert signal.__class__ in [Signal1D]
			
			y = np.random.random(signal.shape)
			y_label = signal._y_label
			x = signal._x
			x_label = signal._x_label
			title = signal._title
			
			return cls.from_array(y=y, x=x, y_label=y_label, x_label=x_label, title=title)
		
	class TDS:
		"""
		Provides user friendly APIs to generate instances of different 
		`TimeDomainSignal` instances.
		"""
		
		@staticmethod
		def from_array(
			y: np.ndarray | list | float | int | np.generic,
			sr: float | int = 1.0,
			t0: float | int = 0.0,
			y_label: str = "Y",
			t_label: str = "Time (sec)",
			title: str = "Time Domain Signal"
		) -> TimeDomainSignal:
			"""
			Create `TimeDomainSignal` instance from basic data structures.
			"""
			assert isinstance(y, (np.ndarray, list, float, int, np.generic))
			assert isinstance(sr, (int, float)) and isinstance(t0, (int, float))
			assert isinstance(y_label, str) and isinstance(t_label, str) and isinstance(title, str)
			
			if isinstance(y, (float, int, np.generic)): y = [y] # Convert to list of 1 element
			y = np.asarray(y)
			
			sr = float(sr)
			t0 = float(t0)
			
			assert y.ndim == 1
			
			return TimeDomainSignal(data=y, sr=sr, t0=t0, data_label=y_label, time_label=t_label, title=title)
		
		@classmethod
		def zeros(cls, shape: int | tuple[int, int]) -> TimeDomainSignal:
			"""
			Create `TimeDomainSignal` instance with all zeros.
			"""
			assert isinstance(shape, (int, tuple))
			y = np.zeros(shape)
			
			return cls.from_array(y=y, title="Zeros")
		
		@classmethod
		def zeros_like(cls, signal: TimeDomainSignal) -> TimeDomainSignal:
			"""
			Create `TimeDomainSignal` instance similar to `signal`
			but with all entries being zeros.
			"""
			
			assert signal.__class__ in [TimeDomainSignal]
			
			y = np.zeros(signal.shape)
			y_label = signal._y_label
			sr = signal._sr
			t0 = signal._t0
			t_label = signal._t_label
			title = signal._title
			
			return cls.from_array(y=y, sr=sr, t0=t0, y_label=y_label, t_label=t_label, title=title)
		
		
		@classmethod
		def ones(cls, shape: int | tuple[int, int]) -> Signal1D:
			"""
			Create `Signal1D` instance with all ones.
			"""
			assert isinstance(shape, (int, tuple))
			y = np.ones(shape)
			
			return cls.from_array(y=y, title="Ones")
		
		@classmethod
		def ones_like(cls, signal: Signal1D) -> Signal1D:
			"""
			Create `Signal1D` instance similar to `signal`
			but with all entries being ones.
			"""
			
			assert signal.__class__ in [TimeDomainSignal]
			
			y = np.ones(signal.shape)
			y_label = signal._y_label
			sr = signal._sr
			t0 = signal._t0
			t_label = signal._t_label
			title = signal._title
			
			return cls.from_array(y=y, sr=sr, t0=t0, y_label=y_label, t_label=t_label, title=title)
		
		@classmethod
		def random(cls, shape: int | tuple[int, int]) -> Signal1D:
			"""
			Create `Signal1D` instance with random entries.
			"""
			assert isinstance(shape, (int, tuple))
			y = np.random.random(shape)
			
			return cls.from_array(y=y, title="Random")
		
		@classmethod
		def random_like(cls, signal: Signal1D) -> Signal1D:
			"""
			Create `Signal1D` instance similar to `signal`
			but with all entries being ones.
			"""
			
			assert signal.__class__ in [TimeDomainSignal]
			
			y = np.random.random(signal.shape)
			y_label = signal._y_label
			sr = signal._sr
			t0 = signal._t0
			t_label = signal._t_label
			title = signal._title
			
			return cls.from_array(y=y, sr=sr, t0=t0, y_label=y_label, t_label=t_label, title=title)
	
	#===============================
	
	#-------------------------------
	# 2D Signals Below
	#-------------------------------
	
	class S2D:
		"""
		Provides user friendly APIs to generate instances of different `Signal2D`
		instances.
		"""
		
		@staticmethod
		def from_array(
			M: np.ndarray | list | float | int | np.generic,
			y: np.ndarray | list | float | int | np.generic | None = None,
			x: np.ndarray | list | float | int | np.generic | None = None,
			M_label: str = "M",
			y_label: str = "Y",
			x_label: str = "X",
			title: str = "2D Signal"
		) -> Signal2D:
			"""
			Create `Signal2D` instance from basic data structures.
			"""
			assert isinstance(M, (np.ndarray, list, float, int, np.generic))
			assert isinstance(x, (np.ndarray, list, float, int, np.generic)) or x is None
			assert isinstance(y, (np.ndarray, list, float, int, np.generic)) or y is None
			assert isinstance(M_label, str) and isinstance(y_label, str) and isinstance(x_label, str) and isinstance(title, str)
			
			if isinstance(M, (float, int, np.generic)): M = [[M]] # Convert to list of 1 element
			if isinstance(y, (float, int, np.generic)): y = [y] # Convert to list of 1 element
			if isinstance(x, (float, int, np.generic)): x = [x] # Convert to list of 1 element
			
			M = np.asarray(M)
			assert M.ndim == 2
			
			if y is None: y = np.arange(M.shape[0])
			else: y = np.asarray(y)
			assert y.ndim == 1
			
			if x is None: x = np.arange(M.shape[1])
			else: x = np.asarray(x)
			assert x.ndim == 1
			
			assert y.shape[0] == M.shape[0], "Shape mismatch"
			assert x.shape[0] == M.shape[1], "Shape mismatch"
			
			y_sax = SAx(values=y, label=y_label) # Creating a signal axis instance
			x_sax = SAx(values=x, label=x_label) # Creating a signal axis instance
			sax = (y_sax, x_sax)
			
			return Signal2D(data=M, sax=sax, data_label=M_label, title=title)
		
		@classmethod
		def zeros(cls, shape: tuple[int, int]) -> Signal2D:
			"""
			Create `Signal2D` instance with all zeros.
			"""
			assert isinstance(shape, tuple)
			M = np.zeros(shape)
			
			return cls.from_array(M=M, title="Zeros")
		
		@classmethod
		def zeros_like(cls, signal: Signal2D) -> Signal2D:
			"""
			Create `Signal2D` instance similar to `signal`
			but with all entries being zeros.
			"""
			
			assert signal.__class__ in [Signal2D]
			
			M = np.zeros(signal.shape)
			y = signal._y
			x = signal._x
			
			M_label = signal._M_label
			y_label = signal._y_label
			x_label = signal._x_label
			title = signal._title
			
			return cls.from_array(M=M, y=y, x=x, M_label=M_label, y_label=y_label, x_label=x_label, title=title)
		
		
		@classmethod
		def ones(cls, shape: tuple[int, int]) -> Signal1D:
			"""
			Create `Signal2D` instance with all ones.
			"""
			assert isinstance(shape, tuple)
			M = np.ones(shape)
			
			return cls.from_array(M=M, title="Ones")
		
		@classmethod
		def ones_like(cls, signal: Signal2D) -> Signal2D:
			"""
			Create `Signal2D` instance similar to `signal`
			but with all entries being ones.
			"""
			
			assert signal.__class__ in [Signal2D]
			
			M = np.ones(signal.shape)
			y = signal._y
			x = signal._x
			
			M_label = signal._M_label
			y_label = signal._y_label
			x_label = signal._x_label
			title = signal._title
			
			return cls.from_array(M=M, y=y, x=x, M_label=M_label, y_label=y_label, x_label=x_label, title=title)
		
		@classmethod
		def random(cls, shape: tuple[int, int]) -> Signal2D:
			"""
			Create `Signal2D` instance with random entries.
			"""
			assert isinstance(shape, tuple)
			M = np.random.random(shape)
			
			return cls.from_array(M=M, title="Random")
		
		@classmethod
		def random_like(cls, signal: Signal2D) -> Signal2D:
			"""
			Create `Signal1D` instance similar to `signal`
			but with all entries being ones.
			"""
			
			assert signal.__class__ in [Signal2D]
			
			M = np.random.random(signal.shape)
			y = signal._y
			x = signal._x
			
			M_label = signal._M_label
			y_label = signal._y_label
			x_label = signal._x_label
			title = signal._title
			
			return cls.from_array(M=M, y=y, x=x, M_label=M_label, y_label=y_label, x_label=x_label, title=title)
		
		
	
	class FTDS:
		"""
		Provides user friendly APIs to generate instances of different 
		`FeatureTimeDomainSignal` instances.
		"""
		
		@staticmethod
		def from_array(
			M: np.ndarray | list | float | int | np.generic,
			f: np.ndarray | list | float | int | np.generic | None = None,
			sr: int | float = 1.0,
			t0: int | float = 0.0,
			M_label: str = "M",
			f_label: str = "Feature",
			t_label: str = "Time (sec)",
			title: str = "Feature Time Domain Signal"
		) -> FeatureTimeDomainSignal:
			"""
			Create `Signal2D` instance from basic data structures.
			"""
			assert isinstance(M, (np.ndarray, list, float, int, np.generic))
			assert isinstance(f, (np.ndarray, list, float, int, np.generic)) or f is None
			assert isinstance(sr, (int, float)) and isinstance(t0, (int, float))
			assert isinstance(M_label, str) and isinstance(f_label, str) and isinstance(t_label, str) and isinstance(title, str)
			
			if isinstance(M, (float, int, np.generic)): M = [[M]] # Convert to list of 1 element
			if isinstance(f, (float, int, np.generic)): f = [f] # Convert to list of 1 element
			
			M = np.asarray(M)
			assert M.ndim == 2
			
			if f is None: f = np.arange(M.shape[0])
			else: f = np.asarray(f)
			assert f.ndim == 1
			assert f.shape[0] == M.shape[0], "Shape mismatch"
			
			sr = float(sr)
			t0 = float(t0)

			return FeatureTimeDomainSignal(data=M, feature=f, frame_rate=sr, t0=t0, data_label=M_label, feature_label=f_label, time_label=t_label, title=title)
		
		@classmethod
		def zeros(cls, shape: tuple[int, int]) -> FeatureTimeDomainSignal:
			"""
			Create `FeatureTimeDomainSignal` instance with all zeros.
			"""
			assert isinstance(shape, tuple)
			M = np.zeros(shape)
			
			return cls.from_array(M=M, title="Zeros")
		
		@classmethod
		def zeros_like(cls, signal: FeatureTimeDomainSignal) -> FeatureTimeDomainSignal:
			"""
			Create `FeatureTimeDomainSignal` instance similar to `signal`
			but with all entries being zeros.
			"""
			
			assert signal.__class__ in [FeatureTimeDomainSignal]
			
			M = np.zeros(signal.shape)
			f = signal._f
			t = signal._t
			
			M_label = signal._M_label
			f_label = signal._f_label
			t_label = signal._t_label
			title = signal._title
			
			return cls.from_array(M=M, f=f, t=t, M_label=M_label, f_label=f_label, t_label=t_label, title=title)
		
		
		@classmethod
		def ones(cls, shape: tuple[int, int]) -> FeatureTimeDomainSignal:
			"""
			Create `FeatureTimeDomainSignal` instance with all ones.
			"""
			assert isinstance(shape, tuple)
			M = np.ones(shape)
			
			return cls.from_array(M=M, title="Ones")
		
		@classmethod
		def ones_like(cls, signal: FeatureTimeDomainSignal) -> FeatureTimeDomainSignal:
			"""
			Create `FeatureTimeDomainSignal` instance similar to `signal`
			but with all entries being ones.
			"""
			
			assert signal.__class__ in [FeatureTimeDomainSignal]
			
			M = np.ones(signal.shape)
			f = signal._f
			t = signal._t
			
			M_label = signal._M_label
			f_label = signal._f_label
			t_label = signal._t_label
			title = signal._title
			
			return cls.from_array(M=M, f=f, t=t, M_label=M_label, f_label=f_label, t_label=t_label, title=title)
		
		@classmethod
		def random(cls, shape: tuple[int, int]) -> FeatureTimeDomainSignal:
			"""
			Create `FeatureTimeDomainSignal` instance with random entries.
			"""
			assert isinstance(shape, tuple)
			M = np.random.random(shape)
			
			return cls.from_array(M=M, title="Random")
		
		@classmethod
		def random_like(cls, signal: FeatureTimeDomainSignal) -> FeatureTimeDomainSignal:
			"""
			Create `FeatureTimeDomainSignal` instance similar to `signal`
			but with random entries.
			"""
			
			assert signal.__class__ in [FeatureTimeDomainSignal]
			
			M = np.random.random(signal.shape)
			f = signal._f
			t = signal._t
			
			M_label = signal._M_label
			f_label = signal._f_label
			t_label = signal._t_label
			title = signal._title
			
			return cls.from_array(M=M, f=f, t=t, M_label=M_label, f_label=f_label, t_label=t_label, title=title)
		
		
		
	class audio:
		"""
		Provides user friendly APIs to generate instances of different 
		`AudioSignal` instances.
		"""
		pass

signal = SignalGenerator() # sigleton object to be used