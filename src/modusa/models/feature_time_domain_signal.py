#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from .signal2D import Signal2D
from .s_ax import SAx
from modusa.tools.math_ops import MathOps
from typing import Self, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
import modusa as ms

class FeatureTimeDomainSignal(Signal2D):
	"""
	Space to represent feature time domain signal (2D).

	Note
	----
	- 
	"""
	
	#--------Meta Information----------
	_name = "Feature Time Domain Signal"
	_nickname = "FTDS" # This is to be used in repr/str methods
	_description = "Space to represent feature time domain signal (2D)."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-21"
	#----------------------------------
	
	@validate_args_type()
	def __init__(
		self,
		data: np.ndarray,
		feature: np.ndarray,
		frame_rate: float,
		t0: float,
		data_label: str,
		feature_label: str,
		time_label: str,
		title: str
	):
		
		feature_sax = SAx(values=feature, label=feature_label)
		time_sax= ms.sax.linear(n_points=data.shape[1], sr=frame_rate, start=t0, label=time_label)
		sax = (feature_sax, time_sax)
	
		super().__init__(data=data, data_label=data_label, sax=sax, title=title) # Instantiating `Signal2D` class
	
	#--------------------------------------
	# Properties (Hidden)
	#--------------------------------------
	
	@immutable_property("Use .set_meta_info method.")
	def _f(self) -> SAx:
		"""Data array."""
		return self._y
	
	@immutable_property("Use .set_meta_info method.")
	def _t(self) -> SAx:
		"""Data array."""
		return self._x
	
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
		
	def set_meta_info(self, data_label: str | None = None, feature_label: str | None = None, time_label: str | None = None, title: str | None = None) -> None:
		"""
		Set meta info about the signal.
		"""
		M = self._M
		f = self._f._values
		sr = self._sr
		t0 = self._t0
		
		data_label = str(data_label) if data_label is not None else self.M_label
		feature_label = str(feature_label) if feature_label is not None else self._f._label
		time_label = str(time_label) if time_label is not None else self._t._label
		title = str(title) if title is not None else self._title
		
		return self.__class__(data=M, feature=f, frame_rate=self._frame_rate, t0=self._t0, data_label=data_label, feature_label=feature_label, time_label=time_label, title=title)
	
	#===================================
	
	
	
	
	
	#-----------------------------------
	# Tools
	#-----------------------------------
	

	#===================================
	
	
	#-----------------------------------
	# Info
	#-----------------------------------
	
	def print_info(self) -> None:
		"""Print key information about the spectrogram signal."""
		
		print("-"*50)
		print(f"{'Title':<20}: {self.title}")
		print("-"*50)
		print(f"{'Type':<20}: {self.__class__.__name__}")
		print(f"{'Shape':<20}: {self.shape} (freq bins × time frames)")
		print(f"{'Duration':<20}: {self.time_axis.duration}")
		print(f"{'Frame Rate':<20}: {self.time_axis.sr} (frames / sec)")
		print(f"{'Frame Duration':<20}: {1 / self.time_axis.sr:.4f} sec ({(1 / self.time_axis.sr) * 1000:.2f} ms)")
		
		# Inheritance chain
		cls_chain = " → ".join(cls.__name__ for cls in reversed(self.__class__.__mro__[:-1]))
		print(f"{'Inheritance':<20}: {cls_chain}")
		print("=" * 50)
		
		
	#===================================