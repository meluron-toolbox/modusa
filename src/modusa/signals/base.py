#!/usr/bin/env python3

from modusa import excp
from modusa.decorators import immutable_property, validate_args_type
from modusa.signals.signal_ops import SignalOps
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self
import numpy as np
import matplotlib.pyplot as plt

class ModusaSignal(ABC):
	"""
	Base class for any signal in the modusa framework.
	
	Note
	----
	- Intended to be subclassed.
	"""
	
	#--------Meta Information----------
	_name = "Modusa Signal"
	_description = "Base class for any signal types in the Modusa framework."
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-06-23"
	#----------------------------------
	
	@validate_args_type()
	def __init__(self):
		self._plugin_chain = []
	
	#----------------------------
	# Properties
	#----------------------------

	def __str__(self):
		cls = self.__class__.__name__
		data = self._data
		
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
		data = self._data
		
		arr_str = np.array2string(
			data,
			separator=", ",
			threshold=50,       # limit number of elements shown
			edgeitems=3,          # show first/last 3 rows and columns
			max_line_width=120,   # avoid wrapping
			formatter={'float_kind': lambda x: f"{x:.4g}"}
		)
		
		return f"Signal({arr_str}, shape={data.shape}, kind={cls})"
	

	