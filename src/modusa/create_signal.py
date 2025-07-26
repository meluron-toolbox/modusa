#!/usr/bin/env python3

import numpy as np
from modusa.signals.time_domain_signal import TimeDomainSignal

def create_signal(y: list | np.ndarray, sr: float = 1.0, t0: float = 0.0, title: str = "Signal") -> TimeDomainSignal:
	"""
	Creates a time domain signal.
	"""
	assert isinstance(y, (list, np.ndarray, float, int))
	assert isinstance(sr, (int, float))
	assert isinstance(t0, float)
	
	# Convert y to a np.ndarray
	y = np.asarray(y)
	sr = float(sr)
	title = str(title)
	
	return TimeDomainSignal(y=y, sr=sr, t0=t0, title=title)

def create_zeros(size: int | tuple[int, ...], sr: float = 1.0, t0: float = 0.0) -> TimeDomainSignal:
	"""
	Creates a signal with all zeros
	"""
	assert isinstance(size, (int, tuple))
	assert isinstance(sr, (int, float))
	
	y = np.zeros(size)
	sr = float(sr)
	return TimeDomainSignal(y=y, sr=sr, t0=t0, title="Zeros")

def create_zeros_like(signal: TimeDomainSignal) -> TimeDomainSignal:
	"""
	Creates a signal with all zeros with same
	configurations as the input.
	"""
	assert isinstance(signal, TimeDomainSignal)
	y = np.zeros_like(signal.y)
	
	return TimeDomainSignal(y=y, sr=signal.sr, t0=signal.t0, title="Zeros")
	
	
def create_ones(size: int | tuple[int, ...], sr: float = 1.0, t0: float = 0.0) -> TimeDomainSignal:
	"""
	Creates a signal with all zeros
	"""
	assert isinstance(size, (int, tuple))
	assert isinstance(sr, (int, float))
	
	y = np.ones(size)
	sr = float(sr)
	return TimeDomainSignal(y=y, sr=sr, t0=t0, title="Ones")
	
def create_ones_like(signal: TimeDomainSignal) -> TimeDomainSignal:
	"""
	Creates a signal with all zeros with same
	configurations as the input.
	"""
	assert isinstance(signal, TimeDomainSignal)
	y = np.ones_like(signal.y)
	
	return TimeDomainSignal(y=y, sr=signal.sr, t0=signal.t0, title="Ones")

	
signal = create_signal
zeros = create_zeros
zeros_like = create_zeros_like
ones = create_ones
ones_like = create_ones_like
	