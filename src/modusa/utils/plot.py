#!/usr/bin/env python3

from modusa.models.signal1D import Signal1D
from modusa.models.signal2D import Signal2D
import matplotlib.pyplot as plt

def _in_notebook() -> bool:
	"""
	To check if we are in jupyter notebook environment.
	"""
	try:
		from IPython import get_ipython
		shell = get_ipython()
		return shell and shell.__class__.__name__ == "ZMQInteractiveShell"
	except ImportError:
		return False

def plot_multiple_signals(
	*args,
	x_lim: tuple[float, float] | None = None,
	highlight_regions: list[tuple[float, float, str]] | None = None,
	vlines: list[float, ...] | None = None,
) -> plt.Figure:
	"""
	Plots multiple instances of uniform `Signal1D` and `Signal2D`
	with proper formatting and time aligned.
	
	Note
	----
	- The signals must be have uniform time axis.
	"""
	assert len(args) >= 1, "No signal provided to plot"
	
	for signal in args: 
		assert isinstance(signal, (Signal1D, Signal2D))
#		assert signal.sax[-1].is_uniform
		
	height_ratios = []
	n_signal1D = 0
	n_signal2D = 0
	for signal in args:
		if isinstance(signal, Signal1D):
			n_signal1D += 1
			height_ratios.append(0.4)
		elif isinstance(signal, Signal2D):
			n_signal2D += 1
			height_ratios.append(1)
		else:
			raise ms.excp.InputTypeError
			
	n_subplots = len(args)
	fig_width = 15
	fig_height = n_signal1D * 2 + n_signal2D * 4 # This is as per the figsize height set in the plotter tool
	fig, axs = plt.subplots(n_subplots, 2, figsize=(fig_width, fig_height), width_ratios=[1, 0.01], height_ratios=height_ratios) # 2nd column for cbar
	if n_subplots == 1:
		axs = [axs]  # axs becomes list of one pair [ (ax, cbar_ax) ]
	for i, signal in enumerate(args):
		if isinstance(signal, Signal1D):
			signal.plot(axs[i][0], x_lim=x_lim, highlight_regions=highlight_regions, show_grid=True, vlines=vlines)
			axs[i][1].remove()
		elif isinstance(signal, Signal2D):
			signal.plot(axs[i][0], x_lim=x_lim, show_colorbar=True, cax=axs[i][1], highlight_regions=highlight_regions, vlines=vlines)
		axs[i][0].sharex(axs[0][0])
		
		
	if _in_notebook():
		plt.tight_layout()
		plt.close(fig)
		return fig
	else:
		plt.tight_layout()
		plt.show()
		return fig