#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#======== 1D ===========
def plot1d(*args, ann=None, events=None, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, legend=None, get_ax=False):
		"""
		Plots a 1D signal using matplotlib.

		.. code-block:: python
	
			import modusa as ms
			import numpy as np
			
			x = np.arange(100) / 100
			y = np.sin(x)
			
			display(ms.plot1d(y, x))
			
	
		Parameters
		----------
		*args : tuple[array-like, array-like]
			- The signal values to be plotted.
			- E.g. (y1, x1), (y2, x2), ...
		ann : list[tuple[Number, Number, str] | None
			- A list of annotations to mark specific points. Each tuple should be of the form (start, end, label).
			- Default: None => No annotation.
		events : list[Number] | None
			- A list of x-values where vertical lines (event markers) will be drawn.
			- Default: None
		xlim : tuple[Number, Number] | None
			- Limits for the x-axis as (xmin, xmax).
			- Default: None
		ylim : tuple[Number, Number] | None
			- Limits for the y-axis as (ymin, ymax).
			- Default: None
		xlabel : str | None
			- Label for the x-axis.
			- - Default: None
		ylabel : str | None
			- Label for the y-axis.
			- Default: None
		title : str | None
			- Title of the plot.
			- Default: None
		legend : list[str] | None
			- List of legend labels corresponding to each signal if plotting multiple lines.
			- Default: None
	
		Returns
		-------
		plt.Figure
			Matplolib figure.
		"""
	
		for arg in args:
			if len(arg) != 2:
				raise ValueError(f"Data to plot needs to have 2 arrays (y, x)")
		if isinstance(legend, str): legend = (legend, )
		
		if legend is not None:
			if len(legend) < len(args):
				raise ValueError(f"Legend should be provided for each signal.")

		fig = plt.figure(figsize=(16, 2))
		gs = gridspec.GridSpec(2, 1, height_ratios=[0.2, 1])
			
		colors = plt.get_cmap('tab10').colors
		
		signal_ax = fig.add_subplot(gs[1, 0])
		annotation_ax = fig.add_subplot(gs[0, 0], sharex=signal_ax)
		
		# Set lim
		if xlim is not None:
			signal_ax.set_xlim(xlim)
		
		if ylim is not None:
			signal_ax.set_ylim(ylim)
		
			
		# Add signal plot
		for signal in args:
			y, x = signal
			signal_ax.plot(x, y, label=legend)
		
		# Add annotations
		if ann is not None:
			annotation_ax.set_ylim(0, 1)
			for i, (start, end, tag) in enumerate(ann):
				if xlim is not None:
					if end < xlim[0] or start > xlim[1]:
						continue  # Skip out-of-view regions
					# Clip boundaries to xlim
					start = max(start, xlim[0])
					end = min(end, xlim[1])
					
				color = colors[i % len(colors)]
				width = end - start
				rect = Rectangle((start, 0), width, 1, color=color, alpha=0.7)
				annotation_ax.add_patch(rect)
				annotation_ax.text((start + end) / 2, 0.5, tag,
									ha='center', va='center',
									fontsize=10, color='white', fontweight='bold', zorder=10)
		# Add vlines
		if events is not None:
			for xpos in events:
				if xlim is not None:
					if xlim[0] <= xpos <= xlim[1]:
						annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
				else:
					annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
					
		# Add legend
		if legend is not None:
			handles, labels = signal_ax.get_legend_handles_labels()
			fig.legend(handles, legend,
								loc='upper right',
								bbox_to_anchor=(0.9, 1.2),
								ncol=len(legend),
								frameon=False)
			
		# Set title, labels
		if title is not None:
			annotation_ax.set_title(title, pad=10, size=11)
		if xlabel is not None:
			signal_ax.set_xlabel(xlabel)
		if ylabel is not None:
			signal_ax.set_ylabel(ylabel)
		
		# Decorating annotation axis thicker
		if ann is not None:
			annotation_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
		else:
			annotation_ax.axis("off")
		
		
		fig.subplots_adjust(hspace=0.01, wspace=0.05)
		plt.close()
		if get_ax is True:
			return fig, annotation_ax, signal_ax
		else:
			return fig

#======== 2D ===========
def plot2d(*args, ann=None, events=None, xlim=None, ylim=None, origin="lower", Mlabel=None, xlabel=None, ylabel=None, title=None, legend=None):
	"""
	Plots a 2D matrix (e.g., spectrogram or heatmap) with optional annotations and events.

	.. code-block:: python

		import modusa as ms
		import numpy as np
		
		M = np.random.random((10, 30))
		y = np.arange(M.shape[0])
		x = np.arange(M.shape[1])
		
		display(ms.plot2d(M, y, x))

	Parameters
	----------
	*args : tuple[array-like, array-like]
		- The signal values to be plotted.
		- E.g. (M1, y1, x1), (M2, y2, x2), ...
	ann : list[tuple[Number, Number, str]] | None
		- A list of annotation spans. Each tuple should be (start, end, label).
		- Default: None (no annotations).
	events : list[Number] | None
		- X-values where vertical event lines will be drawn.
		- Default: None.
	xlim : tuple[Number, Number] | None
		- Limits for the x-axis as (xmin, xmax).
		- Default: None (auto-scaled).
	ylim : tuple[Number, Number] | None
		- Limits for the y-axis as (ymin, ymax).
		- Default: None (auto-scaled).
	origin : {'upper', 'lower'}
		- Origin position for the image display. Used in `imshow`.
		- Default: "lower".
	Mlabel : str | None
		- Label for the colorbar (e.g., "Magnitude", "Energy").
		- Default: None.
	xlabel : str | None
		- Label for the x-axis.
		- Default: None.
	ylabel : str | None
		- Label for the y-axis.
		- Default: None.
	title : str | None
		- Title of the plot.
		- Default: None.
	legend : list[str] | None
		- Legend labels for any overlaid lines or annotations.
		- Default: None.

	Returns
	-------
	matplotlib.figure.Figure
		The matplotlib Figure object.
	"""
	
	for arg in args:
		if len(arg) != 3:
			raise ValueError(f"Data to plot needs to have 3 arrays (M, y, x)")

	fig = plt.figure(figsize=(16, 4))
	gs = gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 1]) # colorbar, annotation, signal

	colors = plt.get_cmap('tab10').colors
	
	signal_ax = fig.add_subplot(gs[2, 0])
	annotation_ax = fig.add_subplot(gs[1, 0], sharex=signal_ax)
	
	colorbar_ax = fig.add_subplot(gs[0, 0])
	colorbar_ax.axis("off")
	
	
	# Add lim
	if xlim is not None:
		signal_ax.set_xlim(xlim)
		
	if ylim is not None:
		signal_ax.set_ylim(ylim)
		
	# Add signal plot
	for signal in args:
		M, y, x = signal
		dx = x[1] - x[0]
		dy = y[1] - y[0]
		extent=[x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]
		im = signal_ax.imshow(M, aspect="auto", origin=origin, extent=extent)
	
	# Add annotations
	if ann is not None:
		annotation_ax.set_ylim(0, 1)
		for i, (start, end, tag) in enumerate(ann):
			if xlim is not None:
				if end < xlim[0] or start > xlim[1]:
					continue  # Skip out-of-view regions
				# Clip boundaries to xlim
				start = max(start, xlim[0])
				end = min(end, xlim[1])
				
			color = colors[i % len(colors)]
			width = end - start
			rect = Rectangle((start, 0), width, 1, color=color, alpha=0.7)
			annotation_ax.add_patch(rect)
			annotation_ax.text((start + end) / 2, 0.5, tag,
								ha='center', va='center',
								fontsize=10, color='white', fontweight='bold', zorder=10)
	# Add vlines
	if events is not None:
		for xpos in events:
			if xlim is not None:
				if xlim[0] <= xpos <= xlim[1]:
					annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
			else:
				annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
				
	# Add colorbar
	# Create an inset axis on top-right of signal_ax
	cax = inset_axes(
		colorbar_ax,
		width="20%",      # percentage of parent width
		height="20%",      # height in percentage of parent height
		loc='upper right',
		bbox_to_anchor=(0, 0, 1, 1),
		bbox_transform=colorbar_ax.transAxes,
		borderpad=1
	)
	
	cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
	cbar.ax.xaxis.set_ticks_position('top')
	
	if Mlabel is not None:
		cbar.set_label(Mlabel, labelpad=5)
	
		
	# Set title, labels
	if title is not None:
		annotation_ax.set_title(title, pad=10, size=11)
	if xlabel is not None:
		signal_ax.set_xlabel(xlabel)
	if ylabel is not None:
		signal_ax.set_ylabel(ylabel)
	
	
	# Making annotation axis spines thicker
	if ann is not None:
		annotation_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
	else:
		annotation_ax.axis("off")

	fig.subplots_adjust(hspace=0.01, wspace=0.05)
	plt.close()
	return fig