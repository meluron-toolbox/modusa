#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

#======== 1D ===========
def plot1d(y, x, ann=None, events=None, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, legend=None):
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
		y : array-like
			- The signal values to be plotted.
		x : array-like
			- The corresponding x-axis values (e.g., time or sample index).
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
			List of legend labels corresponding to each signal if plotting multiple lines.
	
		Returns
		-------
		plt.Figure
			Matplolib figure.
		"""
		fig = plt.figure(figsize=(16, 3))
		gs = gridspec.GridSpec(2, 3, height_ratios=[0.2, 1], width_ratios=[1, 0.1, 0.01])
		colors = plt.get_cmap('tab10').colors
		
		signal_ax = fig.add_subplot(gs[1, 0])
		annotation_ax = fig.add_subplot(gs[0, 0], sharex=signal_ax)
		
		legend_ax = fig.add_subplot(gs[1, 1])
		colorbar_ax = fig.add_subplot(gs[1, 2])
		
		# Making annotation axis spines thicker
		for spine in annotation_ax.spines.values():
			spine.set_linewidth(2)
		
		# Add xlim
		if xlim is not None:
			x_start, x_end = xlim
			signal_ax.set_xlim([x_start, x_end])
			
		# Add signal plot
		signal_ax.plot(x, y, label=legend)
		
		# Add annotations
		if ann is not None:
			for i, (start, end, tag) in enumerate(ann):
				if xlim is not None:
					if end < x_start or start > x_end:
						continue  # Skip out-of-view regions
					# Clip boundaries to xlim
					start = max(start, x_start)
					end = min(end, x_end)
					
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
					if x_start <= xpos <= x_end:
						annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
				else:
					annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
					
		# Add legend
		if legend is not None:
			handles, labels = signal_ax.get_legend_handles_labels()
			legend_ax.legend(handles, labels, loc='upper left')
			
		# Set title, labels
		if title is not None:
			annotation_ax.set_title(title, pad=10)
		if xlabel is not None:
			signal_ax.set_xlabel(xlabel)
		if ylabel is not None:
			signal_ax.set_ylabel(ylabel)
			
		annotation_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
		legend_ax.axis("off")
		colorbar_ax.axis("off")
		plt.tight_layout()
		plt.close()
		return fig

#======== 2D ===========
def plot2d(M, y, x, ann=None, events=None, xlim=None, ylim=None, origin="lower", Mlabel=None, xlabel=None, ylabel=None, title=None, legend=None):
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
	M : 2D array-like
		- The 2D data matrix to visualize (e.g., magnitude or energy over time and frequency).
	y : array-like
		- Y-axis values (e.g., frequency bins or channels).
	x : array-like
		- X-axis values (e.g., time or sample index).
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
	fig = plt.figure(figsize=(16, 5))
	gs = gridspec.GridSpec(2, 3, height_ratios=[0.1, 1], width_ratios=[1, 0.1, 0.01])
	colors = plt.get_cmap('tab10').colors
	
	signal_ax = fig.add_subplot(gs[1, 0])
	annotation_ax = fig.add_subplot(gs[0, 0], sharex=signal_ax)
	
	legend_ax = fig.add_subplot(gs[1, 1])
	colorbar_ax = fig.add_subplot(gs[1, 2])
	
	# Making annotation axis spines thicker
	for spine in annotation_ax.spines.values():
		spine.set_linewidth(2)
	
	# Add xlim
	if xlim is not None:
		x_start, x_end = xlim
		signal_ax.set_xlim([x_start, x_end])
		
	# Add signal plot
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	extent=[x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]
	im = signal_ax.imshow(M, aspect="auto", origin=origin, extent=extent)
	
	# Add annotations
	if ann is not None:
		for i, (start, end, tag) in enumerate(ann):
			if xlim is not None:
				if end < x_start or start > x_end:
					continue  # Skip out-of-view regions
				# Clip boundaries to xlim
				start = max(start, x_start)
				end = min(end, x_end)
				
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
				if x_start <= xpos <= x_end:
					annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
			else:
				annotation_ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1.5)
				
	# Add colorbar
	cbar = plt.colorbar(im, cax=colorbar_ax)
	if Mlabel is not None:
		cbar.set_label(Mlabel, labelpad=10)
		
	# Set title, labels
	if title is not None:
		annotation_ax.set_title(title, pad=10)
	if xlabel is not None:
		signal_ax.set_xlabel(xlabel)
	if ylabel is not None:
		signal_ax.set_ylabel(ylabel)
		
	annotation_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
	legend_ax.axis("off")
	plt.tight_layout()
	plt.close()
	return fig