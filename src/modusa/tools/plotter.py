#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import validate_args_type
from modusa.tools.base import ModusaTool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import warnings

warnings.filterwarnings("ignore", message="Glyph .* missing from font.*") # To supress any font related warnings, TODO: Add support to Devnagri font


class Plotter(ModusaTool):
	"""
	Plots different kind of signals using `matplotlib`.
	
	Note
	----
	- The class has `plot_` methods to plot different types of signals (1D, 2D).

	"""
	
	#--------Meta Information----------
	_name = ""
	_description = ""
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-06"
	#----------------------------------
	
	@staticmethod
	@validate_args_type()
	def plot_signal(
		y: np.ndarray,
		x: np.ndarray | None,
		ax: plt.Axes | None = None,
		fmt: str = "k",
		title: str | None = None,
		ylabel: str | None = None,
		xlabel: str | None = None,
		ylim: tuple[float, float] | None = None,
		xlim: tuple[float, float] | None = None,
		highlight_regions: list[tuple[float, float, str | None], ...] | None = None,
		vlines: list[float] | None = None,
		hlines: list[float] | None = None,
		show_grid: bool = False,
		show_stem: bool = False,
		legend: tuple[str, str] | str | None = None,
	) -> plt.Figure | None:
		"""
		Plots 1D signal using `matplotlib` with various settings passed through the
		arguments.

		.. code-block:: python
			
			from modusa.io import Plotter
			import numpy as np
			
			# Generate a sample sine wave
			x = np.linspace(0, 2 * np.pi, 100)
			y = np.sin(x)
			
			# Plot the signal
			fig = Plotter.plot_signal(
				y=y,
				x=x,
				ax=None,
				color="blue",
				marker=None,
				linestyle="-",
				stem=False,
				labels=("Time", "Amplitude", "Sine Wave"),
				legend_loc="upper right",
				zoom=None,
				highlight=[(2, 4)]
			)

		
		Parameters
		----------
		y: np.ndarray
			The signal values to plot on the y-axis.
		x: np.ndarray | None
			The x-axis values. If None, indices of `y` are used.
		ax: plt.Axes | None
			matplotlib Axes object to draw on. If None, a new figure and axis are created. Return type depends on parameter value.
		color: str
			Color of the plotted line or markers. (e.g. "k")
		marker: str | None
			marker style for the plot (e.g., 'o', 'x'). If None, no marker is used.
		linestyle: str | None
			Line style for the plot (e.g., '-', '--'). If None, no line is drawn.
		stem: bool
			If True, plots a stem plot.
		labels: tuple[str, str, str] | None
			Tuple containing (title, xlabel, ylabel). If None, no labels are set.
		legend: tuple[str, str | None] | None
			Legend for the plot (e.g., ('label', 'upper right'). If None, no legend is shown.
		zoom: tuple | None
			Tuple specifying x-axis limits for zoom as (start, end). If None, full x-range is shown.
		highlight: list[tuple[float, float, str | None], ...] | None
			List of (start, end) tuples to highlight regions on the plot. e.g. [(1, 2.5), (6, 10)]
		
		Returns
		-------
		plt.Figure | None
			Figure if `ax` is None else None.
		
		
		"""
		
		assert y.ndim == 1 and y.shape[0] > 0 # 1D, Non-empty
		if x is None:
			x = np.arange(y.shape[0])
		else:
			assert x.ndim == 1 and x.shape[0] > 0 and x.shape[0] == y.shape[0] # 1D, Non-empty, Compatible
			
		# Load figure to plot the signal
		if ax is None: # Creating a new figure
			fig, ax = plt.subplots(figsize=(15, 2))
			created_fig = True
		else: # Using the figure passed by the user as `ax`
			fig = ax.get_figure()
			created_fig = False
		
		# Add legend
		if legend is not None:
			if isinstance(legend, str):
				legend = (legend, "best")
			if isinstance(legend, tuple):
				assert len(legend) == 2
			
		if show_stem: ax.stem(x, y, linefmt="k", markerfmt='o')
		else: ax.plot(x, y, fmt, lw=1.5, ms=3)
		
		# Set meta info
		if legend is not None: ax.legend([legend[0]], loc=legend[1]) # The first arg needs to be wrapped in list
		if title is not None: ax.set_title(title)
		if ylabel is not None: ax.set_ylabel(ylabel)
		if xlabel is not None: ax.set_xlabel(xlabel)
		
		# Set limits
		if ylim is not None: ax.set_ylim(ylim)
		if xlim is not None: ax.set_xlim(xlim)
		
		# ====== Adding higlight regions ======
		if highlight_regions is not None:
			# Get the y limit for the box (height of the box)
			y_min = np.min(y) if ylim is None else ylim[0]
			y_max = np.max(y) if ylim is None else ylim[-1]
			y_range = y_max - y_min
			label_box_height = 0.20 * y_range # This is another box on top of the region (to put tag)
			
			for highlight_region in highlight_regions:
				assert isinstance(highlight_region, tuple) and len(highlight_region) == 3 # (start, end, 'tag')
				start, end, tag = highlight_region
				# Main box
				ax.add_patch(Rectangle((start, y_min), end - start, y_range, color='red', alpha=0.2, zorder=10))
				# Tag box
				ax.add_patch(Rectangle((start, y_max - label_box_height), end - start, label_box_height, color='red', alpha=0.7, zorder=11))
				# Putting tag in the tag box
				ax.text((start + end) / 2, y_max - label_box_height / 2, str(tag), ha='center', va='center', fontsize=10, color='white', fontweight='bold', zorder=12)
		# ======================================
		
		# Plot vertical lines
		if vlines: [ax.axvline(x=xpos, color='blue', linestyle='--', linewidth=1.5, zorder=5) for xpos in vlines]
		
		# Plot horizontal lines
		if hlines: [ax.axhline(y=ypos, color='blue', linestyle='--', linewidth=1.5, zorder=5) for ypos in hlines]
				
		# Show grid
		if show_grid: ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
				
		# Show/return the figure as per needed
		if created_fig:
			fig.tight_layout()
			if Plotter._in_notebook():
				plt.tight_layout()
				plt.close(fig)
				return fig
			else:
				plt.tight_layout()
				plt.show()
				return fig
	
	@staticmethod
	@validate_args_type()
	def plot_matrix(
		M: np.ndarray,
		r: np.ndarray | None = None,
		c: np.ndarray | None = None,
		ax: plt.Axes | None = None,
		cmap: str = "gray_r",
		title: str | None = None,
		Mlabel: str | None = None,
		rlabel: str | None = None,
		clabel: str | None = None,
		rlim: tuple[float, float] | None = None,
		clim: tuple[float, float] | None = None,
		highlight_regions: list[tuple[float, float, str | None]] | None = None,
		vlines: list[float] | None = None,
		hlines: list[float] | None = None,
		origin: str = "lower",  # or "lower"
		gamma: int | float | None = None,
		show_colorbar: bool = True,
		cax: plt.Axes | None = None,
		show_grid: bool = True,
		tick_mode: str = "center",  # "center" or "edge"
		n_ticks: tuple[int, int] | None = None,
	) -> plt.Figure:
		"""
		Plot a 2D matrix with optional zooming, highlighting, and grid.

		.. code-block:: python
		
			from modusa.io import Plotter
			import numpy as np
			import matplotlib.pyplot as plt
			
			# Create a 50x50 random matrix
			M = np.random.rand(50, 50)
			
			# Coordinate axes
			r = np.linspace(0, 1, M.shape[0])
			c = np.linspace(0, 1, M.shape[1])
			
			# Plot the matrix
			fig = Plotter.plot_matrix(
				M=M,
				r=r,
				c=c,
				log_compression_factor=None,
				ax=None,
				labels=None,
				zoom=None,
				highlight=None,
				cmap="viridis",
				origin="lower",
				show_colorbar=True,
				cax=None,
				show_grid=False,
				tick_mode="center",
				n_ticks=(5, 5),
			)

		
		Parameters
		----------
		M: np.ndarray
			2D matrix to plot.
		r: np.ndarray
			Row coordinate axes.
		c: np.ndarray
			Column coordinate axes.
		log_compression_factor: int | float | None
			Apply log compression to enhance contrast (if provided).
		ax: plt.Axes | None
			Matplotlib axis to draw on (creates new if None).
		labels: tuple[str, str, str, str] | None
			Labels for the plot (title, Mlabel, xlabel, ylabel).
		zoom: tuple[float, float, float, float] | None
			Zoom to (r1, r2, c1, c2) in matrix coordinates.
		highlight: list[tuple[float, float]] | None
			List of rectangles (c1, c2) to highlight.
		cmap: str
			Colormap to use.
		origin: str
			Image origin, e.g., "upper" or "lower".
		show_colorbar: bool
			Whether to display colorbar.
		cax: plt.Axes | None
			Axis to draw colorbar on (ignored if show_colorbar is False).
		show_grid: bool
			Whether to show grid lines.
		tick_mode: str
			Tick alignment mode: "center" or "edge".
		n_ticks: tuple[int, int]
			Number of ticks on row and column axes.
	
		Returns
		-------
		plt.Figure
			Matplotlib figure containing the plot.
		
		"""
		
		# Validate the important args and get the signal that needs to be plotted
		
		assert M.ndim == 2

		if r is None: r = M.shape[0]
		else: assert r.ndim == 1 and r.shape[0] == M.shape[0]
			
		if c is None: c = M.shape[1]
		else: assert c.ndim == 1 and c.shape[0] == M.shape[1]
			
		# Scale the signal if needed
		if gamma is not None: M = np.log1p(float(gamma) * M)
			
		# Load figure to plot the signal
		if ax is None: # Creating a new figure
			fig, ax = plt.subplots(figsize=(15, 4))
			created_fig = True
		else: # Using the figure passed by the user as `ax`
			fig = ax.get_figure()
			created_fig = False
		
		# Plot image
		extent = Plotter._compute_centered_extent(r, c, origin) # TODO: What does it do?
		im = ax.imshow(M, aspect="auto", cmap=cmap, origin=origin, extent=extent)
		
		# Set meta info
		if title is not None: ax.set_title(title)
		if rlabel is not None: ax.set_ylabel(rlabel)
		if clabel is not None: ax.set_xlabel(clabel)
		
		# ===== Set the ticks =====
		if n_ticks is None: n_ticks = (10, 10)
		
		if tick_mode == "center":
			ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks[0]))
			ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks[1]))  # limits ticks
			
		elif tick_mode == "edge":
			dr = np.diff(r).mean() if len(r) > 1 else 1
			dc = np.diff(c).mean() if len(c) > 1 else 1
		
			# Edge tick positions (centered)
			xticks_all = np.append(c, c[-1] + dc) - dc / 2
			yticks_all = np.append(r, r[-1] + dr) - dr / 2
		
			# Determine number of ticks
			nr, nc = n_ticks
		
			# Choose evenly spaced tick indices
			xtick_idx = np.linspace(0, len(xticks_all) - 1, nc, dtype=int)
			ytick_idx = np.linspace(0, len(yticks_all) - 1, nr, dtype=int)
		
			ax.set_xticks(xticks_all[xtick_idx])
			ax.set_yticks(yticks_all[ytick_idx])
		# ==========================
		
		# Setting limits
		if rlim is not None: ax.set_ylim(rlim)
		if clim is not None: ax.set_xlim(clim)
		
		# ===== Set Higlight regions =====
		if highlight_regions is not None:
			r_min = r.min() if rlim is None else rlim[0]
			r_max = r.max() if rlim is None else rlim[-1]
			row_range = r_max - r_min
			label_box_height = 0.08 * row_range
			
			for highlight_region in highlight_regions:
				assert len(highlight_region) == 3 # (start, end, 'tag')

				c1, c2, tag = highlight_region # start, end, 'tag'
				
				# Find the width and height if the box
				width = np.abs(c1 - c2)
				height = r_max - r_min
				
				# Main box
				ax.add_patch(Rectangle((c1, r_min), width, height, color='red', alpha=0.2, zorder=10))
				
				# Tag box
				ax.add_patch(Rectangle((c1, r_max - label_box_height), width, label_box_height, color='red', alpha=0.4, zorder=11))
				
				# Putting tag in the tag box
				ax.text((c1 + c2) / 2, r_max - (label_box_height / 2), str(tag), ha='center', va='center', fontsize=10, color='white', fontweight='bold', zorder=12)
				
		# Show colorbar
		if show_colorbar is not None:
			cbar = fig.colorbar(im, ax=ax, cax=cax)
			if Mlabel is not None: cbar.set_label(Mlabel)
				
		# Plot vertical lines
		if vlines: [ax.axvline(x=xpos, color='blue', linestyle='--', linewidth=2, zorder=5) for xpos in vlines]
				
		# Plot horizontal lines
		if hlines: [ax.axhline(y=ypos, color='blue', linestyle='--', linewidth=2, zorder=5) for ypos in hlines]
				
		# Show grid
		if show_grid: ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
			
		# Show/Return the figure as per needed
		if created_fig:
			fig.tight_layout()
			if Plotter._in_notebook():
				plt.close(fig)
				return fig
			else:
				plt.show()
				return fig
	
	@staticmethod
	def _compute_centered_extent(r: np.ndarray, c: np.ndarray, origin: str) -> list[float]:
		"""
		
		"""
		dc = np.diff(c).mean() if len(c) > 1 else 1
		dr = np.diff(r).mean() if len(r) > 1 else 1
		left   = c[0] - dc / 2
		right  = c[-1] + dc / 2
		bottom = r[0] - dr / 2
		top    = r[-1] + dr / 2
		return [left, right, top, bottom] if origin == "upper" else [left, right, bottom, top]
	
	@staticmethod
	def _in_notebook() -> bool:
		try:
			from IPython import get_ipython
			shell = get_ipython()
			return shell and shell.__class__.__name__ == "ZMQInteractiveShell"
		except ImportError:
			return False
	