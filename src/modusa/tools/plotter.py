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
	def plot_signal(
		y: np.ndarray,
		x: np.ndarray | None,
		ax: plt.Axes | None = None,
		fmt: str = "k",
		title: str | None = None,
		label: str | None = None,
		ylabel: str | None = None,
		xlabel: str | None = None,
		ylim: tuple[float, float] | None = None,
		xlim: tuple[float, float] | None = None,
		highlight: list[tuple[float, float], ...] | None = None,
		vlines: list[float] | None = None,
		hlines: list[float] | None = None,
		show_grid: bool = False,
		stem: bool = False,
		legend_loc: str = None,
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
		legend_loc: str | None
			Location string for legend placement (e.g., 'upper right'). If None, no legend is shown.
		zoom: tuple | None
			Tuple specifying x-axis limits for zoom as (start, end). If None, full x-range is shown.
		highlight: list[tuple[float, float], ...] | None
			List of (start, end) tuples to highlight regions on the plot. e.g. [(1, 2.5), (6, 10)]
		
		Returns
		-------
		plt.Figure | None
			Figure if `ax` is None else None.
		
		
		"""
		
		# Validate the important args and get the signal that needs to be plotted
		if y.ndim != 1:
			raise excp.InputValueError(f"`y` must be of dimension 1 not {y.ndim}.")
		if y.shape[0] < 1:
			raise excp.InputValueError(f"`y` must not be empty.")
			
		if x is None:
			x = np.arange(y.shape[0])
		elif x.ndim != 1:
			raise excp.InputValueError(f"`x` must be of dimension 1 not {x.ndim}.")
		elif x.shape[0] < 1:
			raise excp.InputValueError(f"`x` must not be empty.")
			
		if x.shape[0] != y.shape[0]:
			raise excp.InputValueError(f"`y` and `x` must be of same shape")
			
		# Create a figure
		if ax is None:
			fig, ax = plt.subplots(figsize=(15, 2))
			created_fig = True
		else:
			fig = ax.get_figure()
			created_fig = False
		
		# Add legend
		if label is not None:
			legend_loc = legend_loc or "best"
			# Plot the signal and attach the label
			if stem:
				ax.stem(x, y, linefmt="k", markerfmt='o', label=label)
			else:
				ax.plot(x, y, fmt, lw=1.5, ms=3, label=label)
			ax.legend(loc=legend_loc)
		else:
			# Plot the signal without label
			if stem:
				ax.stem(x, y, linefmt="k", markerfmt='o')
			else:
				ax.plot(x, y, fmt, lw=1.5, ms=3)
		
		
			
		# Set the labels
		if title is not None:
			ax.set_title(title)
		if ylabel is not None:
			ax.set_ylabel(ylabel)
		if xlabel is not None:
			ax.set_xlabel(xlabel)
				
		# Applying axes limits into a region
		if ylim is not None:
			ax.set_ylim(ylim)
		if xlim is not None:
			ax.set_xlim(xlim)
			
		if highlight is not None:
			y_min = np.min(y)
			y_max = np.max(y)
			y_range = y_max - y_min
			label_box_height = 0.20 * y_range
			
			for i, highlight_region in enumerate(highlight):
				if len(highlight_region) != 2:
					raise excp.InputValueError("`highlight` should be a list of tuple of 2 values (left, right) => [(1, 10.5)]")
					
				l, r = highlight_region
				l = x[0] if l is None else l
				r = x[-1] if r is None else r
				
				# Highlight rectangle (main background)
				ax.add_patch(Rectangle(
					(l, y_min),
					r - l,
					y_range,
					color='red',
					alpha=0.2,
					zorder=10
				))
				
				# Label box inside the top of the highlight
				ax.add_patch(Rectangle(
					(l, y_max - label_box_height),
					r - l,
					label_box_height,
					color='red',
					alpha=0.4,
					zorder=11
				))
				
				# Centered label inside that box
				ax.text(
					(l + r) / 2,
					y_max - label_box_height / 2,
					str(i + 1),
					ha='center',
					va='center',
					fontsize=10,
					color='white',
					fontweight='bold',
					zorder=12
				)
		
		# Vertical lines
		if vlines:
			for xpos in vlines:
				ax.axvline(x=xpos, color='blue', linestyle='--', linewidth=2, zorder=5)
				
		# Horizontal lines
		if hlines:
			for ypos in hlines:
				ax.axhline(y=ypos, color='blue', linestyle='--', linewidth=2, zorder=5)
				
		# Show grid
		if show_grid:
			ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
				
		# Show/Return the figure as per needed
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
		highlight: list[tuple[float, float, float, float]] | None = None,
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
		highlight: list[tuple[float, float, float, float]] | None
			List of rectangles (r1, r2, c1, c2) to highlight.
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
		if M.ndim != 2:
			raise excp.InputValueError(f"`M` must have 2 dimension not {M.ndim}")
		if r is None:
			r = M.shape[0]
		if c is None:
			c = M.shape[1]
			
		if r.ndim != 1 and c.ndim != 1:
			raise excp.InputValueError(f"`r` and `c` must have 2 dimension not r:{r.ndim}, c:{c.ndim}")
			
		if r.shape[0] != M.shape[0]:
			raise excp.InputValueError(f"`r` must have shape as `M row` not {r.shape}")
		if c.shape[0] != M.shape[1]:
			raise excp.InputValueError(f"`c` must have shape as `M column` not {c.shape}")
			
		# Scale the signal if needed
		if gamma is not None:
			M = np.log1p(float(gamma) * M)
			
		# Create a figure
		if ax is None:
			fig, ax = plt.subplots(figsize=(15, 4))
			created_fig = True
		else:
			fig = ax.get_figure()
			created_fig = False
			
		# Plot the signal with right configurations
		# Compute extent
		extent = Plotter._compute_centered_extent(r, c, origin)
		
		# Plot image
		im = ax.imshow(
			M,
			aspect="auto",
			cmap=cmap,
			origin=origin,
			extent=extent
		)
		
		# Set the ticks and labels
		if n_ticks is None:
			n_ticks = (10, 10)
		
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
		
		# Set the labels
		if title is not None:
			ax.set_title(title)
		if rlabel is not None:
			ax.set_ylabel(rlabel)
		if clabel is not None:
			ax.set_xlabel(clabel)
			
		# Applying axes limits into a region
		if rlim is not None:
			ax.set_ylim(rlim)
		if clim is not None:
			ax.set_xlim(clim)
				
		# Applying axes limits into a region
		if rlim is not None:
			ax.set_ylim(rlim)
		if clim is not None:
			ax.set_xlim(clim)
			
		if highlight is not None:
			row_range = r.max() - r.min()
			label_box_height = 0.08 * row_range
			
			for i, highlight_region in enumerate(highlight):
				if len(highlight_region) != 4 and len(highlight_region) != 2:
					raise excp.InputValueError(
						"`highlight` should be a list of tuple of 4 or 2 values (row_min, row_max, col_min, col_max) or (col_min, col_max) => [(1, 10.5, 2, 40)] or [(2, 40)] "
					)
				
				if len(highlight_region) == 2:
					r1, r2 = None, None
					c1, c2 = highlight_region
				elif len(highlight_region) == 4:
					r1, r2, c1, c2 = highlight_region
				
				r1 = r[0] if r1 is None else r1
				r2 = r[-1] if r2 is None else r2
				c1 = c[0] if c1 is None else c1
				c2 = c[-1] if c2 is None else c2
				
				row_min, row_max = min(r1, r2), max(r1, r2)
				col_min, col_max = min(c1, c2), max(c1, c2)
				
				width = col_max - col_min
				height = row_max - row_min
				
				# Main red highlight box
				ax.add_patch(Rectangle(
					(col_min, row_min),
					width,
					height,
					color='red',
					alpha=0.2,
					zorder=10
				))
				
				# Label box inside top of highlight (just below row_max)
				ax.add_patch(Rectangle(
					(col_min, row_max - label_box_height),
					width,
					label_box_height,
					color='red',
					alpha=0.4,
					zorder=11
				))
				
				# Centered label in that box
				ax.text(
					(col_min + col_max) / 2,
					row_max - (label_box_height / 2),
					str(i + 1),
					ha='center',
					va='center',
					fontsize=10,
					color='white',
					fontweight='bold',
					zorder=12
				)
				
		# Show colorbar
		if show_colorbar is not None:
			cbar = fig.colorbar(im, ax=ax, cax=cax)
			if Mlabel is not None:
				cbar.set_label(Mlabel)
				
		# Vertical lines
		if vlines:
			for xpos in vlines:
				ax.axvline(x=xpos, color='blue', linestyle='--', linewidth=2, zorder=5)
				
		# Horizontal lines
		if hlines:
			for ypos in hlines:
				ax.axhline(y=ypos, color='blue', linestyle='--', linewidth=2, zorder=5)
				
		# Show grid
		if show_grid:
			ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
			
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
	