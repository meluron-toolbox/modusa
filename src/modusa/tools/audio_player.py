#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import validate_args_type
from modusa.tools.base import ModusaTool
from IPython.display import display, HTML, Audio
import numpy as np

class AudioPlayer(ModusaTool):
	"""
	Provides audio player in the jupyter notebook environment.
	"""
	
	#--------Meta Information----------
	_name = "Audio Player"
	_description = ""
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-08"
	#----------------------------------
	
	@staticmethod
	def play(
		y: np.ndarray,
		sr: int,
		regions: list[tuple[float, float]] | None = None,
		title: str | None = None
	) -> None:
		"""
		Plays audio clips for given regions in Jupyter Notebooks.

		Parameters
		----------
		y : np.ndarray
			Audio time series.
		sr : int
			Sampling rate.
		regions : list of (float, float), optional
			Regions to extract and play (in seconds).
		title : str, optional
			Title to display above audio players.

		Returns
		-------
		None
		"""
		if not AudioPlayer._in_notebook():
			return
		
		if title:
			display(HTML(f"<h4>{title}</h4>"))
		
		clip_numbers = []
		timings = []
		players = []
		
		if regions:
			for i, (start_sec, end_sec) in enumerate(regions):
				
				if start_sec is None:
					start_sec = 0.0
				if end_sec is None:
					end_sec = y.shape[0] / sr
				
				start_sample = int(start_sec * sr)
				end_sample = int(end_sec * sr)
				clip = y[start_sample:end_sample]
				audio_tag = Audio(data=clip, rate=sr)._repr_html_()
				
				clip_numbers.append(f"<td style='text-align:center; border-right:1px solid #ccc; padding:6px;'>{i+1}</td>")
				timings.append(f"<td style='text-align:center; border-right:1px solid #ccc; padding:6px;'>{start_sec:.2f}s → {end_sec:.2f}s</td>")
				players.append(f"<td style='padding:6px;'>{audio_tag}</td>")
		else:
			total_duration = len(y) / sr
			audio_tag = Audio(data=y, rate=sr)._repr_html_()
			
			clip_numbers.append(f"<td style='text-align:center; border-right:1px solid #ccc; padding:6px;'>1</td>")
			timings.append(f"<td style='text-align:center; border-right:1px solid #ccc; padding:6px;'>0.00s → {total_duration:.2f}s</td>")
			players.append(f"<td style='padding:6px;'>{audio_tag}</td>")
			
		# Wrap rows in a table with border
		table_html = f"""
		<div style="display:inline-block; border:1px solid #ccc; border-radius:6px; overflow:hidden;">
			<table style="border-collapse:collapse;">
				<tr style="background-color:#f2f2f2;">
					<th style="text-align:left; padding:6px 12px;">Clip</th>
					{''.join(clip_numbers)}
				</tr>
				<tr style="background-color:#fcfcfc;">
					<th style="text-align:left; padding:6px 12px;">Timing</th>
					{''.join(timings)}
				</tr>
				<tr>
					<th style="text-align:left; padding:6px 12px;">Player</th>
					{''.join(players)}
				</tr>
			</table>
		</div>
		"""
		
		return HTML(table_html)
		
			
	@staticmethod
	def _in_notebook() -> bool:
		try:
			from IPython import get_ipython
			shell = get_ipython()
			return shell and shell.__class__.__name__ == "ZMQInteractiveShell"
		except ImportError:
			return False
		
		
