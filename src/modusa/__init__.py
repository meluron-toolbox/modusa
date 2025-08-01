from modusa.utils import excp, config


from modusa.generators.s_ax import SAxGen as sax
from modusa.generators.t_ax import TAxGen as tax

from modusa.generators.s1d import S1DGen as s1d
from modusa.generators.tds import TDSGen as tds
from modusa.generators.audio import AudioGen as audio

from modusa.generators.s2d import S2DGen as s2d
from modusa.generators.ftds import FTDSGen as ftds

#=====Giving access to plot functions to plot multiple signals.=====
from modusa.utils.plot import plot_multiple_signals as plot
#=====