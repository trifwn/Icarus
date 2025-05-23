import distinctipy
from matplotlib.markers import MarkerStyle

markers_str: list[str | int] = list(MarkerStyle.markers.keys())
markers = [MarkerStyle(marker) for marker in markers_str]

colors_ = distinctipy.get_colors(36)
