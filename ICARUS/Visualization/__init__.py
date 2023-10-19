from matplotlib import colors
from matplotlib.markers import MarkerStyle

cdict = {
    'red': ((0.0, 0.22, 0.0), (0.5, 1.0, 1.0), (1.0, 0.89, 1.0)),
    'green': ((0.0, 0.49, 0.0), (0.5, 1.0, 1.0), (1.0, 0.12, 1.0)),
    'blue': ((0.0, 0.72, 0.0), (0.5, 0.0, 0.0), (1.0, 0.11, 1.0)),
}

colors_ = colors.LinearSegmentedColormap('custom', cdict)

markers_str: list[str] = ["x", "o", ".", "*"]
markers = [MarkerStyle(marker) for marker in markers_str]
