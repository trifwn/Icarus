from matplotlib.markers import MarkerStyle


colors: list[str] = [
    "r",
    "k",
    "b",
    "g",
    "c",
    "m",
    "y",
    "r",
    "k",
    "b",
    "g",
    "r",
    "k",
    "b",
    "g",
    "c",
    "m",
    "y",
    "r",
    "k",
    "b",
    "g",
]
markers_str: list[str] = ["x", "o", ".", "*"]
markers = [MarkerStyle(marker) for marker in markers_str]
