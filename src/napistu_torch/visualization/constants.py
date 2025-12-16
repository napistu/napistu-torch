"""Constants for the visualization module."""

from types import SimpleNamespace

HEATMAP_AXIS = SimpleNamespace(
    ROWS="rows",
    COLUMNS="columns",
    BOTH="both",
    NONE="none",
)

VALID_HEATMAP_AXIS = list[str](HEATMAP_AXIS.__dict__.values())

CLUSTERING_DISTANCE_METRICS = SimpleNamespace(
    EUCLIDEAN="euclidean",
    MANHATTAN="manhattan",
    COSINE="cosine",
    CORRELATION="correlation",
)

VALID_CLUSTERING_DISTANCE_METRICS = list[str](
    CLUSTERING_DISTANCE_METRICS.__dict__.values()
)

CLUSTERING_LINKS = SimpleNamespace(
    SINGLE="single",
    COMPLETE="complete",
    AVERAGE="average",
    WEIGHTED="weighted",
    CENTROID="centroid",
    MEDIAN="median",
    WARD="ward",
)

VALID_CLUSTERING_LINKS = list[str](CLUSTERING_LINKS.__dict__.values())

HEATMAP_KWARGS = SimpleNamespace(
    ANNOT="annot",
    CBAR_KWS="cbar_kws",
    CMAP="cmap",
    CENTER="center",
    FMT="fmt",
    MASK="mask",
    SQUARE="square",
    XTICKLABELS="xticklabels",
    VMIN="vmin",
    VMAX="vmax",
    YTICKLABELS="yticklabels",
)
