from types import SimpleNamespace

TRANSFORMATION = SimpleNamespace(
    COLUMNS="columns",
    TRANSFORMER="transformer",
    # attributes
    FIT="fit",
    TRANSFORM="transform",
    PASSTHROUGH="passthrough",
    # merges
    BASE="base",
    OVERRIDE="override",
)

TRANSFORM_TABLE = SimpleNamespace(
    TRANSFORM_NAME="transform_name",
    COLUMN="column",
    TRANSFORMER_TYPE="transformer_type",
)
