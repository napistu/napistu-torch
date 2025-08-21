import logging
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from sklearn.compose import ColumnTransformer

from napistu_torch.load.constants import TRANSFORM_TABLE, TRANSFORMATION

logger = logging.getLogger(__name__)


class TransformConfig(BaseModel):
    """Configuration for a single transformation.

    Parameters
    ----------
    columns : List[str]
        Column names to transform. Must be non-empty strings.
    transformer : Any
        sklearn transformer object or 'passthrough'.
    """

    columns: List[str] = Field(..., min_length=1)
    transformer: Any = Field(...)

    @field_validator(TRANSFORMATION.COLUMNS)
    @classmethod
    def validate_columns(cls, v):
        for col in v:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("all columns must be non-empty strings")
        return v

    @field_validator(TRANSFORMATION.TRANSFORMER)
    @classmethod
    def validate_transformer(cls, v):
        if not (
            hasattr(v, TRANSFORMATION.FIT)
            or hasattr(v, TRANSFORMATION.TRANSFORM)
            or v == TRANSFORMATION.PASSTHROUGH
        ):
            raise ValueError(
                'transformer must have fit/transform methods or be "passthrough"'
            )
        return v

    model_config = {"arbitrary_types_allowed": True}


class EncodingConfig(RootModel[Dict[str, TransformConfig]]):
    """Complete encoding configuration with conflict validation.

    Parameters
    ----------
    root : Dict[str, TransformConfig]
        Dictionary mapping transform names to their configurations.
    """

    @model_validator(mode="after")
    def check_no_column_conflicts(self):
        """Ensure no column appears in multiple transforms."""
        root_dict = self.root

        column_to_transforms = defaultdict(list)
        for transform_name, transform_config in root_dict.items():
            for column in transform_config.columns:
                column_to_transforms[column].append(transform_name)

        conflicts = {
            col: transforms
            for col, transforms in column_to_transforms.items()
            if len(transforms) > 1
        }

        if conflicts:
            conflict_details = [
                f"'{col}': {transforms}" for col, transforms in conflicts.items()
            ]
            raise ValueError(f"Column conflicts: {'; '.join(conflict_details)}")

        return self


def compose_configs(
    base_config: Dict[str, Dict],
    override_config: Dict[str, Dict],
    verbose: bool = False,
) -> Dict[str, Dict]:
    """Compose two configs with merge strategy.

    Merges configs at the transform level. For cross-config column conflicts,
    the override config takes precedence while preserving non-conflicted
    columns from the base config.

    Parameters
    ----------
    base_config : Dict[str, Dict]
        Base configuration dictionary.
    override_config : Dict[str, Dict]
        Configuration to merge in.
    verbose : bool, default=False
        If True, log detailed information about conflicts and final transformations.

    Returns
    -------
    Dict[str, Dict]
        Composed configuration dictionary.

    Raises
    ------
    ValueError
        If either config is invalid.
    """

    # Validate both configs
    base_table = validate_config(base_config)
    override_table = validate_config(override_config)

    # Find cross-config conflicts
    cross_conflicts = _find_cross_config_conflicts(base_table, override_table)

    if verbose and cross_conflicts:
        logger.info("Cross-config conflicts detected:")
        for column, details in cross_conflicts.items():
            logger.info(
                f"  Column '{column}': base transforms {details[TRANSFORMATION.BASE]} -> override transforms {details[TRANSFORMATION.OVERRIDE]}"
            )
    elif verbose:
        logger.info("No cross-config conflicts detected")

    # Merge configs
    composed_dict = _merge_configs(base_config, override_config, cross_conflicts)

    if verbose:
        logger.info("Final composed transformations:")
        for transform_name, transform_config in composed_dict.items():
            transformer_type = (
                type(transform_config[TRANSFORMATION.TRANSFORMER]).__name__
                if transform_config[TRANSFORMATION.TRANSFORMER]
                != TRANSFORMATION.PASSTHROUGH
                else TRANSFORMATION.PASSTHROUGH
            )
            columns = transform_config[TRANSFORMATION.COLUMNS]
            logger.info(f"  {transform_name} ({transformer_type}): {columns}")

    # Validate final result
    validate_config(composed_dict)

    return composed_dict


def config_to_column_transformer(config_dict: Dict[str, Dict]) -> ColumnTransformer:
    """Convert validated config dict to sklearn ColumnTransformer.

    Parameters
    ----------
    config_dict : Dict[str, Dict]
        Configuration dictionary (will be validated first).

    Returns
    -------
    ColumnTransformer
        sklearn ColumnTransformer ready for fit/transform.

    Raises
    ------
    ValueError
        If config is invalid.

    Examples
    --------
    >>> config = {
    ...     'categorical': {
    ...         'columns': ['node_type', 'species_type'],
    ...         'transformer': OneHotEncoder(handle_unknown='ignore')
    ...     },
    ...     'numerical': {
    ...         'columns': ['weight', 'score'],
    ...         'transformer': StandardScaler()
    ...     }
    ... }
    >>> preprocessor = config_to_column_transformer(config)
    >>> # Equivalent to:
    >>> # ColumnTransformer([
    >>> #     ('categorical', OneHotEncoder(handle_unknown='ignore'), ['node_type', 'species_type']),
    >>> #     ('numerical', StandardScaler(), ['weight', 'score'])
    >>> # ])
    """
    # Validate config first
    validate_config(config_dict)

    # Build transformers list for ColumnTransformer
    transformers = []
    for transform_name, transform_config in config_dict.items():
        transformer = transform_config["transformer"]
        columns = transform_config["columns"]

        transformers.append((transform_name, transformer, columns))

    return ColumnTransformer(transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Get feature names from fitted ColumnTransformer using sklearn's standard method.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer instance.

    Returns
    -------
    List[str]
        List of feature names in the same order as transform output columns.

    Examples
    --------
    >>> preprocessor = config_to_column_transformer(config)
    >>> preprocessor.fit(data)  # Must fit first!
    >>> feature_names = get_feature_names(preprocessor)
    >>> # ['cat__node_type_A', 'cat__node_type_B', 'num__weight']
    """
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("ColumnTransformer must be fitted first")

    # Use sklearn's built-in method (available since sklearn 1.0+)
    return preprocessor.get_feature_names_out().tolist()


def validate_config(config_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Validate a config dict and return transform table.

    Parameters
    ----------
    config_dict : Dict[str, Dict]
        Raw configuration dictionary with transform definitions.

    Returns
    -------
    pd.DataFrame
        Transform table showing transform_name, column, transformer_type relationships.

    Raises
    ------
    ValueError
        If config structure is invalid or column conflicts exist.
    """
    try:
        validated_config = EncodingConfig(config_dict)
    except ValueError as e:
        raise ValueError(f"Config validation failed: {e}")

    return _create_transform_table(validated_config.root)


# private


def _create_transform_table(config: Dict[str, TransformConfig]) -> pd.DataFrame:
    """Create transform table from validated config."""
    rows = []
    for transform_name, transform_config in config.items():
        transformer_type = (
            type(transform_config.transformer).__name__
            if transform_config.transformer != TRANSFORMATION.PASSTHROUGH
            else TRANSFORMATION.PASSTHROUGH
        )

        for column in transform_config.columns:
            rows.append(
                {
                    TRANSFORM_TABLE.TRANSFORM_NAME: transform_name,
                    TRANSFORM_TABLE.COLUMN: column,
                    TRANSFORM_TABLE.TRANSFORMER_TYPE: transformer_type,
                }
            )

    return pd.DataFrame(rows)


def _find_cross_config_conflicts(
    base_table: pd.DataFrame, override_table: pd.DataFrame
) -> Dict[str, Dict]:
    """Find columns that appear in both config tables."""
    if base_table.empty or override_table.empty:
        return {}

    base_columns = set(base_table[TRANSFORM_TABLE.COLUMN])
    override_columns = set(override_table[TRANSFORM_TABLE.COLUMN])
    conflicted_columns = base_columns & override_columns

    conflicts = {}
    for column in conflicted_columns:
        base_transforms = base_table[base_table[TRANSFORM_TABLE.COLUMN] == column][
            TRANSFORM_TABLE.TRANSFORM_NAME
        ].tolist()
        override_transforms = override_table[
            override_table[TRANSFORM_TABLE.COLUMN] == column
        ][TRANSFORM_TABLE.TRANSFORM_NAME].tolist()

        conflicts[column] = {
            TRANSFORMATION.BASE: base_transforms,
            TRANSFORMATION.OVERRIDE: override_transforms,
        }

    return conflicts


def _merge_configs(
    base_config: Dict, override_config: Dict, cross_conflicts: Dict
) -> Dict:
    """Merge configs with merge strategy."""
    composed = base_config.copy()
    conflicted_columns = set(cross_conflicts.keys())

    for transform_name, transform_config in override_config.items():
        if transform_name in composed:
            # Merge column lists
            base_columns = set(composed[transform_name][TRANSFORMATION.COLUMNS])
            override_columns = set(transform_config[TRANSFORMATION.COLUMNS])

            # Remove conflicts from base (override wins)
            base_columns -= conflicted_columns
            merged_columns = list(base_columns | override_columns)

            composed[transform_name] = {
                TRANSFORMATION.COLUMNS: merged_columns,
                TRANSFORMATION.TRANSFORMER: transform_config[
                    TRANSFORMATION.TRANSFORMER
                ],
            }
        else:
            composed[transform_name] = transform_config

    return composed
