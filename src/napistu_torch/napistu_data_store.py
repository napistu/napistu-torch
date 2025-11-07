import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from napistu.network.ng_core import NapistuGraph
from napistu.sbml_dfs_core import SBML_dfs

from napistu_torch.configs import DataConfig, TaskConfig
from napistu_torch.constants import (
    ARTIFACT_TYPES,
    NAPISTU_DATA,
    NAPISTU_DATA_STORE,
    NAPISTU_DATA_STORE_STRUCTURE,
    VALID_ARTIFACT_TYPES,
    VERTEX_TENSOR,
)
from napistu_torch.load.artifacts import (
    DEFAULT_ARTIFACT_REGISTRY,
    ArtifactDefinition,
    create_artifact,
    validate_artifact_registry,
)
from napistu_torch.load.constants import VALID_SPLITTING_STRATEGIES
from napistu_torch.ml.constants import DEVICE
from napistu_torch.napistu_data import NapistuData
from napistu_torch.vertex_tensor import VertexTensor

logger = logging.getLogger(__name__)


class NapistuDataStore:
    """
    Manage data objects related to a single SBML_dfs/NapistuGraph pair.

    Directory structure:
    store_dir/
    ├── registry.json           # Registry of all objects in this store
    ├── napistu_raw/            # (optional raw directory)
    │   ├── sbml_dfs.pkl        # (optional copy)
    │   └── napistu_graph.pkl   # (optional copy)
    ├── napistu_data/           # organizes NapistuData objects
    |   └── (NapistuData .pt files)
    ├── vertex_tensors/          # organizes VertexTensor objects
    │   └── (VertexTensor .pt files)
    └── pandas_dfs/             # organizes pandas DataFrames
        └── (DataFrame .parquet files)

    Each store manages objects for a single biological network.

    Public Methods
    --------------
    create(store_dir, sbml_dfs_path, napistu_graph_path, copy_to_store=False, overwrite=False)
        Create a new NapistuDataStore
    ensure_artifacts(artifact_names, artifact_registry=DEFAULT_ARTIFACT_REGISTRY, overwrite=False)
        Ensure specified artifacts exist in the store, creating if missing.
    get_missing_artifacts(artifact_names)
        Check which artifacts are missing from the store.
    from_config(config)
        Create or load a NapistuDataStore from a DataConfig.
    list_napistu_datas()
        List all NapistuData names in the store
    list_vertex_tensors()
        List all VertexTensor names in the store
    list_pandas_dfs()
        List all pandas DataFrame names in the store
    load_sbml_dfs()
        Load the SBML_dfs from disk
    load_napistu_data(name, map_location="cpu")
        Load a NapistuData object from the store
    load_napistu_graph()
        Load the NapistuGraph from disk
    load_vertex_tensor(name, map_location="cpu")
        Load a VertexTensor from the store
    load_pandas_df(name)
        Load a pandas DataFrame from the store
    save_napistu_data(napistu_data, name=None, overwrite=False)
        Save a NapistuData object to the store
    save_vertex_tensor(vertex_tensor, name=None, overwrite=False)
        Save a VertexTensor to the store
    save_pandas_df(dataframe, name=None, overwrite=False)
        Save a pandas DataFrame to the store
    summary()
        Get a summary of the store contents

    Private Methods
    ---------------
    _load_registry()
        Load the registry from disk
    _save_registry()
        Save the registry to disk
    """

    def __init__(self, store_dir: Union[str, Path]):
        """
        Initialize the NapistuDataStore from an existing registry.

        Parameters
        ----------
        store_dir : Union[str, Path]
            Root directory for this store. Must contain a registry.json file.

        Raises
        ------
        FileNotFoundError
            If the registry.json file does not exist

        Examples
        --------
        >>> # Load an existing store
        >>> store = NapistuDataStore('.store')
        """
        self.store_dir = Path(store_dir)
        self.registry_path = self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.REGISTRY_FILE

        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Registry not found at {self.registry_path}. "
                f"Use NapistuDataStore.create() to initialize a new store."
            )

        # Load registry
        self.registry = self._load_registry()

        # set attributes based on values in the registry
        napistu_raw = self.registry[NAPISTU_DATA_STORE.NAPISTU_RAW]
        self.sbml_dfs_path = _resolve_path(
            napistu_raw[NAPISTU_DATA_STORE.SBML_DFS], self.store_dir
        )
        self.napistu_graph_path = _resolve_path(
            napistu_raw[NAPISTU_DATA_STORE.NAPISTU_GRAPH], self.store_dir
        )

    @classmethod
    def create(
        cls,
        store_dir: Union[str, Path],
        sbml_dfs_path: Union[str, Path],
        napistu_graph_path: Union[str, Path],
        copy_to_store: bool = False,
        overwrite: bool = False,
    ) -> "NapistuDataStore":
        """
        Create a new NapistuDataStore.

        Parameters
        ----------
        store_dir : Union[str, Path]
            Root directory for this store
        sbml_dfs_path : Union[str, Path]
            Path to the SBML_dfs pickle file
        napistu_graph_path : Union[str, Path]
            Path to the NapistuGraph pickle file
        copy_to_store : bool, default=False
            If True, copy the files into the store directory and store relative paths.
            If False, store absolute paths to the original files.
        overwrite : bool, default=False
            If True, remove existing store_dir if it exists before creating new store.
            If False, raise FileExistsError if store_dir already exists.

        Returns
        -------
        NapistuDataStore
            The newly created store

        Raises
        ------
        FileExistsError
            If a registry.json already exists at store_dir and overwrite=False
        FileNotFoundError
            If the specified napistu files don't exist

        Examples
        --------
        >>> # Create a new store with external paths
        >>> store = NapistuDataStore.create(
        ...     store_dir='./stores/ecoli',
        ...     sbml_dfs_path='/data/ecoli_sbml_dfs.pkl',
        ...     napistu_graph_path='/data/ecoli_ng.pkl',
        ...     copy_to_store=False
        ... )
        """
        store_dir = Path(store_dir)
        sbml_dfs_path = Path(sbml_dfs_path)
        napistu_graph_path = Path(napistu_graph_path)
        registry_path = store_dir / NAPISTU_DATA_STORE_STRUCTURE.REGISTRY_FILE

        # Handle overwrite logic
        if overwrite and store_dir.exists():
            logger.warning(f"Overwriting existing store at {store_dir}")
            shutil.rmtree(store_dir)

        _validate_create_inputs(registry_path, sbml_dfs_path, napistu_graph_path)

        # create directories
        store_dir.mkdir(parents=True, exist_ok=True)
        napistu_data_dir = store_dir / NAPISTU_DATA_STORE_STRUCTURE.NAPISTU_DATA
        napistu_data_dir.mkdir(exist_ok=True)
        vertex_tensors_dir = store_dir / NAPISTU_DATA_STORE_STRUCTURE.VERTEX_TENSORS
        vertex_tensors_dir.mkdir(exist_ok=True)
        pandas_dfs_dir = store_dir / NAPISTU_DATA_STORE_STRUCTURE.PANDAS_DFS
        pandas_dfs_dir.mkdir(exist_ok=True)
        if copy_to_store:
            napistu_raw_dir = store_dir / NAPISTU_DATA_STORE_STRUCTURE.NAPISTU_RAW
            napistu_raw_dir.mkdir(exist_ok=True)

        # copy sbml_dfs and napistu_graph to store if requested
        if copy_to_store:

            # Copy files to store
            cached_sbml_path = napistu_raw_dir / sbml_dfs_path.name
            cached_ng_path = napistu_raw_dir / napistu_graph_path.name

            logger.info(f"Copying SBML_dfs from {sbml_dfs_path} to {cached_sbml_path}")
            logger.info(
                f"Copying NapistuGraph from {napistu_graph_path} to {cached_ng_path}"
            )

            shutil.copy2(sbml_dfs_path, cached_sbml_path)
            shutil.copy2(napistu_graph_path, cached_ng_path)

            # Store relative paths from store_dir
            sbml_relative = cached_sbml_path.relative_to(store_dir)
            ng_relative = cached_ng_path.relative_to(store_dir)

            napistu_entry = {
                NAPISTU_DATA_STORE.SBML_DFS: str(sbml_relative),
                NAPISTU_DATA_STORE.NAPISTU_GRAPH: str(ng_relative),
            }
        else:
            # Store normalized absolute paths to original files
            napistu_entry = {
                NAPISTU_DATA_STORE.SBML_DFS: str(
                    _resolve_path(sbml_dfs_path, store_dir)
                ),
                NAPISTU_DATA_STORE.NAPISTU_GRAPH: str(
                    _resolve_path(napistu_graph_path, store_dir)
                ),
            }

        # Create initial registry
        registry = {
            NAPISTU_DATA_STORE.NAPISTU_RAW: napistu_entry,
            NAPISTU_DATA_STORE.NAPISTU_DATA: {},
            NAPISTU_DATA_STORE.VERTEX_TENSORS: {},
            NAPISTU_DATA_STORE.PANDAS_DFS: {},
        }

        # Save registry
        logger.info(f"Saving registry to {registry_path}")
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        # Return new instance
        return cls(store_dir)

    def ensure_artifacts(
        self,
        artifact_names: List[str],
        artifact_registry: Dict[str, ArtifactDefinition] = DEFAULT_ARTIFACT_REGISTRY,
        overwrite: bool = False,
    ) -> None:
        """
        Ensure specified artifacts exist in the store, creating if missing.

        This is the key method for efficient batch artifact creation.
        It loads raw data (sbml_dfs, ng) ONCE and creates all missing artifacts.

        Only checks registry for artifacts not already in store. This allows
        custom artifacts to exist in store without being in registry.

        Parameters
        ----------
        artifact_names : List[str]
            Names of artifacts to ensure exist
        artifact_registry : Dict[str, ArtifactDefinition], default=DEFAULT_ARTIFACT_REGISTRY
            Registry of artifact definitions
        overwrite : bool, default=False
            If True, recreate artifacts even if they exist

        Raises
        ------
        KeyError
            If artifact not in store and not in registry

        Examples
        --------
        >>> # Works with registry artifacts
        >>> store.ensure_artifacts(["unlabeled", "edge_prediction"])
        >>>
        >>> # Also works with custom artifacts already in store
        >>> custom_data = construct_custom_pyg_data(...)
        >>> store.save_napistu_data(custom_data, name="my_custom_artifact")
        >>> store.ensure_artifacts(["my_custom_artifact"])  # Just verifies existence
        """

        # validate artifact registry
        validate_artifact_registry(artifact_registry)

        # Determine which artifacts need creation
        if overwrite:
            to_create = artifact_names
        else:
            to_create = []
            for name in artifact_names:
                # Check if exists in store
                in_napistu_data = name in self.list_napistu_datas()
                in_vertex_tensors = name in self.list_vertex_tensors()
                in_pandas_dfs = name in self.list_pandas_dfs()

                if not (in_napistu_data or in_vertex_tensors or in_pandas_dfs):
                    to_create.append(name)

        if not to_create:
            logger.info("All requested artifacts already exist in store")
            return

        logger.info(f"Need to create {len(to_create)} artifacts: {to_create}")

        # Validate all artifacts to create are in registry
        missing_from_registry = [
            name for name in to_create if name not in artifact_registry
        ]
        if missing_from_registry:
            available = sorted(artifact_registry.keys())
            raise KeyError(
                f"Cannot create artifacts not in registry: {missing_from_registry}. "
                f"Available in registry: {available}. "
                f"To use custom artifacts, save them to the store directly using "
                f"store.save_napistu_data() or store.save_vertex_tensor() or store.save_pandas_df()."
            )

        # Load raw data ONCE (expensive operation)
        logger.info("Loading SBML_dfs and NapistuGraph for artifact creation...")
        sbml_dfs = self.load_sbml_dfs()
        napistu_graph = self.load_napistu_graph()

        # Create all missing artifacts
        for name in to_create:
            logger.info(f"Creating artifact: {name}")

            artifact = create_artifact(
                name, sbml_dfs, napistu_graph, artifact_registry=artifact_registry
            )
            definition = artifact_registry[name]

            # Save based on type
            if definition.artifact_type == ARTIFACT_TYPES.NAPISTU_DATA:
                self.save_napistu_data(artifact, name=name, overwrite=overwrite)
            elif definition.artifact_type == ARTIFACT_TYPES.VERTEX_TENSOR:
                self.save_vertex_tensor(artifact, name=name, overwrite=overwrite)
            elif definition.artifact_type == ARTIFACT_TYPES.PANDAS_DFS:
                self.save_pandas_df(artifact, name=name, overwrite=overwrite)
            else:
                raise ValueError(f"Unknown artifact type: {definition.artifact_type}")

            # Free memory
            del artifact
            logger.info(f"Successfully created and saved artifact: {name}")

        # Free raw data
        del sbml_dfs, napistu_graph
        logger.info("Artifact creation complete")

    def get_missing_artifacts(
        self,
        artifact_names: List[str],
        artifact_registry: Dict[str, ArtifactDefinition] = DEFAULT_ARTIFACT_REGISTRY,
    ) -> List[str]:
        """
        Check which artifacts are missing from the store.

        Parameters
        ----------
        artifact_names : List[str]
            Names of artifacts to check
        artifact_registry : Dict[str, ArtifactDefinition], default=DEFAULT_ARTIFACT_REGISTRY
            Registry of artifact definitions

        Returns
        -------
        List[str]
            Names of artifacts that don't exist in store

        Raises
        ------
        KeyError
            If any artifact name is not in the registry

        Examples
        --------
        >>> missing = store.get_missing_artifacts([
        ...     "unlabeled",
        ...     "edge_prediction",
        ...     "custom_artifact"
        ... ])
        >>> print(missing)
        ['edge_prediction']
        """

        missing = []
        for name in artifact_names:
            if name not in artifact_registry:
                available = sorted(artifact_registry.keys())
                raise KeyError(
                    f"Unknown artifact: '{name}'. Available in registry: {available}"
                )

            definition = artifact_registry[name]
            exists = name in self.list_artifacts(definition.artifact_type)

            if not exists:
                missing.append(name)

        return missing

    @classmethod
    def from_config(
        cls,
        config: DataConfig,
        task_config: Optional[TaskConfig] = None,
        ensure_artifacts: bool = True,
    ) -> "NapistuDataStore":
        """
        Create or load a NapistuDataStore from a DataConfig.

        Flow:
        1. If store exists and not config.overwrite: load existing store
        2. If store doesn't exist or config.overwrite: create new store
        - Uses sbml_dfs_path and napistu_graph_path from config
        - Copies to store if config.copy_to_store is True
        3. Ensure napistu_data_name, other_artifacts, and task artifacts exist (always, regardless of store creation)

        Parameters
        ----------
        config : DataConfig
            Configuration with store location, artifact paths, and requirements.
            Must include sbml_dfs_path and napistu_graph_path.
        task_config : Optional[TaskConfig], default=None
            Optional task configuration. If provided, artifacts required by the task
            will be added to the required artifacts list.
        ensure_artifacts : bool, default=True
            Whether to ensure that napistu_data_name, other_artifacts, and task artifacts exist.
            Set to False when side-loading napistu_data directly (e.g., during testing).

        Returns
        -------
        NapistuDataStore
            Ready-to-use store with all required artifacts (if ensure_artifacts=True)

        Raises
        ------
        FileNotFoundError
            If sbml_dfs_path or napistu_graph_path don't exist when creating new store

        Examples
        --------
        >>> from napistu_torch.configs import DataConfig
        >>> from pathlib import Path
        >>>
        >>> config = DataConfig(
        ...     store_dir=Path(".store/ecoli"),
        ...     sbml_dfs_path=Path("/data/ecoli_sbml_dfs.pkl"),
        ...     napistu_graph_path=Path("/data/ecoli_ng.pkl"),
        ...     copy_to_store=True,
        ...     napistu_data_name="edge_prediction",
        ...     other_artifacts=["unlabeled"]
        ... )
        >>> store = NapistuDataStore.from_config(config)
        """
        store_dir = config.store_dir
        registry_path = store_dir / NAPISTU_DATA_STORE_STRUCTURE.REGISTRY_FILE

        # Determine if we need to create or load the store
        if registry_path.exists() and not config.overwrite:
            # Store exists - load it
            logger.info(f"Loading existing store from {store_dir}")
            store = cls(store_dir)
        else:
            # Store doesn't exist or we're overwriting - create new store
            logger.info(f"Creating new store at {store_dir}")

            # Validate that paths exist
            if not config.sbml_dfs_path.is_file():
                raise FileNotFoundError(
                    f"SBML_dfs file not found: {config.sbml_dfs_path}. "
                    "Please provide a valid path in config.sbml_dfs_path"
                )
            if not config.napistu_graph_path.is_file():
                raise FileNotFoundError(
                    f"NapistuGraph file not found: {config.napistu_graph_path}. "
                    "Please provide a valid path in config.napistu_graph_path"
                )

            # Create store
            store = cls.create(
                store_dir=store_dir,
                sbml_dfs_path=config.sbml_dfs_path,
                napistu_graph_path=config.napistu_graph_path,
                copy_to_store=config.copy_to_store,
                overwrite=config.overwrite,
            )

        # Conditionally ensure required artifacts exist
        if ensure_artifacts:
            required_artifacts = [config.napistu_data_name] + config.other_artifacts

            # Add additional artifacts required by the task
            if task_config is not None:
                from napistu_torch.configs import task_config_to_artifact_names

                task_artifacts = task_config_to_artifact_names(task_config)
                required_artifacts = list(set(required_artifacts) | set(task_artifacts))

            if required_artifacts:
                logger.info(f"Ensuring required artifacts exist: {required_artifacts}")
                store.ensure_artifacts(required_artifacts, overwrite=config.overwrite)

        return store

    def list_artifacts(self, artifact_type: Optional[str] = None) -> list[str]:
        """
        List all artifact names in the store.

        Parameters
        ----------
        artifact_type : Optional[str], default=None
            Type of artifact to list. If not provided, all artifact types will be listed.

        Returns
        -------
        list[str]
            List of artifact names in the store
        """

        napistu_datas = self.list_napistu_datas()
        vertex_tensors = self.list_vertex_tensors()
        pandas_dfs = self.list_pandas_dfs()

        if artifact_type is None:
            # flag duplicates across 2 or more sets
            self._validate_no_duplicate_names(raise_error=False)
            return set(napistu_datas + vertex_tensors + pandas_dfs)
        else:
            if artifact_type == ARTIFACT_TYPES.NAPISTU_DATA:
                return napistu_datas
            elif artifact_type == ARTIFACT_TYPES.VERTEX_TENSOR:
                return vertex_tensors
            elif artifact_type == ARTIFACT_TYPES.PANDAS_DFS:
                return pandas_dfs
            else:
                raise ValueError(f"Invalid artifact type: {artifact_type}")

    def list_napistu_datas(self) -> list[str]:
        """
        List all NapistuData names in the store.

        Returns
        -------
        list[str]
        List of NapistuData names in the store
        """
        return list(self.registry[NAPISTU_DATA_STORE.NAPISTU_DATA].keys())

    def list_vertex_tensors(self) -> list[str]:
        """
        List all VertexTensor names in the store.

        Returns
        -------
        list[str]
            List of VertexTensor names in the store
        """
        return list(self.registry[NAPISTU_DATA_STORE.VERTEX_TENSORS].keys())

    def list_pandas_dfs(self) -> list[str]:
        """
        List all pandas DataFrame names in the store.

        Returns
        -------
        list[str]
            List of pandas DataFrame names in the store
        """
        return list(self.registry[NAPISTU_DATA_STORE.PANDAS_DFS].keys())

    def load_artifact(
        self, name: str, artifact_type: str
    ) -> Union[NapistuData, VertexTensor, pd.DataFrame]:
        """
        Load an artifact from the store.

        Parameters
        ----------
        name : str
            Name of the artifact to load
        artifact_type : str
            Type of the artifact to load

        Returns
        -------
        Union[NapistuData, VertexTensor, pd.DataFrame]
            The loaded artifact

        Raises
        ------
        ValueError
            If invalid artifact type
        """
        if artifact_type == ARTIFACT_TYPES.NAPISTU_DATA:
            return self.load_napistu_data(name)
        elif artifact_type == ARTIFACT_TYPES.VERTEX_TENSOR:
            return self.load_vertex_tensor(name)
        elif artifact_type == ARTIFACT_TYPES.PANDAS_DFS:
            return self.load_pandas_df(name)
        else:
            raise ValueError(f"Invalid artifact type: {artifact_type}")

    def load_napistu_data(
        self, name: str, map_location: str = DEVICE.CPU
    ) -> NapistuData:
        """
        Load a NapistuData object from the store.

        Parameters
        ----------
        name : str
            Name of the NapistuData to load
        map_location : str, default="cpu"
            Device to map tensors to

        Returns
        -------
        NapistuData
            The loaded NapistuData object

        Raises
        ------
        KeyError
            If name not found in registry
        FileNotFoundError
            If the .pt file doesn't exist
        """
        # Check if name exists in registry
        if name not in self.registry[NAPISTU_DATA_STORE.NAPISTU_DATA]:
            raise KeyError(
                f"NapistuData '{name}' not found in registry. "
                f"Available: {list(self.registry[NAPISTU_DATA_STORE.NAPISTU_DATA].keys())}"
            )

        # Get filename from registry
        entry = self.registry[NAPISTU_DATA_STORE.NAPISTU_DATA][name]
        filename = entry[NAPISTU_DATA_STORE.FILENAME]
        filepath = self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.NAPISTU_DATA / filename

        # Load and return
        logger.info(f"Loading NapistuData from {filepath}")
        return NapistuData.load(filepath, map_location=map_location)

    def load_napistu_graph(self) -> NapistuGraph:
        """Load the NapistuGraph from disk."""
        if self.napistu_graph_path.is_file():
            return NapistuGraph.from_pickle(self.napistu_graph_path)
        else:
            raise FileNotFoundError(
                f"NapistuGraph file not found: {self.napistu_graph_path}"
            )

    def load_sbml_dfs(self) -> SBML_dfs:
        """Load the SBML_dfs from disk."""
        if self.sbml_dfs_path.is_file():
            return SBML_dfs.from_pickle(self.sbml_dfs_path)
        else:
            raise FileNotFoundError(f"SBML_dfs file not found: {self.sbml_dfs_path}")

    def load_vertex_tensor(
        self, name: str, map_location: str = DEVICE.CPU
    ) -> VertexTensor:
        """
        Load a VertexTensor from the store.

        Parameters
        ----------
        name : str
            Name of the VertexTensor to load
        map_location : str, default=DEVICE.CPU
            Device to map tensors to

        Returns
        -------
        VertexTensor
            The loaded VertexTensor object

        Raises
        ------
        KeyError
            If name not found in registry
        FileNotFoundError
            If the .pt file doesn't exist
        """

        # Check if name exists in registry
        if name not in self.registry[NAPISTU_DATA_STORE.VERTEX_TENSORS]:
            raise KeyError(
                f"VertexTensor '{name}' not found in registry. "
                f"Available: {list(self.registry[NAPISTU_DATA_STORE.VERTEX_TENSORS].keys())}"
            )

        # Get filename from registry
        entry = self.registry[NAPISTU_DATA_STORE.VERTEX_TENSORS][name]
        filename = entry[NAPISTU_DATA_STORE.FILENAME]
        filepath = (
            self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.VERTEX_TENSORS / filename
        )

        # Load and return
        logger.info(f"Loading VertexTensor from {filepath}")
        return VertexTensor.load(filepath, map_location=map_location)

    def load_pandas_df(self, name: str) -> pd.DataFrame:
        """
        Load a pandas DataFrame from the store.

        Parameters
        ----------
        name : str
            Name of the pandas DataFrame to load

        Returns
        -------
        pd.DataFrame
            The loaded pandas DataFrame

        Raises
        ------
        KeyError
            If name not found in registry
        FileNotFoundError
            If the .parquet file doesn't exist
        """
        # Check if name exists in registry
        if name not in self.registry[NAPISTU_DATA_STORE.PANDAS_DFS]:
            raise KeyError(
                f"pandas DataFrame '{name}' not found in registry. "
                f"Available: {list(self.registry[NAPISTU_DATA_STORE.PANDAS_DFS].keys())}"
            )

        # Get filename from registry
        entry = self.registry[NAPISTU_DATA_STORE.PANDAS_DFS][name]
        filename = entry[NAPISTU_DATA_STORE.FILENAME]
        filepath = self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.PANDAS_DFS / filename

        # Load and return
        logger.info(f"Loading pandas DataFrame from {filepath}")
        return pd.read_parquet(filepath)

    def save_napistu_data(
        self,
        napistu_data: NapistuData,
        name: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save a NapistuData object to the store.

        Parameters
        ----------
        napistu_data : NapistuData
            The NapistuData object to save. The method will extract the
            splitting_strategy and labeling_manager from the object's attributes.
        name : str, optional
            Name to use for the registry entry and filename. If not provided,
            uses the napistu_data.name attribute.
        overwrite : bool, default=False
            If True, overwrite existing entry with same name
            If False, raise FileExistsError if name already exists

        Raises
        ------
        FileExistsError
            If name already exists in registry and overwrite=False
        ValueError
            If the splitting_strategy from the NapistuData object is invalid
        """
        # Use provided name or fall back to object's name
        if name is None:
            name = napistu_data.name
        splitting_strategy = napistu_data.splitting_strategy
        labeling_manager = getattr(napistu_data, NAPISTU_DATA.LABELING_MANAGER, None)

        # Check if name already exists
        if name in self.registry[NAPISTU_DATA_STORE.NAPISTU_DATA] and not overwrite:
            raise FileExistsError(
                f"NapistuData '{name}' already exists in registry. "
                f"Use overwrite=True to replace it."
            )
        if (
            splitting_strategy is not None
            and splitting_strategy not in VALID_SPLITTING_STRATEGIES
        ):
            raise ValueError(
                f"Invalid splitting strategy: {splitting_strategy}. Must be one of {VALID_SPLITTING_STRATEGIES}"
            )

        # Save the NapistuData object
        napistu_data_dir = self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.NAPISTU_DATA
        filename = NAPISTU_DATA_STORE.PT_TEMPLATE.format(name=name)
        filepath = napistu_data_dir / filename

        logger.info(f"Saving NapistuData to {filepath}")
        napistu_data.save(filepath)

        # Create registry entry
        entry = {
            NAPISTU_DATA_STORE.FILENAME: filename,
            NAPISTU_DATA_STORE.CREATED: datetime.now().isoformat(),
            NAPISTU_DATA.NAME: name,
            NAPISTU_DATA.LABELING_MANAGER: (
                labeling_manager.to_dict() if labeling_manager is not None else None
            ),
            NAPISTU_DATA.SPLITTING_STRATEGY: splitting_strategy,
        }

        # Update registry
        self.registry[NAPISTU_DATA_STORE.NAPISTU_DATA][name] = entry
        self._save_registry()

    def save_vertex_tensor(
        self,
        vertex_tensor: VertexTensor,
        name: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save a VertexTensor to the store.

        Parameters
        ----------
        vertex_tensor : VertexTensor
            The VertexTensor object to save
        name : str, optional
            Name for storage (registry key and filename stem). If not provided,
            uses the vertex_tensor.name attribute.
        overwrite : bool, default=False
            If True, overwrite existing entry with same name
            If False, raise FileExistsError if name already exists

        Raises
        ------
        FileExistsError
            If name already exists in registry and overwrite=False
        """
        # Use provided name or fall back to object's name
        if name is None:
            name = vertex_tensor.name

        # Check if name already exists
        if name in self.registry[NAPISTU_DATA_STORE.VERTEX_TENSORS] and not overwrite:
            raise FileExistsError(
                f"VertexTensor '{name}' already exists in registry. "
                f"Use overwrite=True to replace it."
            )

        # Save the VertexTensor object
        vertex_tensors_dir = (
            self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.VERTEX_TENSORS
        )
        vertex_tensors_dir.mkdir(exist_ok=True)
        filename = NAPISTU_DATA_STORE.PT_TEMPLATE.format(name=name)
        filepath = vertex_tensors_dir / filename

        logger.info(f"Saving VertexTensor to {filepath}")
        vertex_tensor.save(filepath)

        # Create registry entry
        entry = {
            NAPISTU_DATA_STORE.FILENAME: filename,
            NAPISTU_DATA_STORE.CREATED: datetime.now().isoformat(),
            VERTEX_TENSOR.NAME: vertex_tensor.name,
            VERTEX_TENSOR.DESCRIPTION: vertex_tensor.description,
        }

        # Update registry
        self.registry[NAPISTU_DATA_STORE.VERTEX_TENSORS][name] = entry
        self._save_registry()

    def save_pandas_df(
        self,
        df: pd.DataFrame,
        name: str,
        overwrite: bool = False,
    ) -> None:
        """
        Save a pandas DataFrame to the store.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The pandas DataFrame to save
        name : str
            Name for storage (registry key and filename stem).
        overwrite : bool, default=False
            If True, overwrite existing entry with same name
            If False, raise FileExistsError if name already exists

        Raises
        ------
        FileExistsError
            If name already exists in registry and overwrite=False
        """
        # Check if name already exists
        if name in self.registry[NAPISTU_DATA_STORE.PANDAS_DFS] and not overwrite:
            raise FileExistsError(
                f"pandas DataFrame '{name}' already exists in registry. "
                f"Use overwrite=True to replace it."
            )

        # Save the pandas DataFrame
        pandas_dfs_dir = self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.PANDAS_DFS
        pandas_dfs_dir.mkdir(exist_ok=True)
        filename = NAPISTU_DATA_STORE.PARQUET_TEMPLATE.format(name=name)
        filepath = pandas_dfs_dir / filename

        logger.info(f"Saving pandas DataFrame to {filepath}")
        df.to_parquet(filepath)

        # Create registry entry
        entry = {
            NAPISTU_DATA_STORE.FILENAME: filename,
            NAPISTU_DATA_STORE.CREATED: datetime.now().isoformat(),
            "shape": df.shape,
            "columns": list(df.columns),
        }

        # Update registry
        self.registry[NAPISTU_DATA_STORE.PANDAS_DFS][name] = entry
        self._save_registry()

    def summary(self) -> dict:
        """
        Get a summary of the store contents.

        Returns
        -------
        dict
            Dictionary containing summary information about the store
        """
        return {
            "store_dir": str(self.store_dir),
            "napistu_data_count": len(self.registry[NAPISTU_DATA_STORE.NAPISTU_DATA]),
            "vertex_tensors_count": len(
                self.registry[NAPISTU_DATA_STORE.VERTEX_TENSORS]
            ),
            "pandas_dfs_count": len(self.registry[NAPISTU_DATA_STORE.PANDAS_DFS]),
            "napistu_data_names": self.list_napistu_datas(),
            "vertex_tensor_names": self.list_vertex_tensors(),
            "pandas_df_names": self.list_pandas_dfs(),
            "last_modified": self.registry.get(NAPISTU_DATA_STORE.LAST_MODIFIED),
        }

    def validate(self) -> None:
        """
        Validate the store contents.
        """
        self._validate_no_duplicate_names(raise_error=False)

    def validate_artifact_name(
        self,
        name: str,
        artifact_registry: Dict[str, ArtifactDefinition] = DEFAULT_ARTIFACT_REGISTRY,
        required_type: Optional[str] = None,
    ) -> None:
        """
        Validate an artifact name by ensuring that it is either already in the store or available from the registry

        Parameters
        ----------
        name : str
            Name of artifact to validate
        artifact_registry : Dict[str, ArtifactDefinition], default=DEFAULT_ARTIFACT_REGISTRY
            Registry of artifact definitions
        required_type : Optional[str], default=None
            Type of artifact that is required. If not provided, any type is allowed.

        Raises
        ------
        KeyError
            If artifact is not in store and not in registry
        """

        if required_type is not None:
            if required_type not in VALID_ARTIFACT_TYPES:
                raise ValueError(
                    f"Invalid 'required_type' value ({required_type}). This must be one of: {VALID_ARTIFACT_TYPES}"
                )

            existing_w_valid_type = self.list_artifacts(required_type)
            if name in existing_w_valid_type:
                # existing name of the correct type
                return None

        # Check if already in store
        in_store = name in self.list_artifacts()
        if in_store:
            if required_type is not None:
                raise KeyError(
                    f"Artifact '{name}' already exists in store but is not of type {required_type}"
                )
            else:
                return None

        # Not in store - check if we can create it
        in_registry = name in artifact_registry

        if not in_registry:
            # Not in store and not in registry
            available_in_store = self.list_artifacts(required_type)
            available_in_registry = [
                name
                for name, defn in artifact_registry.items()
                if defn.artifact_type == required_type
            ]
            required_type_str = (
                required_type if required_type is not None else "any type"
            )

            raise ValueError(
                f"An artifact of type {required_type_str} named '{name}' was not found.\n"
                f"Available in store: {available_in_store}\n"
                f"Available from registry: {available_in_registry}\n"
                f"To add a custom artifact, save it to the store first using "
                f"store.save_napistu_data(), store.save_vertex_tensor(), or store.save_pandas_df()."
            )

        # In registry - check it's the right type
        artifact_def = artifact_registry[name]
        if required_type is not None and artifact_def.artifact_type != required_type:
            raise KeyError(
                f"Artifact '{name}' is a {artifact_def.artifact_type}, not a {required_type}"
            )

    def _load_registry(self) -> dict:
        """Load the registry from disk."""
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save_registry(self) -> None:
        """Save the registry to disk."""
        self.registry[NAPISTU_DATA_STORE.LAST_MODIFIED] = datetime.now().isoformat()

        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _validate_no_duplicate_names(self, raise_error: bool = True) -> None:
        """Check for duplicate names across artifact types and warn if found."""
        name_to_types = defaultdict(list)

        for name in self.list_napistu_datas():
            name_to_types[name].append(ARTIFACT_TYPES.NAPISTU_DATA)
        for name in self.list_vertex_tensors():
            name_to_types[name].append(ARTIFACT_TYPES.VERTEX_TENSOR)
        for name in self.list_pandas_dfs():
            name_to_types[name].append(ARTIFACT_TYPES.PANDAS_DFS)

        duplicates = {
            name: types for name, types in name_to_types.items() if len(types) > 1
        }

        if duplicates:
            msg = "Duplicate artifact names found:\n" + "\n".join(
                f"  - '{name}' in {types}" for name, types in sorted(duplicates.items())
            )

            if raise_error:
                raise ValueError(msg)
            else:
                logger.warning(msg)


# private functions


def _validate_create_inputs(
    registry_path: Path,
    sbml_dfs_path: Path,
    napistu_graph_path: Path,
) -> None:
    """
    Validate inputs for creating a new NapistuDataStore.

    Parameters
    ----------
    registry_path : Path
        Path where the registry file should be created
    sbml_dfs_path : Union[str, Path]
        Path to the SBML_dfs pickle file
    napistu_graph_path : Union[str, Path]
        Path to the NapistuGraph pickle file

    Raises
    ------
    FileExistsError
        If a registry already exists at registry_path
    FileNotFoundError
        If the specified napistu files don't exist
    """
    # Check if registry already exists
    if registry_path.exists():
        raise FileExistsError(
            f"Registry already exists at {registry_path}. "
            f"Use NapistuDataStore(store_dir) to load it."
        )

    if not sbml_dfs_path.is_file():
        raise FileNotFoundError(f"SBML_dfs file not found: {sbml_dfs_path}")
    if not napistu_graph_path.is_file():
        raise FileNotFoundError(f"NapistuGraph file not found: {napistu_graph_path}")


def _resolve_path(path_str: str, store_dir: Path) -> Path:
    """
    Resolve a path string to a normalized absolute Path.

    If the path starts with '/', it's treated as an absolute path.
    Otherwise, it's treated as relative to store_dir.
    All paths are normalized to resolve .. components and symbolic links.

    Parameters
    ----------
    path_str : str
        Path string from registry (either absolute or relative)
    store_dir : Path
        Store directory to resolve relative paths against

    Returns
    -------
    Path
        Resolved and normalized absolute path
    """
    path = Path(path_str)

    if path.is_absolute():
        return path.resolve()
    else:
        # paths are relative to store_dir
        return (store_dir / path).resolve()
