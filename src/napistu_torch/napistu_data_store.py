import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from napistu.network.ng_core import NapistuGraph
from napistu.sbml_dfs_core import SBML_dfs

from napistu_torch.constants import (
    NAPISTU_DATA,
    NAPISTU_DATA_STORE,
    NAPISTU_DATA_STORE_STRUCTURE,
    VERTEX_TENSOR,
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
    pd_load(name)
        Load a pandas DataFrame from the store
    save_napistu_data(napistu_data, name=None, overwrite=False)
        Save a NapistuData object to the store
    save_vertex_tensor(vertex_tensor, name=None, overwrite=False)
        Save a VertexTensor to the store
    pd_save(dataframe, name=None, overwrite=False)
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

    def load_sbml_dfs(self) -> SBML_dfs:
        """Load the SBML_dfs from disk."""
        if self.sbml_dfs_path.is_file():
            return SBML_dfs.from_pickle(self.sbml_dfs_path)
        else:
            raise FileNotFoundError(f"SBML_dfs file not found: {self.sbml_dfs_path}")

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

    def pd_load(self, name: str) -> pd.DataFrame:
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

    def pd_save(
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

    def _load_registry(self) -> dict:
        """Load the registry from disk."""
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save_registry(self) -> None:
        """Save the registry to disk."""
        self.registry[NAPISTU_DATA_STORE.LAST_MODIFIED] = datetime.now().isoformat()

        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)


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
