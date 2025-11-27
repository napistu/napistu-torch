import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Optional

from pydantic import BaseModel, Field

from napistu_torch.utils.constants import (
    PACKAGES_TO_VERSION_ATTRS,
)


class EnvironmentInfo(BaseModel):
    """
    Python environment information for reproducibility.

    Captures key package versions and Python environment details
    to help reproduce training environments and debug compatibility issues.
    """

    python_version: str = Field(..., description="Python version (e.g., '3.10.12')")
    python_implementation: str = Field(
        ..., description="Python implementation (e.g., 'CPython')"
    )
    platform_system: str = Field(
        ..., description="Operating system (e.g., 'Linux', 'Darwin')"
    )
    platform_release: str = Field(..., description="OS release version")

    # Key packages
    napistu_version: Optional[str] = Field(None, description="Napistu package version")
    napistu_torch_version: Optional[str] = Field(
        None, description="Napistu-Torch package version"
    )
    torch_version: Optional[str] = Field(None, description="PyTorch version")
    torch_geometric_version: Optional[str] = Field(
        None, description="PyTorch Geometric version"
    )
    pytorch_lightning_version: Optional[str] = Field(
        None, description="PyTorch Lightning version"
    )

    # Additional packages
    extra_packages: Dict[str, str] = Field(
        default_factory=dict, description="Additional package versions"
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_current_env(
        cls, extra_packages: Optional[list[str]] = None
    ) -> "EnvironmentInfo":
        """
        Create EnvironmentInfo from the current Python environment.

        Captures Python version, platform details, and versions of key packages
        including napistu, napistu-torch, torch, torch_geometric, and lightning.

        Parameters
        ----------
        extra_packages : Optional[list[str]], optional
            Additional package names to capture versions for, by default None

        Returns
        -------
        EnvironmentInfo
            Environment information from current Python environment

        Examples
        --------
        >>> env_info = EnvironmentInfo.from_current_env()
        >>> print(env_info.napistu_torch_version)
        '0.1.0'

        >>> # Capture additional packages
        >>> env_info = EnvironmentInfo.from_current_env(
        ...     extra_packages=['numpy', 'pandas', 'wandb']
        ... )
        >>> print(env_info.extra_packages)
        {'numpy': '1.26.0', 'pandas': '2.1.0', 'wandb': '0.16.0'}
        """
        # Get Python info
        python_version = sys.version.split()[0]  # e.g., "3.10.12"
        python_implementation = platform.python_implementation()  # e.g., "CPython"
        platform_system = platform.system()  # e.g., "Linux", "Darwin", "Windows"
        platform_release = platform.release()  # OS version

        # Get key package versions using constants
        package_versions = {}
        for package_name, field_name in PACKAGES_TO_VERSION_ATTRS.items():
            try:
                package_versions[field_name] = version(package_name)
            except PackageNotFoundError:
                package_versions[field_name] = None

        # Get extra package versions if requested
        # Only include packages that are found (filter out None values)
        extra_package_versions = {}
        if extra_packages:
            for package_name in extra_packages:
                try:
                    pkg_version = version(package_name)
                    if pkg_version:  # Only add if version is found
                        extra_package_versions[package_name] = pkg_version
                except PackageNotFoundError:
                    # Skip packages that aren't found
                    pass

        return cls(
            python_version=python_version,
            python_implementation=python_implementation,
            platform_system=platform_system,
            platform_release=platform_release,
            **package_versions,
            extra_packages=extra_package_versions,
        )

    def get_summary(self) -> Dict[str, str]:
        """
        Convert to a flat dictionary for display.

        Returns
        -------
        Dict[str, str]
            Flat dictionary with all environment info

        Examples
        --------
        >>> env_info = EnvironmentInfo.from_current_env()
        >>> summary = env_info.to_summary_dict()
        >>> for key, value in summary.items():
        ...     print(f"{key}: {value}")
        """
        summary = {
            "python": f"{self.python_version} ({self.python_implementation})",
            "platform": f"{self.platform_system} {self.platform_release}",
        }

        # Add key packages using constants and getattr
        for package_name, version_attr in PACKAGES_TO_VERSION_ATTRS.items():
            version = getattr(self, version_attr, None)
            if version:
                summary[package_name] = version

        # Add extra packages
        for package, pkg_version in self.extra_packages.items():
            if pkg_version:
                summary[package] = pkg_version

        return summary

    def get_install_directions(self) -> str:
        """
        Format environment information as installation directions.

        Returns a multiline string with suggested installation commands to reproduce the environment.

        Returns
        -------
        str
            Multiline string with installation directions
        """

        return create_install_directions(
            self.torch_version, self.napistu_version, self.napistu_torch_version
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        summary = self.get_summary()
        lines = [f"{key}: {value}" for key, value in summary.items()]
        return "\n".join(lines)


def create_install_directions(
    torch_version: str, napistu_version: str, napistu_torch_version: str
) -> str:
    """
    Format environment information as installation directions.

    Returns a multiline string with suggested installation commands to reproduce the environment.

    Parameters
    ----------
    torch_version : str
        PyTorch version
    napistu_version : str
        Napistu version
    napistu_torch_version : str
        Napistu-Torch version

    Returns
    -------
    str
        Multiline string with installation directions
    """

    lines = [
        f"pip install torch=={torch_version}",
        f"pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/{torch_version}+cpu.html",
        f"pip install 'napistu=={napistu_version}'",
        f"pip install 'napistu-torch[pyg,lightning]=={napistu_torch_version}'",
    ]

    return "\n".join(lines)
