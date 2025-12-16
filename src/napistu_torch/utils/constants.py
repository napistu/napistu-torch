"""Constants for the utils module."""

from types import SimpleNamespace

KEY_PACKAGES = SimpleNamespace(
    NAPISTU="napistu",
    NAPISTU_TORCH="napistu-torch",
    TORCH="torch",
    TORCH_GEOMETRIC="torch_geometric",
    PYTORCH_LIGHTNING="pytorch_lightning",
)

ENVIRONMENT_INFO = SimpleNamespace(
    PYTHON_VERSION="python_version",
    PYTHON_IMPLEMENTATION="python_implementation",
    PLATFORM_SYSTEM="platform_system",
    PLATFORM_RELEASE="platform_release",
    NAPISTU_VERSION="napistu_version",
    NAPISTU_TORCH_VERSION="napistu_torch_version",
    TORCH_VERSION="torch_version",
    TORCH_GEOMETRIC_VERSION="torch_geometric_version",
    PYTORCH_LIGHTNING_VERSION="pytorch_lightning_version",
    EXTRA_PACKAGES="extra_packages",
)

PACKAGES_TO_VERSION_ATTRS = {
    KEY_PACKAGES.NAPISTU: ENVIRONMENT_INFO.NAPISTU_VERSION,
    KEY_PACKAGES.NAPISTU_TORCH: ENVIRONMENT_INFO.NAPISTU_TORCH_VERSION,
    KEY_PACKAGES.TORCH: ENVIRONMENT_INFO.TORCH_VERSION,
    KEY_PACKAGES.TORCH_GEOMETRIC: ENVIRONMENT_INFO.TORCH_GEOMETRIC_VERSION,
    KEY_PACKAGES.PYTORCH_LIGHTNING: ENVIRONMENT_INFO.PYTORCH_LIGHTNING_VERSION,
}

METRIC_VALUE_TABLE = SimpleNamespace(
    METRIC="metric",
    VALUE="value",
)

CORRELATION_METHODS = SimpleNamespace(
    SPEARMAN="spearman",
    PEARSON="pearson",
)
