"""Tests for EnvironmentInfo utility."""

from napistu_torch.utils.constants import KEY_PACKAGES
from napistu_torch.utils.environment_info import EnvironmentInfo


def test_from_current_env_to_summary_dict():
    """Test that from_current_env -> get_summary runs successfully."""
    # Create EnvironmentInfo from current environment
    env_info = EnvironmentInfo.from_current_env()

    # Verify it's a valid EnvironmentInfo instance
    assert isinstance(env_info, EnvironmentInfo)
    assert env_info.python_version is not None
    assert env_info.python_implementation is not None
    assert env_info.platform_system is not None
    assert env_info.platform_release is not None

    # Convert to summary dictionary
    summary_dict = env_info.get_summary()

    # Verify summary_dict is a dictionary
    assert isinstance(summary_dict, dict)

    # Verify required fields are present (get_summary uses simplified keys)
    assert "python" in summary_dict
    assert "platform" in summary_dict

    # Verify python field format (should be "version (implementation)")
    python_value = summary_dict["python"]
    assert isinstance(python_value, str)
    assert env_info.python_version in python_value
    assert env_info.python_implementation in python_value

    # Verify platform field format (should be "system release")
    platform_value = summary_dict["platform"]
    assert isinstance(platform_value, str)
    assert env_info.platform_system in platform_value
    assert env_info.platform_release in platform_value

    # Verify packages are included if they have versions (get_summary only includes non-None)
    if env_info.napistu_version:
        assert KEY_PACKAGES.NAPISTU in summary_dict
        assert summary_dict[KEY_PACKAGES.NAPISTU] == env_info.napistu_version

    if env_info.napistu_torch_version:
        assert KEY_PACKAGES.NAPISTU_TORCH in summary_dict
        assert (
            summary_dict[KEY_PACKAGES.NAPISTU_TORCH] == env_info.napistu_torch_version
        )

    if env_info.torch_version:
        assert KEY_PACKAGES.TORCH in summary_dict
        assert summary_dict[KEY_PACKAGES.TORCH] == env_info.torch_version

    if env_info.torch_geometric_version:
        assert KEY_PACKAGES.TORCH_GEOMETRIC in summary_dict
        assert (
            summary_dict[KEY_PACKAGES.TORCH_GEOMETRIC]
            == env_info.torch_geometric_version
        )

    if env_info.pytorch_lightning_version:
        assert KEY_PACKAGES.PYTORCH_LIGHTNING in summary_dict
        assert (
            summary_dict[KEY_PACKAGES.PYTORCH_LIGHTNING]
            == env_info.pytorch_lightning_version
        )


def test_from_current_env_with_extra_packages():
    """Test from_current_env with extra packages."""
    # Test with extra packages (some may not be installed, which is fine)
    env_info = EnvironmentInfo.from_current_env(
        extra_packages=["numpy", "pandas", "nonexistent_package"]
    )

    # Verify extra packages are captured (only if installed)
    # Packages that aren't found won't be in extra_packages dict
    assert isinstance(env_info.extra_packages, dict)

    # Verify that only packages with versions are included
    for package, pkg_version in env_info.extra_packages.items():
        assert isinstance(pkg_version, str)
        assert pkg_version  # Should not be empty

    # Verify nonexistent package is not included
    assert "nonexistent_package" not in env_info.extra_packages

    # Convert to summary dict
    summary_dict = env_info.get_summary()

    # Extra packages should appear in summary dict
    for package, version in env_info.extra_packages.items():
        assert package in summary_dict
        assert summary_dict[package] == version


def test_get_install_directions():
    """Test get_install_directions returns formatted installation commands."""
    env_info = EnvironmentInfo(
        python_version="3.11.13",
        python_implementation="CPython",
        platform_system="Darwin",
        platform_release="25.1.0",
        torch_version="2.8.0",
        napistu_version="0.8.1",
        napistu_torch_version="0.2.12",
    )
    directions = env_info.get_install_directions()
    assert "pip install torch==2.8.0" in directions
    assert "pip install 'napistu==0.8.1'" in directions
    assert "napistu-torch[pyg,lightning]==0.2.12" in directions
