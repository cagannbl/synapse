"""
Tests for Synapse Unbundled Modular Packages (synapse-web & synapse-orm)
"""

import os
import sys
import pytest

# Add packages directory to sys.path
packages_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages"))
if os.path.isdir(packages_dir):
    sys.path.insert(0, os.path.join(packages_dir, "synapse-web"))
    sys.path.insert(0, os.path.join(packages_dir, "synapse-orm"))


def test_core_unbundled_shims_metadata():
    """Core synapse.web and synapse.orm should have unbundled metadata."""
    import synapse.web as web
    import synapse.orm as orm

    assert getattr(web, "IS_UNBUNDLED", False) is True
    assert getattr(web, "PACKAGE_NAME", "") == "synapse-web"

    assert getattr(orm, "IS_UNBUNDLED", False) is True
    assert getattr(orm, "PACKAGE_NAME", "") == "synapse-orm"


def test_modular_synapse_web_manifest():
    """packages/synapse-web should contain valid package configuration and code."""
    web_pkg_dir = os.path.join(packages_dir, "synapse-web")
    toml_path = os.path.join(web_pkg_dir, "synapse.toml")
    assert os.path.isfile(toml_path)

    with open(toml_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert 'name = "synapse-web"' in content

    # Test direct package import
    import synapse_web
    assert hasattr(synapse_web, "SynapseApp")
    assert hasattr(synapse_web, "Request")
    assert hasattr(synapse_web, "Response")


def test_modular_synapse_orm_manifest():
    """packages/synapse-orm should contain valid package configuration and code."""
    orm_pkg_dir = os.path.join(packages_dir, "synapse-orm")
    toml_path = os.path.join(orm_pkg_dir, "synapse.toml")
    assert os.path.isfile(toml_path)

    with open(toml_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert 'name = "synapse-orm"' in content

    # Test direct package import
    import synapse_orm
    assert hasattr(synapse_orm, "Database")
    assert hasattr(synapse_orm, "Model")
    assert hasattr(synapse_orm, "Field")
    assert hasattr(synapse_orm, "SQLiteDialect")
