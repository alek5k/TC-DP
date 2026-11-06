"""Compatibility shim for legacy PEP 517 frontends.

All package metadata is defined in ``pyproject.toml``.  This module only
adapts that data for older tooling that falls back to executing ``setup.py``.
"""

from pathlib import Path

from setuptools import find_packages, setup

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10; provided by build-system.requires
    import tomli as tomllib


ROOT = Path(__file__).parent
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = CONFIG["project"]
SETUPTOOLS = CONFIG["tool"]["setuptools"]

setup(
    name=PROJECT["name"],
    version=PROJECT["version"],
    description=PROJECT["description"],
    long_description=(ROOT / PROJECT["readme"]).read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=PROJECT["requires-python"],
    author=PROJECT["authors"][0]["name"],
    license="MIT",
    classifiers=PROJECT["classifiers"],
    install_requires=PROJECT["dependencies"],
    extras_require=PROJECT["optional-dependencies"],
    package_dir=SETUPTOOLS["package-dir"],
    packages=find_packages(
        where=SETUPTOOLS["packages"]["find"]["where"][0],
        include=SETUPTOOLS["packages"]["find"]["include"],
    ),
    include_package_data=SETUPTOOLS["include-package-data"],
    package_data=CONFIG["tool"]["setuptools"]["package-data"],
)
