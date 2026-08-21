"""Setuptools build customization for the public runtime wheel."""

from __future__ import annotations

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class RuntimeBuild(_build_py):
    """Exclude repository test modules from the installable runtime wheel."""

    def find_package_modules(self, package: str, package_dir: str):
        modules = super().find_package_modules(package, package_dir)
        return [record for record in modules if not record[1].startswith("test")]


setup(cmdclass={"build_py": RuntimeBuild})
