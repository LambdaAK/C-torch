from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="ctorch",
    version="0.1.0",
    description="Typed ctypes bindings for the C-torch C++ machine-learning toolkit.",
    long_description=Path("python/README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    include_package_data=True,
    package_data={"ctorch": ["py.typed", "*.so", "*.dylib", "*.dll"]},
)
