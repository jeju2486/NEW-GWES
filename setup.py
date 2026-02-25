#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_text(relpath: str) -> str:
    p = ROOT / relpath
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_version() -> str:
    """
    Prefer a single source of truth:
    put __version__ = "0.1.0" in gwes/__init__.py
    """
    init_py = read_text("gwes/__init__.py")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_py)
    if not m:
        # fallback so packaging doesn't break if you forget __version__
        return "0.0.0"
    return m.group(1)


setup(
    name="gwes",
    version=read_version(),
    description="Tree/Ecology-aware GWES pipeline for pangenome epistasis-like dependencies",
    long_description=read_text("README.md"),
    long_description_content_type="text/markdown",
    author="",
    url="",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,

    # If you keep scripts/ at repo root, this installs it into a shared location.
    # Users should not assume scripts/ exists after installation; use package resources or a wrapper.
    data_files=[
        ("share/gwes/scripts", ["scripts/gwes_plotting.r"])
    ] if (ROOT / "scripts" / "gwes_plotting.r").exists() else [],

    install_requires=[
        "numpy>=1.23",
        "scipy>=1.10",
        "pandas>=1.5",
        "tqdm>=4.64",
        # Add others your code actually imports at runtime, e.g.:
        # "biopython>=1.80",
        # "dendropy>=4.6",
        # "ete3>=3.1.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
            "ruff>=0.4",
            "black>=24",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
    "console_scripts": [
        "gwes-stage0=gwes.stages.stage0_prepare:main",
        "gwes-stage1=gwes.stages.stage1_phylo_cov:main",
        "gwes-stage2=gwes.stages.stage2_score_pairs:main",
        "gwes-stage3=gwes.stages.stage3_fit_global_sigma:main",
        "gwes-stage4=gwes.stages.stage4_pairs_null_and_delta:main",
        "gwes-stage5=gwes.stages.stage5_fit_bias_refit_sigma:main",
        "gwes-stage6=gwes.stages.stage6_refit_flagged_loci_sigma_grid:main",
        "gwes-stage7=gwes.stages.stage7_patch_pairs_refined_bias:main",
        "gwes-stage8=gwes.stages.stage8_bootstrap_top_pairs:main",
    ]
    },
)
