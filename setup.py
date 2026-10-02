from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="leo-drc-dualnet",
    version="0.1.0",
    description=(
        "Reliability-aware physics-guided dual-stream network "
        "for LEO signal recognition under high Doppler-rate dynamics."
    ),
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="Yehui",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.10",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "torch>=2.0",
        "h5py>=3.8",
        "PyYAML>=6.0",
        "tqdm>=4.65",
        "matplotlib>=3.7",
        "rich>=13.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
