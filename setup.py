"""
B-PLIS-RAG: Bilingual Legal/Commerce RAG with ReFT and Activation Steering
Setup script for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="b-plis-rag",
    version="0.1.0",
    author="B-PLIS Team",
    author_email="team@example.com",
    description="Bilingual Legal/Commerce RAG with ReFT and Activation Steering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/B-PLIS-rag",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/B-PLIS-rag/issues",
        "Documentation": "https://github.com/your-org/B-PLIS-rag#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "ruff>=0.1.8",
            "mypy>=1.7.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "bplis-rag=main:main",
            "bplis-train=train_reft:main",
            "bplis-eval=evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.toml", "*.json", "*.txt"],
    },
)
