"""Setup script for memory_lib package."""

from setuptools import setup, find_packages

setup(
    name="memory-lib",
    version="0.1.0",
    description="Memory system for code intelligence",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # No external dependencies - uses SQLite3 (built-in)
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
)
