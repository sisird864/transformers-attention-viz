from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="transformers-attention-viz",
    version="0.1.1",
    author="Sisir Doppalapudi",
    author_email="sisird864@gmail.com",
    description="Interactive attention visualization for multi-modal transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sisird864/transformers-attention-viz",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.30.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "gradio>=3.0.0",
        "pillow>=8.0.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.900",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "attention-viz-dashboard=attention_viz.dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "attention_viz": ["assets/*", "templates/*"],
    },
)