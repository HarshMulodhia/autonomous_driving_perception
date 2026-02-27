from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# The primary dependency specification is environment.yml (conda).
# requirements.txt mirrors those dependencies for pip-only workflows and
# is used here so `pip install -e .` works inside an already-activated
# conda environment without re-specifying every package.
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="autonomous_driving_perception",
    version="0.1.0",
    author="HarshMulodhia",
    description="Object detection pipeline for autonomous driving perception",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HarshMulodhia/autonomous_driving_perception",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
