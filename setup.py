import setuptools


"""
Setup script for packaging and distributing the US Visa Approval Prediction ML Project.
Uses setuptools to define package metadata, dependencies, and configuration.

library used to facilitate packaging Python projects. 
It helps build, install, and distribute Python code in a standardized way.


"""

# Read the long description from README.md for PyPI page
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


# ----------------------------
# Package Metadata
# ----------------------------

# Initial version of the package
__version__ = "0.0.0"

# GitHub repository details
REPO_NAME = "US-Visa-Approval-Prediction-ML-Project-with-Deployment"
AUTHOR_USER_NAME = "RVA-TheCoder"
AUTHOR_EMAIL = "aakash.sharma00004@gmail.com"

# The folder name inside 'src' directory containing the Python package : contains our actual Python package(s).
SRC_REPO = "us_visa"

# ----------------------------
# Setup Configuration
# ----------------------------
setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ML app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)

"""
Key Notes:
----------

Explanation of above code :

(a) name: The package name used for installation.(cnn_classifier).
(b) version, author, author_email: Metadata for package distribution.

(c) description: Short one-line description of our package for PyPI.
(d) long_description: The README content for PyPI's detailed description.
(e) long_description_content : Specifies the README format (fixed from 'long_description_content').

(f) url: Repository homepage link 
(g) project_urls: Additional helpful URLs links (e.g., Bug Tracker).
(h) package_dir={"": "src"}: Tells setuptools to look in the src/ directory for our package modules.

(i) packages=setuptools.find_packages(where="src"): Finds all packages inside the src/ directory to include in the distribution.

This script is typically saved as setup.py, and is used to:

    - Build our Python package.
    - Upload it to PyPI.
    - Install it via pip (pip install . or pip install <your-package-name> once published).

"""


