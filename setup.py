#!/usr/bin/env python
import os, sys, subprocess

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)

# import versioneer  # noqa: E402

# get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    # metadata
    name="tsadar",
    description="Thomson Scattering Analysis software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ergodicio/tsadar",
    author="Avi Milder, Archis Joglekar",
    author_email="amild@lle.rochester.edu, archis@ergodic.io",
    version="0.1.1",
    # cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    python_requires=">=3.10",
    package_data={"tsadar": ["external/**/*"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "cv2",
        "skimage",
        "matplotlib",
        # "pyhdf", # install using conda, has hdf5 dependencies that need configuring otherwise
        "xlrd",
        "pyyaml",
        "mlflow",
        "mlflow_export_import",
        "boto3",
        "flatten-dict",
        "optax",
        "tqdm",
        "jaxopt",
        "xarray",
        "mlflow_export_import",
        "pandas",
        "interpax",
        "tabulate",
        "psutil",
    ],
    extras_require={
        "cpu": ["jax"],
        "gpu": ["jax[cuda12]"],
    },
)
