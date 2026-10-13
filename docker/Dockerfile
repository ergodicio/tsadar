# base image
FROM mambaorg/micromamba:latest

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER tsadar /app/tsadar
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md /app/
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/fit_for_app.py docker/run.sh /app/
RUN chmod +x run.sh

# The build context has no git history, so the version is passed in.
ARG TSADAR_VERSION=0+unknown

# HDF4 system lib is needed for the optional pyhdf extra (legacy data loader).
RUN micromamba install -y -n base -c conda-forge python=3.11 pip hdf4 && \
    micromamba clean --all --yes && \
    micromamba run -n base pip install --no-cache-dir /app[gpu,hdf] && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${TSADAR_VERSION} \
    micromamba run -n base pip install --no-cache-dir --no-deps --force-reinstall /app
