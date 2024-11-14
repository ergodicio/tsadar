# base image
FROM mambaorg/micromamba:latest

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER tsadar /app/tsadar
COPY --chown=$MAMBA_USER:$MAMBA_USER env_gpu.yml docker/fit_for_app.py docker/run.sh /app/
RUN chmod +x run.sh

RUN micromamba install -y -n base -f /app/env_gpu.yml && \
    micromamba clean --all --yes
