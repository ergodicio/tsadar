# TSADAR
`TSADAR` performs Thomson Scattering analysis using Automatic Differentiation (AD) and GPUs (if available). At this time, it is heavily specialized towards analyzing data 
from OMEGA experiments at the Laboratory for Laser Energetics. However, there is no reason this cannot be extended to work with data
from other facilities

## Thomson Scattering
-- work in progress -- 

## Installation

`tsadar` is configured via `pyproject.toml` and is installed with [`uv`](https://docs.astral.sh/uv/) (or plain `pip`). The available extras are:

| Extra | Purpose |
| ----- | ------- |
| *(default)* | CPU-only JAX install — everything needed for forward/inverse model runs |
| `gpu`  | CUDA 12 JAX build + `pynvml` for GPU runs |
| `hdf`  | `pyhdf` for loading legacy HDF4 streak-camera data (requires the HDF4 system library) |
| `docs` | Sphinx + theme deps for building the docs |
| `test` | `pytest` for running the test suite |

### CPU
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### GPU (CUDA 12)
```bash
uv pip install -e ".[gpu]"
```

### Loading legacy HDF4 data
`pyhdf` needs the HDF4 system library. Install it first, then the extra:
- macOS: `brew install hdf4`
- Debian/Ubuntu: `apt install libhdf4-dev`

```bash
uv pip install -e ".[hdf]"
```

If you skip this extra, `tsadar` still imports and runs — only the legacy HDF4 loader will raise a clear error if you try to use it.

## Testing

Run the pull-request suite with:

```bash
pytest tests/ -m "not slow"
```

Physics-consistency cases can be selected independently:

```bash
pytest tests/ -m "physics and not slow"
pytest tests/ -m "physics and slow"
```

The second command runs the scheduled high-resolution reference lane. See the
[physics-validation guide](docs/source/physics_validation.rst) for the invariant
inventory, tolerance rationale, and contribution checklist.

### Windows note
If cloning onto Windows you may need `git config --global core.protectNTFS false`.

## Documentation
Go to https://tsadar.readthedocs.io/ for detailed documentation.

## Automatic Differentiation
In Thomson Scattering, as in other parameter estimation inverse problems, there can be many parameters. In the case where the forward model is known, 
gradient-based methods can be applied to solve this many parameter optimization problem. Automatic Differentiation (AD) enables fast and efficient calculation of (relatively) arbitrary numerical programs. Here, we apply it to the form factor calculation.

## Citation
1. Milder, A. L., Joglekar, A. S., Rozmus, W. & Froula, D. H. Qualitative and quantitative enhancement of parameter estimation for model-based diagnostics using automatic differentiation with an application to inertial fusion. Mach. Learn.: Sci. Technol. 5, 015026 (2024).

