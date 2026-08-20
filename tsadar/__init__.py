"""tsadar: inverse Thomson scattering analysis and diagnostics. Re-exports the package's main entry points
for convenience -- run_for_app, ThomsonScatteringDiagnostic, ThomsonParams, get_scattering_angles."""
from .runner import run_for_app
from .core import ThomsonScatteringDiagnostic, ThomsonParams
from .data.calibration import get_scattering_angles
