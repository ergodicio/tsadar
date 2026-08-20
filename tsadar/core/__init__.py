"""Core forward-model machinery: the physics (form factor, spectrum generation), parameter modules
(ThomsonParams and distribution functions), and the instrument-response-wrapped ThomsonScatteringDiagnostic."""
from .thomson_diagnostic import ThomsonScatteringDiagnostic
from .modules.ts_params import ThomsonParams
