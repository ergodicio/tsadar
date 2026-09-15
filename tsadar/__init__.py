"""tsadar: inverse Thomson scattering analysis and diagnostics. Re-exports the package's main entry points
for convenience -- run_for_app, ThomsonScatteringDiagnostic, ThomsonParams, get_scattering_angles."""

__all__ = ["run_for_app", "ThomsonScatteringDiagnostic", "ThomsonParams", "get_scattering_angles"]


def __getattr__(name):
    if name == "run_for_app":
        from .runner import run_for_app

        globals()["run_for_app"] = run_for_app
        return run_for_app
    if name in ("ThomsonScatteringDiagnostic", "ThomsonParams"):
        from .core import ThomsonScatteringDiagnostic, ThomsonParams

        globals().update(ThomsonScatteringDiagnostic=ThomsonScatteringDiagnostic, ThomsonParams=ThomsonParams)
        return globals()[name]
    if name == "get_scattering_angles":
        from .data.calibration import get_scattering_angles

        globals()["get_scattering_angles"] = get_scattering_angles
        return get_scattering_angles
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
