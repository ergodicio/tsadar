## Release

Merging cuts a patch release. Add a label to change that: `release:minor`, `release:major`, `release:draft` (release notes written by hand before publishing), or `release:skip`.

## Validation

- [ ] Fast suite passed: `pytest -m "not slow" tests/`
- [ ] The slow suite was run before submission: `pytest -m slow tests/` (note who ran it or link the result)
- [ ] GPU/JAX numerical changes were checked on a GPU (note who ran `pytest -q tests/test_form_factor/test_ratintn_gpu.py` or link the result)
