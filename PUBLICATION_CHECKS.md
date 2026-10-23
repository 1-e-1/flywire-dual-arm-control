# Publication checks

Date: 2026-10-24.

## root

Status: passed.

Test selection:

```text
flybrain_lab/test_circuit.py::test_no_ik_qp_or_ppo_fallback_in_direct_controller
```

Observed: 1 passed.

Environment: local Python 3.14 with PyTorch 2.9.1 and NumPy 2.4, CPU tests only.

Only the source-level controller separation check was run. The full FlyWire graph and learned readout were not loaded or validated in this release.

All distributed Python source passed static compilation. Selected text was checked for common credential/private-key formats. These checks do not rerun historical research training, simulator acceptance or physical hardware commissioning.
