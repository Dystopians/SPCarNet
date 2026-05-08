# ECSR Phase-C Renderer Smoke

This smoke test loads materialized PASS_STATIC checkpoint copies and
renders one train view per candidate. It verifies renderer loadability
and surface evidence generation before any longer policy-val experiment.
Held-out test views are not used.

| candidate | status | train view | valid face-id | top-error addressable | mean L1 | dMean L1 vs compact |
|---|---|---|---|---|---|---|
| bicycle_C0001 | PASS_RENDER_SMOKE | 0 | 99.985% | 99.920% | 0.068976 | -0.000023 |
| bicycle_C0074 | PASS_RENDER_SMOKE | 0 | 99.985% | 99.939% | 0.069000 | +0.000000 |
| kitchen_C0019 | PASS_RENDER_SMOKE | 0 | 99.999% | 99.994% | 0.026151 | -0.000000 |

Passed: `3 / 3`

A pass here does not accept the candidate. It only permits local
before/after rendering certificates and policy-val checks.
