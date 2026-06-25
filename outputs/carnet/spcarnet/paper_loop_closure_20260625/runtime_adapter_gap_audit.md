# Runtime Adapter Gap Audit

Date: 2026-06-25

Render-only runtime evidence existed before this step. The isolated adapter profiler closes the component timing gap:

```text
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.md
```

The integrated no-I/O profiler closes the exact renderer-forward + Phase-J `adapt_frame` timing gap:

```text
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.md
```

Integrated full9 result: `951.410896 ms/view`, `1.051071 FPS`, about `27.044247x` slower than compact render-only. The adapter remains the dominant bottleneck.

Still missing for a deployment-speed claim: a promoted checkpoint-baked endpoint or accelerated adapter, and a deployment profile that includes PNG writing, downstream I/O, and optional metric computation if those are part of the claimed use case.
