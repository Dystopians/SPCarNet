# EXP-BOUNDARY — why rigid translation is NOT a claimed edit class

Evidence: `translated_base.png` (garden table+vase translated +2.2 units in x, base render of the
edited checkpoint; spec in `edit_aware/garden_translate/edit_spec.json`).

Three failure modes, all at the REPRESENTATION level (before any evidence-transport question):
1. **Shared-vertex tearing**: 39,025 boundary vertices are shared with non-selected faces (2.04M
   selected faces, 380,788 exclusive vertices). Translating exclusive vertices stretches every
   bridging triangle into streak shards across the scene — a mesh-splat trained surface is not
   segmented into rigid components.
2. **Baked illumination does not move**: the object arrives at the new location with no contact
   shadow/AO (its shading was photographed at the old location and is baked into vertex colors), and
   the vacated location's shading context is inconsistent.
3. **Evidence-side**: even with a perfect representation edit, the photographs contain the object AT
   THE OLD LOCATION under old illumination; transporting that evidence to the new location would
   require relighting — synthesis, which the no-hallucination policy forbids.

Deletion and recolor avoid all three: they never move photographed content to unphotographed
configurations. This banks the class-3/4 boundary as EVIDENCE (per the red-team plan), not assertion.
