# MeshSplatOpt Stage R3 CSEF Design

Date: 2026-05-02

## Goal

Implement a minimal Counterfactual Surface Evidence Field data contract and diagnostic collector. R3 does not edit geometry. It reads a mesh-like artifact, computes face-level evidence diagnostics, groups faces into regions, and writes samples, regions, summary CSV, and a human report.

## Data Contract

`CSEFSample` records one local surface sample:

- sample id;
- position and normal;
- region id;
- positive surface evidence;
- negative free-space evidence;
- explanation debt;
- prior support;
- topology cost;
- uncertainty;
- evidence sources;
- notes.

`CSEFRegion` groups samples by connected component and boundary/debt behavior:

- region id;
- defect type candidates;
- bounding box;
- boundary loop ids;
- mesh face indices;
- image evidence refs;
- sparse point refs;
- summary statistics.

`CSEFBuildResult` records scene and output-level metadata.

## Minimal Diagnostics

R3 computes:

- triangle area as topology cost and local support proxy;
- face normal and centroid;
- connected component id;
- boundary edge score from edges owned by exactly one triangle;
- compactness/component-size evidence;
- placeholder sparse/image evidence hooks;
- explanation debt from boundary score and dent/low-support cues;
- uncertainty from missing external evidence, small components, and floaters.

The synthetic smoke uses metadata labels only for test construction. The builder itself relies on geometry and optional hints, not clean target meshes.

## Output Files

The builder writes:

- `csef_samples.npz`;
- `csef_regions.json`;
- `csef_summary.csv`;
- `csef_report.md`.

## Gate

`PASS` requires a synthetic mesh with normal ground, a hole boundary, a floater, and a dent to produce:

- high debt on boundary/hole samples;
- high uncertainty or low positive evidence on the floater component;
- low debt on normal ground samples.
