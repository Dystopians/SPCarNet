# Final Stage F13 - Paper Assets Report

Decision: `FINAL_F13_PAPER_ASSETS_PASS_TRACEABLE`.

## Assets

- method diagram: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/meshsplatopt_method_diagram.png`
- triangle count bar chart: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/triangle_count_bar_chart.png`
- Pareto summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/pareto_summary.json`
- multi-scene qualitative montage: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/qualitative_panels/final_multiscene_qualitative_montage.png`
- freeze-failure montage: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/freeze_failure_panels/freeze_failure_montage.png`
- manifest: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/paper_assets_manifest.json`
- parking_phone_tiny qualitative panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/qualitative_panels/parking_phone_tiny_qualitative_panel.png`
- bonsai qualitative panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/qualitative_panels/bonsai_qualitative_panel.png`
- courtyard qualitative panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/qualitative_panels/courtyard_qualitative_panel.png`
- room qualitative panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/qualitative_panels/room_qualitative_panel.png`
- counter qualitative panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/qualitative_panels/counter_qualitative_panel.png`
- courtyard no-freeze failure panel: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/freeze_failure_panels/courtyard_no_freeze_failure_panel.png`
- qualitative montage: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/mesh_splat_opt_cross_scene_qualitative_montage.png`
- qualitative montage: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_paper_assets/room_counter_clean_vs_csef_montage.png`

## Traceability

All quantitative assets are generated from `outputs/carnet/meshsplatopt/final_multiscene_package/main_quantitative_table.csv`. The new qualitative panels use independent render outputs and record every source image path plus the selected frame in `paper_assets_manifest.json`. The freeze-failure panel uses the F35 independent no-freeze render and shows why the strict topology-frozen recovery contract is visually load-bearing.
