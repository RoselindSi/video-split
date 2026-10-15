# Blind whole-timeline graph batch v2

This selection was frozen before any of its videos were viewed for action content.

- Sample size: 30.
- Seed: `interaction-graph-blind30-v2-2026-09-21`.
- Context: 120 seconds around the midpoint of the existing hand-mosaic window.
- Exclusion: every databag in `interaction_graph_mosaic_complement_v1/selection.json`.
- Realized strata: 10 low, 10 medium, and 10 high mosaic-keyframe-count videos.
- Source groups: 30 unique groups.
- Overlap with the previous mosaic batch: 0.
- Realized durations: 26 clips are 120 seconds; four source-limited clips are
  31.033, 65.067, 103.300, and 112.000 seconds. All 30 decode successfully.
- Selection uses only IDs, dates, source groups, and mosaic keyframe counts. It does not use action labels, graph annotations, model scores, or model predictions.

The source videos live on the data machine. Rendering and the offline annotation packet are generated only after the frozen `selection.json` is transferred there.
