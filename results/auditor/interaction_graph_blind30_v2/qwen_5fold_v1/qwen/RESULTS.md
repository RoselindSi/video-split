# Qwen 30-video five-fold blind test

## Status

- Model: `/workspace/tr1/ckpts/Qwen3-VL-8B-Instruct`
- Video sampling: 1 fps
- Decoding: deterministic, temperature 0
- Test design: five folds, six held-out videos per fold
- Human labels exposed for the held-out video: no
- Valid model outputs: 28 / 30 (93.3%)
- Structured-output failures: `blind_graph_0011`, `blind_graph_0020`

Both failures repeatedly created hundreds of synonymous event IDs and reached
the generation limit before producing a complete JSON object. They remain
failures rather than being assigned a hand-authored fallback graph.

## Main metrics

Metrics conditional on the 28 valid outputs:

| Metric | Result |
| --- | ---: |
| Boundary F1, 0.25 s | 0.055 |
| Boundary F1, 0.5 s | 0.069 |
| Boundary F1, 1.0 s | 0.138 |
| Boundary F1, 2.0 s | 0.234 |
| Matched-boundary relation agreement at 1 s | 0.800 (8 / 10) |
| Median temporal event-partition ARI | 0.580 |
| Exact normalized event path | 5 / 28 |
| Internal-repeat time agreement | 0.593 |
| Internal-repeat Cohen kappa | 0.253 |

Failure-aware boundary F1, where a missing model output contributes no
predicted boundaries, is 0.134 at 1 second and 0.228 at 2 seconds.

## Episode-identity diagnostic

Across the 28 valid videos, the human annotation contains 121 segments and 93
boundaries; Qwen contains 80 segments and 52 boundaries. Sixteen videos are
under-segmented. Eleven human multi-segment videos collapse to a single model
segment.

The human boundary relations comprise 51 `SAME_ACTION_NEW_INSTANCE` and 42
`NEW_ACTION` transitions. Qwen predicts only 6 `SAME_ACTION_NEW_INSTANCE`
transitions and 46 `NEW_ACTION` transitions. At a one-second tolerance, only
three of the 51 human same-action/new-instance boundaries are detected at all,
and only one of those three receives the correct relation.

On the 0.5-second repeat grid, the confusion counts are:

| Human repeat | Qwen repeat | Time cells |
| --- | --- | ---: |
| true | true | 1,735 |
| true | false | 327 |
| false | true | 2,273 |
| false | false | 2,050 |

Thus Qwen detects much of the visible within-segment repetition, but marks it
far too broadly. Together with the missing same-action/new-instance boundaries,
this supports a specific failure mode: the model often absorbs a new instance
of the same action into an ongoing internally repetitive episode.

The temporal partition ARI must not be read alone. It can be 1.0 when both
annotations use one event type throughout even if Qwen misses all boundaries
between separate instances. Boundary detection and transition relation are the
primary evidence for the research question.

## Execution audit

The original run produced 26 valid outputs and four format failures. The
runner was then repaired to save raw text before parsing, remove only unused
event declarations, and execute the planned second viewing even when the first
JSON was invalid. Small endpoint snapping is implemented only up to 0.1 seconds
but was not used by any final prediction.

`blind_graph_0002` and `blind_graph_0024` produced valid second-pass outputs.
Their final records each dropped one unused event declaration; no segment,
boundary, event assignment, or repeat flag was synthesized. The other two
failures persisted after the second viewing.

Because these engineering repairs were made after observing formatting
failures, this run is a development evaluation rather than the final locked
confirmatory test. The code and prompt should now be frozen and evaluated on a
new untouched batch before making a confirmatory claim.
