# Historical-label pilot: post-run review

Reviewed 2026-09-14. The original report.json, configuration, folds and scores
are preserved. No score inversion, threshold selection or retraining was done
after inspecting this result.

## Result

Five recording-disjoint folds, shared candidate pool, fixed threshold 0.5.
Only 28 binary-labeled events in 25 recordings have both feature groups:
21 boundaries and 7 non-boundaries. This fails the preset minority-support
gate. These are descriptive development diagnostics, not confirmatory results.

| Arm | OOF AUROC | Candidate TPR | Candidate FPR |
| --- | ---: | ---: | ---: |
| Visual change shape | 0.143 | 0.571 | 1.000 |
| Hand kinematics | 0.122 | 0.714 | 1.000 |
| Visual + hand | 0.116 | 0.619 | 1.000 |
| Observation quality / missingness only | 0.293 | 0.571 | 0.857 |

Fusion minus visual AUROC = -0.027. Paired recording-bootstrap 95% interval:
[-0.207, 0.125], conditional on fixed OOF predictions. There is no demonstrated
fusion gain. This is not a test of actual contact, object state, graph context,
or hand-object relative motion. It does not disprove those hypotheses.

Only one recording contains both positive and negative scored events, so the
within-recording pair accuracy cannot support a claim. One fold has only
positive test events; the other four have fusion AUROCs 0.25, 0.25, 0.0, 0.0.
Thus the poor pooled ranking is not only an across-fold calibration artifact.
All seven non-boundaries receive high boundary scores from the fused model.
Causes such as sampling bias, target mismatch and small-sample generalization
remain hypotheses; these scores alone cannot distinguish them.

## Data and target audit

All three downloaded CSVs matched their remote SHA256 hashes. The report records
the local hashes. Visual CSV: 412 rows; hand kinematics: 313 rows; old HOI: 36.
The latter is quarantined: its summaries match the scripted synthetic pattern,
and no verified real-backend provenance accompanies it.

Of 54 direct relation labels, 22 have no row in the hand CSV, rather than an
existing row whose motion features are all missing. They are cache-coverage
gaps, not a measured detector false-negative rate. The remaining 32 comprise
28 scored internal events, two ambiguous events and two terminal events.
All seven scored negatives lack the independent negative-subtype annotation.
Only 8/18 NEW_ACTION versus 13/16 SAME_ACTION_NEW_INSTANCE labels reach the
common pool, which also changes its composition.

Crucially, src/auditor/boundary/relation_audit_sheet.py explicitly instructed
annotators to require disengagement for a new same-action instance. Direct
human labels from that sheet are still release-conditioned historical labels.
The v1 protocol's warning about historical development judgments was too weak:
this result cannot validate the broader v2 question even with more events.

The old 361-row reannotation queue omits those 54 direct judgments. It is not
a complete target-migration queue, and its descriptive event IDs expose old
candidate origins. A separate blinded development packet now rechecks the
54 old judgments without accepting them automatically as v2 gold.

## Decision

No model promotion and no graph integration based on this run. Keep the final
48 recordings and their protocol untouched. First audit target-definition
drift on the blinded development packet, then add preselected within-recording
counterexamples. Independently validate real contact observations before E2;
the current motion cache is not a substitute for them.

## E2 readiness check

Read-only checks searched /workspace/tr1/ckpts and the installed
/workspace/tr1/third_party/hand_object_detector checkout to depth four, then
searched /shared/models to depth three for hand/DOH/Faster-RCNN names. The
configured faster_rcnn_1_8_132028.pth was not found in these checks. The shared
handfilter directory contains own_ctx_best.pt and ownership-related metadata,
not an identified 100DOH contact checkpoint. This is a bounded search result,
not a claim that no suitable weight exists anywhere on the server.

No real contact inference was run. No unverified pickle cache was deserialized,
no synthetic observations were substituted, and no raw video was downloaded.
The v2 review packet is prepared, but its clips and annotations are not yet
prepared; it must not be reported as a completed new benchmark.
