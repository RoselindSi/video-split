# Graph contract v1: 30-case independent review runbook

Status: the 30-case video packet is prepared; Reviewer 1 completed 30/30,
while Reviewer 2 has not started. **No two-reviewer agreement result or frozen
contract yet.** This is a development-only definition check. Do not touch
`final_test_v1` or treat the prior 150 local reviews as independent graph
labels.

## Inputs and selection

The preparation script uses 26 fixed, alias-mapped development candidates
covering continuous-contact boundaries, release without a boundary, same-type
renewal, cross-type switches, internal motion, missing visibility, candidate
time offsets, and closed-loop actions. It adds two `A -> B -> A` and two
overlap **candidates** from source `segments.json` temporal structure. The
structural search may use recordings represented in the frozen development
reserve, but it rejects every final-test recording and does not alter the 150
local-review sample or promote reserve labels to gold.
These structural candidates are not graph gold. A repeated text label need
not be the same latent type; overlapping segment times need not mean two
independent co-active goals. Human video inspection must confirm both strata.
If either stratum fails, replace cases under a new packet version before
either reviewer begins; never silently relabel the failed case as a positive.

The public manifest contains only anonymous case IDs, clip paths, durations,
candidate offsets, and the contract hash. The private selection key contains
recording IDs, old local judgments and source metadata; keep it outside the
reviewer page and out of any reviewer handout. The first reviewer previously
annotated some of these candidates, so the review is independent of Reviewer 2
but **not fully naive to the source material**. Disclose this limitation.

The source files are packed left/right video at 1280x480. The preparation
script crops the verified left eye to 640x480. It selects up to 150 seconds
of context from `mid.mp4`, centered on each candidate and expanded to nearby
source segments. This context is still only a window: if completion,
resumption or return falls outside it, mark the relevant link unknown and
flag the case for a longer window rather than inventing state history.

## Prepare on the dataset host

Check `df -h /workspace` first; clips and private keys stay in `/workspace`,
not a home directory. Export the latest 150 final reviews from the existing
loopback annotation service to a versioned development input. Use the same
file for both reviewers and save its SHA256. With the repository available at
`/workspace/tr1/interaction_evidence_v2_20260914`:

```bash
curl -fsS http://127.0.0.1:18766/api/export \
  -o results/auditor/interaction_evidence_v2_20260914_review150_final_export.csv
python3 -m src.auditor.boundary.interaction_graph_contract_review prepare \
  --annotations results/auditor/interaction_evidence_v2_20260914_review150_final_export.csv \
  --private-key results/auditor/interaction_evidence_v2_20260914_review150_verified/private_key.csv \
  --data-root /shared/datasets/datasets/human_ego_recording_segmentation_10fps_r01_part_01/recordings \
  --data-root /shared/datasets/datasets/human_ego_recording_segmentation_10fps_r01_part_02/recordings \
  --contract docs/interaction_graph_target_contract_v1.md \
  --final-manifest data/splits/final_test_v1.json \
  --out results/auditor/interaction_graph_contract_v1_review30
```

The output directory must be new. Preparation fails if fewer than two distinct
`A -> B -> A` or overlap candidates are found; do not create a fabricated
stratum. Inspect `private_selection_key.json` only as a curator, not as a
reviewer, and check all clips play from start to end.

## Independent review

Start two separate loopback servers with different `--reviewer` values,
different `--out` subdirectories and ports. Each person reviews all 30 cases
alone, with no access to the other's answers, old labels, selection kind,
model scores, or `segments.json` text. Expose each port only through an SSH
tunnel, never on a public interface. Use a second human for Reviewer 2; an
LLM or copied answer sheet does not establish inter-annotator agreement.

The review UI groups the non-negotiable rules above the relation choices and
the required human judgments under "必须确认". It asks for prior-episode
status, return-to-node links, and boundary time only when relevant. Contact
and release remain optional evidence, not boundary rules. Type equivalence,
next-episode status, ordinary instance links, anonymous type paths, and
overlap are derived from the selected relation and its follow-ups. These
canonical fields remain present in the saved record and scorer schema;
reviewers are not asked to enter the same decision twice.

For this 30-case packet, both reviewers use one operational convention for a
candidate that falls in a visible idle gap before the nearest later onset:
classify the later observable onset (cross-type as `NEW_ACTION`, equivalent
renewal as `SAME_ACTION_NEW_INSTANCE`) and place the boundary timestamp at
the actual onset, not the candidate offset. Use `END_ONLY` only if no later
goal is visible in the available context. Reviewer 1 reports having used
this convention; give the same written instruction to Reviewer 2 before
their first case. This convention differs from the provisional contract's
strict relation-at-proposal wording. Record that discrepancy in adjudication
and reconcile the contract text before freezing; do not silently describe the
result as agreement under the unamended wording.

```bash
python3 -m src.auditor.boundary.interaction_graph_contract_review_server \
  --manifest results/auditor/interaction_graph_contract_v1_review30/review_manifest.json \
  --out results/auditor/interaction_graph_contract_v1_review30/human_reviews \
  --reviewer reviewer_1 --port 18768
```

Reviewer 2 uses `reviewer_2` and port `18770` (`18769` belongs to an older
150-case service). Export each review via its own
`/api/export`; both must contain 30 final, human-confirmed rows.

## Agreement, adjudication, freeze

```bash
python3 -m src.auditor.boundary.interaction_graph_contract_review score \
  --manifest results/auditor/interaction_graph_contract_v1_review30/review_manifest.json \
  --reviewer-1 reviewer_1_export.csv --reviewer-2 reviewer_2_export.csv \
  --out results/auditor/interaction_graph_contract_v1_review30/agreement.json
```

The primary gate is at least 27/30 exact relation agreements. Separately
report type equivalence, prior-episode status, return-to-prior-type, instance
link and overlap agreement, plus timing interval gaps when both reviewers
mark a boundary. The scorer **never** sets `ratified=true` merely
because 27/30 agree. Humans must inspect every disagreement against the video,
record evidence and the decision-table row used, and explicitly decide whether
the contract is internally consistent. If not, version the amended contract
and repeat the affected independent reviews. Only then may a frozen copy and
its SHA256 be published. Do not call agreement on the prior 150 candidate
reviews inter-annotator reliability for this graph target.
