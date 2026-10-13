# Latent interaction graph target contract v1 (provisional)

Status: **definition candidate, not ratified gold**. The 30-case independent
video review and adjudication gate at the end of this document has not run.
This document defines a new, taxonomy-free sequence target. It does not change
`final_test_v1`, its split or answer key, the 150-event v2 development review,
or the release-oriented `ontology_v1_constitution.yaml`. In particular, the
old ontology's disengagement rule must not be silently imported as a graph
boundary rule. No result under one target may be reported as a result under
another without a separately versioned evaluation.

## Research unit and graph meaning

Judge an **observable, immediate outcome-directed interaction**, not a verb,
hand motion, object detection, whole household task, or presumed intent. A
**type node** is a recording-local equivalence class of immediate outcomes and
the object roles involved. It is anonymous (`A`, `B`, etc.); it is not a label
from a predefined action vocabulary and is not assumed to have the same ID in
another recording. Changing the object instance need not change the type:
finishing one fastener and starting an equivalent fastener can stay at `A`.
Changing the immediate outcome can change the type while the object stays the
same: winding then unwinding a cable is `A -> B`.

An **episode instance** is one pursuit of that outcome. Repeated strokes,
rotation cycles, direction changes, regrasping, a brief release, or an idle
pause can all occur inside one instance. Picking up, using, and returning an
object are normally phases of one closed-loop episode when they serve one
immediate outcome and no independently completed outcome intervenes. The
episode ends only when the outcome visibly completes, the attempt visibly
terminates or is abandoned, or a reset and renewed attempt is evidenced. A
release alone is not such evidence. Do not invent an invisible completion.

The graph has type nodes, **time-indexed episode instances**, and an observed
manipulation focus. An internal candidate has two separate questions:

1. Has the immediate outcome in focus changed type? If yes, `NEW_ACTION`
   emits `A -> B` (`A != B`). The `B` episode may be new or a resumption; the
   prior `A` episode may be completed, suspended, or still co-active.
2. If the type has not changed, did the prior episode visibly finish or reset
   and an equivalent episode begin? If yes, `SAME_ACTION_NEW_INSTANCE` emits
   `A -> A` and creates a new instance. Otherwise, `CONTINUE` keeps the same
   instance and **emits no self-loop edge**.

This distinction is essential: ordinary dwell at `A` is not a self-loop.
`instance_id` increments for every genuinely new visit to a type, including
when a completed `A` is revisited after `B`. A return to an **unfinished,
suspended** `A` resumes its original instance instead. At `B -> A`, the edge
is still `NEW_ACTION` because the adjacent goals differ; it is never relabeled
as `SAME_ACTION_NEW_INSTANCE` just because `A` appeared earlier. Thus node
identity, episode identity, and the adjacent edge label are separate facts.
If `B -> A` is observable but completion versus suspension of the old `A`
cannot be established, keep the cross-node relation and mark the `A` instance
link unknown. A time gap by itself does not prove a new instance.

For concurrent independent interactions, maintain a set of active episodes.
Starting `B` while `A` remains active adds `B` and an onset relation from the
active context; it does not assert that `A` ended. Do not force overlapping
episodes into a single-state transition. An observable switch in manipulation
focus between active `A` and `B` is a cross-node edge, but neither instance is
renewed. A gaze or camera shift alone is not such a switch. The exact
serialization of concurrent episodes belongs to the next schema step, not to
an implicit single-pointer assumption here.

An episode can end into an idle gap with no next goal in focus. Record its
offset as `END_ONLY`, a lifecycle marker **not an internal relation or graph
edge**. At the later onset, compare the new episode with the last observable
episode across the gap if context permits. `END_ONLY` is not `TERMINAL`:
`TERMINAL` is reserved for a one-sided recording end. Likewise, a concurrent
episode can end without ending the recording or its co-active episode.

## Evidence and abstention

The reviewer must state the visible outcome/target before and after, evidence
of completion or suspension, and why the type link is same, different, or
unknown. Object identity, hand release, continuous contact, velocity, pose,
appearance change, and camera movement are **evidence fields**, never rules
that determine the relation. A change in verb or motion direction by itself
cannot establish a boundary. An absent detector result does not establish
release or unobservability of the human video.

Use `BOUNDARY_TYPE_UNRESOLVED` if a new episode/goal is visibly engaged but the
equivalence of its outcome to the prior one cannot be judged. This is eligible
for binary boundary analysis but not self-loop-versus-cross-edge or node-reuse
analysis.
Use `UNOBSERVABLE` if even the existence of an episode boundary cannot be
established from the available context; it is **not** a negative example.
One-sided initial starts and terminal ends are recorded separately, not as
internal edges. `BOUNDARY_TYPE_UNRESOLVED` and `END_ONLY` are provisional graph
extensions of the four-way v2 local review; never coerce them into an existing
v2 class.

Locate a boundary at the observable change in episode status or onset of the
new immediate outcome. If the transition is gradual, record an uncertainty
interval rather than manufacturing a sharp frame. If the proposal time is
early or late, keep the relation at that proposal distinct from the true
boundary interval. Transition shape is independent of whether a boundary
exists. Overlap describes co-active episodes, not automatically a new type.

## Decision table

Use the first seven rows to decide the local relation. The last two refine the
`NEW_ACTION` row for returns. `Same type` requires observable equivalence of
the immediate outcome and object roles, not matching motion or a model
embedding.

| Prior and next observation | Episode renewal? | Same type? | Relation or marker | Graph effect |
| --- | --- | --- | --- | --- |
| Recording starts or ends with only one side visible | Not assessable | Not assessable | `INITIAL` / `TERMINAL` | No internal edge |
| One episode ends into idle, with no new goal yet | No successor yet | Not assessable | `END_ONLY` marker | Close episode; no edge until a later onset |
| Available video establishes neither a goal switch nor same-type episode renewal | Unknown | Unknown | `UNOBSERVABLE` | Abstain; no fabricated edge |
| Same outcome still being pursued, including motion cycles, pauses, release or regrasp | No | Yes | `CONTINUE` | Dwell in same instance; no edge |
| Prior outcome/attempt visibly ended, equivalent outcome begins again | Yes | Yes | `SAME_ACTION_NEW_INSTANCE` | `A -> A`; new instance |
| Distinct immediate outcome takes focus after or during prior activity | New goal or resumed prior episode | No | `NEW_ACTION` | `A -> B`; preserve prior status if overlapping |
| New episode/goal is visible, but type equivalence is not decidable | New engagement visible | Unknown | `BOUNDARY_TYPE_UNRESOLVED` | Boundary with unresolved edge type |
| `A -> B -> A`, first `A` completed before `B` | Yes on return | A reidentified | `NEW_ACTION` at `B -> A` | Cross-edge; new `A` instance |
| `A -> B -> A`, first `A` suspended and visibly resumed | No new `A` instance | A reidentified | `NEW_ACTION` at `B -> A` | Cross-edge; resume old `A` instance |

The last two rows do not imply that every camera return to an object is a
return to its graph node. Re-identification requires visible evidence of the
same immediate outcome and relevant object roles. If the evidence is missing,
record the link as unknown rather than relying on a common object name.

## Worked positive and negative cases

Each row states the **observable condition** required for its expected answer.
Without that condition, the case may be unobservable. These are calibration
vignettes, not a report of video agreement or new gold labels.

| Case | Observable condition | Expected relation | Why |
| --- | --- | --- | --- |
| C01 | One hand wipes the same unfinished area left and right | `CONTINUE` | Stroke reversal is internal motion |
| C02 | Cap is briefly released and regrasped while still being unscrewed | `CONTINUE` | Same outcome remains pending |
| C03 | Bottle is picked up, poured into a pot, then returned as one pour cycle | `CONTINUE` at pickup-to-pour and pour-to-return | Preparation and return serve the one pour episode |
| C04 | Drain plug is lifted, rinsed, and returned as one cleaning cycle | `CONTINUE` within the cycle | Internal phases, including final release |
| C05 | Tape is pulled to read one length and retracted as part of that measurement | `CONTINUE` | Retraction closes that one measurement |
| C06 | A slipper is moved left and right to settle its placement | `CONTINUE` | Local correction without a completed placement |
| C07 | Cable is repeatedly wound in the same uninterrupted winding attempt | `CONTINUE` | Repeated rotations are not new instances |
| C08 | Stirring pauses briefly, then resumes before the mixing outcome completes | `CONTINUE` | Pause alone is insufficient |
| S01 | Kettle is set down in a stable finished placement, then picked up for another placement | `SAME_ACTION_NEW_INSTANCE` | A second pick-place episode begins; release is incidental |
| S02 | Bottle is returned after one completed pour, then picked up for another equivalent pour | `SAME_ACTION_NEW_INSTANCE` | Closed-loop episode renews |
| S03 | One fastener is fully tightened, then an equivalent next fastener is tightened without releasing the tool | `SAME_ACTION_NEW_INSTANCE` | New target instance, same immediate outcome |
| S04 | One surface is fully wiped, then a distinct surface is wiped to the same standard | `SAME_ACTION_NEW_INSTANCE` | First outcome completed, next equivalent outcome starts |
| S05 | A measured length is recorded, tape is retracted, then a fresh measurement begins | `SAME_ACTION_NEW_INSTANCE` | Two completed measurement cycles |
| S06 | First cable-winding attempt is visibly abandoned/reset; a fresh winding attempt starts | `SAME_ACTION_NEW_INSTANCE` | Visible termination and renewal, not each rotation |
| N01 | Cable winding outcome ends and intentional unwinding begins | `NEW_ACTION` | Opposite immediate outcome |
| N02 | Slipper placement is completed; the slipper is later retrieved | `NEW_ACTION` | Place versus retrieve |
| N03 | Cap opening finishes and cap closing begins without a release | `NEW_ACTION` | Different outcome despite continuous contact |
| N04 | Table wiping finishes and the person begins rinsing the cloth | `NEW_ACTION` | Cleaning the cloth is a distinct outcome |
| N05 | Mixing pauses while seasoning is added, with visible ingredient transfer | `NEW_ACTION` at mixing-to-seasoning | Independent immediate outcome, even if mixing later resumes |
| N06 | Mixing resumes after N05 before its original outcome completed | `NEW_ACTION` at seasoning-to-mixing | `B -> A`; resume original `A` instance |
| N07 | Finished winding is followed by unwinding, then a fresh winding attempt | `NEW_ACTION` at both switches | `A -> B -> A`; return creates new `A` instance |
| N08 | Hand maintains a tool grip while switching from tightening to loosening | `NEW_ACTION` | Contact continuity cannot suppress a goal change |
| O01 | While one hand keeps an unfinished object operation active, the other begins a separate task on another object | `NEW_ACTION` for second onset | Add co-active `B`; do not close `A` |
| O02 | Manipulation focus visibly alternates between two already active, unfinished tasks | `NEW_ACTION` at each focus switch | Cross-node edges; neither episode instance renews |
| O03 | An added hand only stabilizes the same target during an unfinished operation | `CONTINUE` | More hands do not imply another goal |
| O04 | A second object operation finishes while the first remains active, with no observable focus switch | `END_ONLY` for second episode | No edge and no inferred end for the first episode |
| U01 | Hands and manipulated object leave view across the proposed change | `UNOBSERVABLE` | Boundary existence cannot be established |
| U02 | Motion changes sharply but no observable outcome or episode completion changes | `CONTINUE` | Sharp appearance change is not a boundary |
| U03 | One episode visibly ends and another starts, but the new target/outcome is obscured | `BOUNDARY_TYPE_UNRESOLVED` | Binary boundary known; self-loop versus cross-edge unknown |
| U04 | A recording begins after manipulation has started, or stops before the next goal appears | `INITIAL` / `TERMINAL` | One-sided edge, not an internal boundary |

## Ratification gate: independent video review

The 30 rows above are a **rule-coverage suite**, not 30 independent human
judgments. Before calling this contract frozen, prepare 30 real sequence
excerpts from development recordings, including at least four continuous-
contact boundaries, four releases without boundaries, four self-loops, four
cross-node switches, two `A -> B -> A` returns, two overlaps, and two uncertain
cases. A case can meet more than one stratum; record counts and source IDs.
Prefer complete pre/post context over the existing candidate-centered 12/30 s
clips when episode completion or node return falls outside that window.

Two reviewers must independently record the local relation, type equivalence,
episode completion/suspension, any return-to-node link, boundary interval,
visibility, and one-line observable evidence **without seeing each other's
answers, old labels, or model scores**. Compare on the same 30 cases. Require
at least 27/30 exact agreement on the primary relation (90%). Separately
report agreement on resolvable node returns and completion-versus-suspension;
do not hide disagreement by collapsing to binary boundary. Adjudicate every
disagreement by citing a decision-table row and video evidence. If any case
requires incompatible interpretations of a rule, revise the contract,
version it, and repeat independent review of the affected cases before
freezing. Do not replace human agreement with model-generated answers.

**Current gate:** not run. The existing 150 local reviews are a single
development annotation pass, not 30 independent sequence-level graph reviews.
