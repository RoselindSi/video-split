# Core tables — graph / interaction history

Generated from saved JSON; no new model runs. See figure_manifest.json for source hashes.

## Table 1. Structural opportunity by annotation unit

| Dataset / annotation | Units | Decisions | Support nonempty | Gold in support, conditional | Multi-successor decisions | Strict G-branch | Vacuous / multi-successor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Internal / contract | 49 | 322 | 63.66% | 95.61% | 11 | 2 | 9/11 |
| Ego-Exo cooking / coarse human | 20 | 54 | 7.41% | 0.00% | 0 | 0 | — |
| Ego-Exo cooking / keystep | 20 | 240 | 15.83% | 10.53% | 9 | 9 | 0/9 |
| Ego-Exo antigen / keystep | 30 | 356 | 19.10% | 10.29% | 17 | 17 | 0/17 |
| Bike action-level / raw | 10 | 55 | — | — | 5 | 5 | — |
| Bike action-level / acquisition folded | 10 | 33 | — | — | 0 | 0 | — |

A dash means not extracted from this result, never zero. The bike comparison uses the same 10 human windows; the 30-window keystep pool is not paired.
branch_b1 is a separate document-supported result: 15 selected clips, 96 decisions, 0 G-branch, 62/62 repeated successors deterministic. It is not a new rerun.

## Table 2. Oracle dependencies and evaluation interfaces

| Condition | Previous-state query | Transition history | Prototype grouping | NEW handling | Evaluation |
|---|---|---|---|---|---|
| D visual | None | None used | Oracle prefix | Visual threshold | Historical D selection/reporting |
| D oracle | Gold | Oracle prefix | Oracle prefix | Visual threshold | Historical D selection/reporting |
| D prev | Recursive point | Oracle prefix | Oracle prefix | NEW later aligned to gold ID | Historical D selection/reporting |
| D full | Recursive predicted identity | Predicted | Predicted | Mint predicted node | Historical D; majority-map evaluation |
| U1 V | None | None used | Oracle prefix | NEW candidate logit | 5 outer / 4 inner video folds |
| U1 P/B/PG/BG | Saved causal visual posterior; point or belief | Oracle prefix | Oracle prefix | NEW mass retained / uniform fallback | Nested video CV + train-fold NLL calibration |
| U1 FBG | Not run | Not run | Not run | Not run | Gate not passed; U1 closed |

D fusion families and U1 protocols are distinct: do not compute cross-protocol method gains.

## Appendix Table A1. E2 rescored arm accuracies

| Arm | Accuracy |
|---|---:|
| M0 | 67.8% |
| M1raw | 65.1% |
| M1canon | 59.6% |
| M2raw | 76.5% |
| M2canon | 59.6% |
| M3 | 64.2% |
| M2rawd | 61.2% |
| M3raw | 48.9% |
| M3raw_noloop | 48.5% |
| M3raw_shuffled | 50.8% |
| M3raw_other | 59.9% |
| M2lab_perm | 75.9% |
| M2lab_opaque | 73.9% |
| M2lab_first | 68.7% |

Use measured correction +11.4 → −3.9pp. Earlier prose +20.4 → −6.6pp does not reproduce. Semantic-versus-permuted (+0.7pp) and semantic-versus-opaque (+2.6pp) are non-significant contrasts.

## Appendix Table A2. U1 result families

| Result file | Role | Main evidence? |
|---|---|---|
| u1_query_nested.json | Optimistic self-inclusive predecessor prototype | No; exploratory audit |
| u1_query_calibrated.json | Optimistic interface calibration | No; exploratory audit |
| u1_query_diagnostics.json | Optimistic-interface matched/noise analysis | No; not final causal robustness |
| u1_query_causal.json | Causal snapshot before query calibration | Intermediate |
| u1_query_causal_calibrated.json | Calibrated, but frozen P parameters select lambda=0 | Calibration diagnostic, not final method table |
| u1_causal_nested_calibrated.json | Final calibrated causal nested-CV query | Yes |
| u1_causal_rescue_audit.json | Four cases + nonzero matched lambda controls | Appendix mechanism |
| u1_causal_softening.json | Point-uniform / cache controls | Yes, with primary table |
| u1_causal_memory_null.json | True vs successor-identity permutation | Yes; must accompany B>V |
| u1_causal_known_bias.json | Content-free NEW threshold control | Secondary control |

Full result paths are under results/graph_u1/. These are repeated analyses of the same benchmark, not independent replications.
