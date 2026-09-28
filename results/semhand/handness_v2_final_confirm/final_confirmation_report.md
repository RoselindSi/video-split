# Handness V2 Canonical-Track Confirmation

## Frozen protocol

- Seven recordings dated 2026-09-10 through 2026-09-14, absent from the
  229-recording development manifest.
- 29 raw tracks fused into 28 canonical tracks.
- The semantic check was restricted to 21 owner-candidate tracks.
- Up to five temporally distributed views were scored per canonical track.
- Frozen ensemble: three Qwen-vision feature heads.
- Frozen per-view threshold: `0.31682583689689636`.
- Fewer than three scored views fails open.
- Any view at or above threshold retains the track.
- Rejection requires at least three views, all below threshold.

## Blind labels

| Label | Tracks |
|---|---:|
| Hand | 19 |
| Non-hand | 2 |
| Mixed | 0 |
| Unsure | 0 |

## Frozen result

| Metric | Result |
|---|---:|
| Real-hand retention | 19/19 (100%) |
| False deletions | 0/19 |
| Non-hand rejection | 0/2 (0%) |
| Fail-open due to fewer than three views | 1 track |
| Mixed canonical tracks | 0 |

The 95% Wilson interval for real-hand retention is 83.2%-100%. The non-hand
sample is too small for a stable rejection estimate; its 0/2 Wilson interval is
0%-65.8%.

All 91 hand views were above threshold (`p >= 0.894`). The two non-hand tracks
contained ten views, all above threshold (`p = 0.786-0.931`). One was a stable
thigh/knee detection and the other a stable upper-arm/elbow detection. They were
not canonical-track mixing failures.

## Interpretation

The frozen fail-open policy met its immediate safety objective on this small
lockbox: it did not remove a reviewed hand track. It supplied no cleaning value
against the observed non-hand failure class. Multi-view voting cannot repair a
consistently high semantic score on every view.

The current feature extractor embeds only the candidate crop. The failure mode
is therefore consistent with a skin/limb shortcut and missing hand-anatomy or
full-frame context. The next development cycle should add body-part hard
negatives and compare crop-only against crop-plus-context heads. Any revised
model or threshold requires a new untouched recording-level confirmation set.
