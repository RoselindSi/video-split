# Handness V2 label contract

## Question

Does the candidate box contain visible evidence of a human hand, including a
hand wearing a glove?

The label concerns presence, not ownership and not which object occupies most
of the box.

## Labels

- `hand`: a palm, finger, back of a hand, or gloved hand is visibly present.
  A box dominated by a held object is still `hand` when any hand evidence is
  visible.
- `nothand`: no visible hand evidence is present. Machine parts, tools, cloth,
  food, sleeves without a visible hand, and background belong here.
- `unsure`: the candidate is too small, blurred, occluded, or cropped to make a
  reliable decision.

`unsure` is excluded from binary training. At inference it must not cause an
owner candidate to be removed.

## Legacy labels

Legacy `owner` and `other` labels prove that a hand is present and may be
mapped to `hand`. Legacy `nothand` labels were collected under several older
tasks and must be reviewed under this contract before use as human negatives.

## Independence

All splits are grouped by recording. Frames from one recording, and therefore
frames from one track, must not appear in more than one split. Exact duplicate
view pairs are deduplicated by image hashes rather than by stem, because older
packages reused stems for different detectors and boxes.

## Track aggregation

The model predicts per view. A canonical track may be marked `nothand` only
from multiple consistent valid views. One high-confidence hand view vetoes a
track-level `nothand` decision. Missing or conflicting evidence remains
`unsure` and is treated as hand by the production gate.
