# add_claim: two predictions, written before the run

The reranker turned `drop_claim` from a text prior into the strongest result on
the board — excess `-0.017 → +0.522`, and it did so by the NULL collapsing
(`0.810 → 0.237`), not by accuracy rising (`0.793 → 0.759`). The open question
is what that means: a scorer that checks each claim against the video, or a
scorer that prefers whichever text claims more.

`add_claim` is the other direction. The counterfactual carries one extra claim
that did not happen.

## The two predictions

**Prediction A — bidirectional completeness.** The reranker detects the
unsupported extra claim, so the original wins under the true video, and under a
wrong video neither text is supported, so the null sits near chance. Excess is
large and readable.

**Prediction B — the null cannot separate them here.** The reranker's
`drop_claim` null of 0.237 says that under a wrong video it prefers whichever
text claims LESS: with nothing supported, every extra claim only costs. In
`add_claim` the original is the shorter text, so it wins under a wrong video
too. The null lands around 0.7–0.8, not chance, and the excess is small —
which would NOT be a failure to detect overclaiming, but the null and the
effect pointing the same way.

They are mutually exclusive on one printed number: the `add_claim` null.

    A: null ~= 0.5
    B: null ~= 0.7-0.8

## Which is decided by which number

|                        | A holds        | B holds            |
| ---------------------- | -------------- | ------------------ |
| `add_claim` null       | near 0.5       | 0.7–0.8            |
| `add_claim` excess     | large          | small, uninformative |

If B holds, the readable test is `replace_claim`, emitted alongside: one whole
clause swapped for a length-matched borrowed one, so the claim COUNT and the
word count both hold and only truth moves. Under a wrong video both texts make
the same number of unsupported claims, so its null should sit at chance under
either account, and its excess is readable either way.

## What this cannot settle

The borrowed clause is only *probably* false. It comes from another label and
is rejected if its verb or head noun already appears in the target label, but
nothing verifies it is absent from that segment's video — an adjacent action
can genuinely be present. Every such case pushes both kinds toward the null, so
a positive result is conservative and a null result is not clean evidence of
absence. The note field records which clause was borrowed, so a sample can be
audited by hand if the result turns on it.
