"""A semantic retrieval layer on V1 and on Qwen3.8-27B, under one set of conditions.

THE QUESTION. EgoHandICL retrieves labelled exemplars by what a VLM says the
hand is doing, and puts them in context beside the query. Does that layer make
the wearer's own hands more stable, on V1 and on Qwen, measured exactly the way
the user asked:

    M1  own-hand flips      label changes per 100 consecutive own-hand frame
                            pairs, on gold-owner tracks          (lower is better)
    M2  own label on others foreign-hand frames labelled self, on gold-other
                            tracks; tracks with any such frame beside it
                                                                 (lower is better)

    G   guard, not ranked   own-hand frames labelled self / own-hand frames.
                            Without it a model that never says "self" wins both.

WHAT IS HELD FIXED FOR EVERY ARM.
    hands     fresh29 detections that V1's dump, the stereo index and the
              panorama crops all have (the 18,373 of the earlier table), every
              STRIDE-th frame from each clip's start, and among those the rows
              whose keypoints were recovered (V1 needs `hand_span`)
    tracks    the dump's track ids; gold = fresh_gold_p1 + p2
    bank      V1's own training rows (own_ctx.load on otherpkg_1 + trainpkg_T1)
              with a re-rendered clean frame; human labels
    text      one Qwen3.8 description per hand (DESCRIBE, no ownership words),
              embedded by Qwen3-VL-Embedding-2B
    retrieval one file for both families: K templates from other databags, same
              coarse hand side, by text cosine (semantic), crop-image cosine
              (visual), or at random (shuffled)
    post      two forms, both reported: raw (P >= 0.5) and held (V1's geometric
              prior blended 0.5, then OwnHold with V1's parameters and a cap of
              two owners, replayed over the same sampled frames)

THE ARMS.
    V1 family  V1     the shipped classifier's P, as dumped
               B0r    frozen V1 features, retrained head (no text, no templates)
               B1     + the hand's own description token
               B2     + K templates by semantic retrieval       <- semantic layer
               B2v    + K templates by visual retrieval
               B2s    + K random templates                     <- null
    Qwen       Q0     one boxed hand, zero-shot
               Q1     + its own description
               Q2     + K templates by semantic retrieval       <- semantic layer
               Q2v    + K templates by visual retrieval
               Q2s    + K random templates                     <- null
    Qwen's P is the probability of `true` against `false` after `{"wearer":`.
    Every B arm has the same transformer and recipe, five seeds, dev-F1
    selection on held-out recordings as V1 was selected; B0r exists so a gain
    from retraining a head is not credited to semantics.

WRITTEN BEFORE RUNNING -- THE VERDICT, per family, on the HELD form:

    the semantic layer helps  iff
      (a) M1(S) <= M1(base) and M2(S) <= M2(base), and
      (b) on at least one of the two, S is lower with a 95% recording-bootstrap
          interval of the difference that excludes zero, and
      (c) on that metric S is also lower than every control
          (V1 family: B0r and B2s; Qwen: Q2s), and
      (d) G(S) >= G(base) - GUARD

    V1 family: S = B2, base = V1.   Qwen: S = Q2, base = Q0.

The raw form gets the same rule, printed as secondary. Cross-family numbers are
descriptive only. `V1 deployed` (the dump's own 30 fps verdict at the sampled
frames) is printed as a reference and enters no verdict.

KNOWN LIMITS, STATED NOW. fresh29 has been looked at for other arms, so this is
evidence, not a clean held-out test. OwnHold's constants were tuned at 30 fps
and are replayed unchanged at 30 / STRIDE fps for every arm alike. Qwen and V1
both read the 0.6 m panorama; none of the stereo arms is part of the verdict.
"""

STRIDE = 3
K = 2
N_CANDIDATES = 10
SEEDS = 5
GUARD = 0.02
BOOT = 2000

TRAIN_PKGS = ("/workspace/otherpkg_1", "/workspace/crops/trainpkg_T1")
QWEN = "/shared/datasets/public_model/Qwen3.8-27B"
EMBEDDER = "/workspace/tr1/ckpts/Qwen3-VL-Embedding-2B"

V1_ARMS = ("B0r", "B1", "B2", "B2v", "B2s")
QWEN_ARMS = ("Q0", "Q2", "Q2s", "Q1", "Q2v")      # run in this order
VERDICTS = {"V1": ("B2", "V1", ("B0r", "B2s")),
            "Qwen": ("Q2", "Q0", ("Q2s",))}
