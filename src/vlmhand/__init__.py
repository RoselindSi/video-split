"""Ask Qwen3.8-27B, zero-shot, which detected hands are the camera wearer's.

ONE QUESTION PER FRAME, NOT PER HAND. Every detected hand in the frame gets a
numbered box and the model names the ones that belong to the wearer. That is
the question as the user phrased it -- which two hands are the operator's --
and it lets the model use what a crop cannot: where the arms go, whose torso
they join, how many hands the frame holds. The one fact the prompt states is
anatomical (a person has at most two hands); no geometric rule of thumb is
given, because the exit-height rule is exactly the kind of prior this project
has already watched fail.

TWO ENVIRONMENTS, SO FOUR STEPS. Rendering needs the rig code (`venv_rig`);
the model needs transformers 5.11 (`/workspace/lvs/.venv`). They meet only
through files:
    render.py   frames with numbered boxes + manifest.jsonl     (venv_rig)
    judge.py    manifest -> answers.jsonl                       (lvs venv)
    score.py    answers vs the package labels, beside V1 and the arms
    video.py    answers -> an mp4 boxing only the wearer's hands

BF16, NOT THE FP8 COPY. The FP8 weights need a patched kernel and a config
workaround to load; the unquantised model fits one 98 GB card without either.
Greedy decoding, thinking off, so an answer can be reproduced.
"""
