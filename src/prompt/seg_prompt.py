"""Prompt + output-format for multi-segment procedure segmentation.

Output contract the model must follow (parsed by src/rewards/seg_rewards.py):

    <think> ...reasoning... </think>
    <segments>
    <seg><name>Adjust and fold first outer box flap</name><span>2.60 to 15.30</span></seg>
    <seg><name>Align inner box flaps</name><span>15.30 to 19.80</span></seg>
    </segments>
"""

QUESTION_TEMPLATE_SEG = (
    "Watch the video and segment it into the ordered sequence of distinct "
    "sub-tasks the operator performs. For each sub-task give a short name and "
    "its start and end time in seconds (two decimals).\n"
    "Reason step by step inside <think> </think>. Then output every segment as\n"
    "<seg><name>NAME</name><span>START to END</span></seg>\n"
    "wrapped in a single <segments> </segments> block, in chronological order "
    "with no gaps or overlaps between adjacent segments."
)


def render_target(segments, reasons=None):
    """Build the gold completion text for SFT cold-start.

    segments : list of (name, start, end)
    reasons  : optional list of boundary-reason strings (from dataset), folded
               into a single leading <think> block.
    """
    think = ""
    if reasons:
        joined = " ".join(r.strip() for r in reasons if r)
        think = "<think>\n" + joined + "\n</think>\n"
    lines = ["<segments>"]
    for name, s, e in segments:
        lines.append(
            "<seg><name>{}</name><span>{:.2f} to {:.2f}</span></seg>".format(
                name, float(s), float(e)
            )
        )
    lines.append("</segments>")
    return think + "\n".join(lines)
