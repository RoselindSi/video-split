"""The separation between the self/other experiment and V1, checked, not promised.

Static, so it runs anywhere and cannot be skipped for want of a GPU: parse
every file in `src/selfother` and fail on a V1 import, on a V1 output column
named outside the one file that scores baselines, or on a hardcoded path to
V1's checkpoint.
"""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "src" / "selfother"

V1_MODULES = ("src.rig.own_ctx", "src.rig.own_cnn", "src.rig.own_label",
              "src.rig.own_dump", "src.rig.own_census", "src.rig.own_gold",
              "src.rig.hand_ownership", "src.rig.owner_arm",
              "src.rig.zone_prior")
V1_NAMES = {"OwnHold", "own_ctx", "own_cnn", "own_label", "hand_ownership"}
V1_COLUMNS = {"p_owner_raw", "logit_owner", "ema_owner", "ownhold_pre_cap",
              "owner_count_pre_cap", "cap_triggered", "cap_demoted",
              "final_owner_post_cap", "reference_owner", "rule_owner"}
COLUMN_READERS = {"evaluate.py"}


def _files():
    return sorted(PKG.glob("*.py"))


def test_package_is_there():
    assert _files(), f"nothing to check under {PKG}"


def test_no_v1_module_is_imported():
    for f in _files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            else:
                continue
            for n in names:
                assert not any(n == m or n.startswith(m + ".")
                               for m in V1_MODULES), f"{f.name} imports {n}"
                assert n.rsplit(".", 1)[-1] not in V1_NAMES, \
                    f"{f.name} imports {n}"


def test_v1_columns_only_in_evaluate():
    for f in _files():
        if f.name in COLUMN_READERS:
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in V1_COLUMNS, \
                    f"{f.name} names V1 column {node.value!r}"


def test_v1_checkpoint_only_by_argument():
    for f in _files():
        assert "own_ctx_best" not in f.read_text(encoding="utf-8"), \
            f"{f.name} hardcodes the V1 checkpoint"
