"""The eight rules, enforced at runtime. A README cannot fail; this can.

Every check here exists because the project broke that rule once, and each of
those breaks was SILENT -- a leaked answer field trains a model that scores
beautifully, a coefficient that learned the wrong sign still produces a number,
a review rate reported alone looks like progress. None of them announce
themselves, so none of them can be left to discipline.

The rules live in `configs/auditor/ontology_v1_constitution.yaml` and are read,
never restated here. A checker carrying its own copy would drift from the
config the first time someone edited one of them.

Usage:
    from src.auditor.boundary.ontology_constitution import Constitution
    C = Constitution()
    C.check_disjoint(train_rids, eval_rids)
    C.check_features(feature_names)
    C.check_candidate_pool(cands)
    C.check_report(review_rate=..., true_boundary_loss_rate=..., ...)

    python -m src.auditor.boundary.ontology_constitution --self_test
"""
from __future__ import annotations

import os

CONFIG = "configs/auditor/ontology_v1_constitution.yaml"


class ConstitutionViolation(Exception):
    """Raised, never warned. Every rule here guards a silent failure."""


class Constitution:
    def __init__(self, path=CONFIG):
        import yaml
        if not os.path.exists(path):
            raise ConstitutionViolation(
                f"{path} not found. The rules are config rather than code so "
                f"that editing them is a visible act; running without them is "
                f"not a fallback.")
        with open(path, encoding="utf-8") as f:
            self.c = yaml.safe_load(f)
        if not self.c.get("frozen"):
            raise ConstitutionViolation(
                f"{path} is not marked frozen; every check below would be "
                f"against a moving target.")

    # --- 1 -----------------------------------------------------------------
    def tolerance_s(self):
        return float(self.c["boundary_definition"]["tolerance_s"])

    def check_tolerance(self, tol):
        want = self.tolerance_s()
        if abs(float(tol) - want) > 1e-9:
            raise ConstitutionViolation(
                f"tolerance {tol} != {want}. 0.5s was retired on 2026-08-19, "
                f"and a new measurement at the old tolerance is not comparable "
                f"to anything beside it -- which is exactly how it goes "
                f"unnoticed.")

    # --- 2 -----------------------------------------------------------------
    def check_candidate_pool(self, cands, manifest_path=None):
        p = self.c["candidate_pool"]
        n, nr = len(cands), len({c["recording_id"] for c in cands})
        if n != p["n_candidates"] or nr != p["n_recordings"]:
            raise ConstitutionViolation(
                f"candidate pool is {n} candidates over {nr} recordings, "
                f"frozen at {p['n_candidates']}/{p['n_recordings']}.\n"
                f"  Every arm must start from the same pool. A pool that "
                f"differs by even one candidate\n  turns a comparison between "
                f"methods into a comparison between populations.")
        if manifest_path and not manifest_path.endswith(
                os.path.basename(p["manifest"])):
            raise ConstitutionViolation(
                f"candidates came from {manifest_path}, not the frozen "
                f"{p['manifest']}")

    # --- 4 -----------------------------------------------------------------
    def check_disjoint(self, train_rids, eval_rids):
        over = set(train_rids) & set(eval_rids)
        cap = self.c["data_isolation"]["train_eval_recording_overlap_max"]
        if len(over) > cap:
            raise ConstitutionViolation(
                f"TRAIN INTERSECT EVAL = {len(over)}, max {cap}: "
                f"{sorted(over)[:5]}\n"
                f"  A model that trained on a recording it is about to score "
                f"carries information\n  the evaluation split was built to "
                f"exclude, and every downstream number\n  measures that "
                f"instead of the method.")

    def fitted_may_not_see(self):
        return list(self.c["data_isolation"]
                    ["eval_recordings_may_not_contribute_to"])

    # --- 5 -----------------------------------------------------------------
    def may_license(self, cls):
        return cls in self.c["morphology"]["may_license_action"]

    def check_licensing(self, cls):
        if not self.may_license(cls):
            allowed = self.c["morphology"]["may_license_action"]
            raise ConstitutionViolation(
                f"{cls} may not license an automatic action; only {allowed} "
                f"has the supervision behind it (101 external training "
                f"events against 25 and 17).")

    # --- 6 -----------------------------------------------------------------
    def check_abstention(self, cls, action):
        a = self.c["abstention"]
        if cls == "UNOBSERVABLE" and action != a["unobservable_action"]:
            raise ConstitutionViolation(
                f"UNOBSERVABLE routed to {action!r}, must be "
                f"{a['unobservable_action']!r}.\n"
                f"  'no boundary was seen' and 'it was seen that there is no "
                f"boundary' are different\n  claims. Only the second licenses "
                f"a suppression, and conflating them buys a\n  better-looking "
                f"review rate with an unmeasured risk.")

    # --- 7 -----------------------------------------------------------------
    def check_energy_signs(self, coeffs):
        """coeffs: {name: float}. Every ontology coefficient must be >= 0.

        The sign is architecture, not a parameter. Left free, an optimiser can
        and will learn that NO_TRANSITION supports a boundary if that fits the
        training set, and the resulting model is indistinguishable from a
        working one until it meets data where the ontology matters."""
        e = self.c["energy_signs"]
        if not e["non_negative_coefficients"]:
            return
        bad = {k: v for k, v in coeffs.items() if v < 0}
        if bad:
            raise ConstitutionViolation(
                f"ontology coefficients went negative: {bad}\n"
                f"  Use softplus on the raw parameter. A coefficient free to "
                f"change sign is a model\n  free to invert the ontology, and "
                f"nothing downstream would notice.")

    def sign_of(self, name):
        e = self.c["energy_signs"]
        if name in e["supports_boundary"]:
            return +1
        if name in e["suppresses_boundary"]:
            return -1
        raise ConstitutionViolation(
            f"{name} has no declared sign. Adding an evidence term is a "
            f"deliberate act: declare it in energy_signs first.")

    # --- 8 -----------------------------------------------------------------
    def check_features(self, names):
        f = self.c["feature_leakage"]
        bad = [n for n in names if n in f["forbidden_runtime_features"]]
        if bad:
            raise ConstitutionViolation(
                f"answer fields used as runtime features: {bad}\n"
                f"  These are what the model is being asked to predict. A "
                f"scorer that reads them\n  scores its own target, and the "
                f"number it produces is not about the video.")

    # --- reporting ---------------------------------------------------------
    def check_report(self, **kw):
        need = self.c["reporting"]["review_rate_requires"]
        if "review_rate" in kw:
            missing = [k for k in need if kw.get(k) is None]
            if missing:
                raise ConstitutionViolation(
                    f"review_rate reported without {missing}.\n"
                    f"  A review rate alone is satisfiable by automating "
                    f"everything in either\n  direction: release all "
                    f"candidates or suppress them all, and it reads as "
                    f"progress.")

    # --- actions -----------------------------------------------------------
    def check_action_enabled(self, action):
        a = self.c["actions"].get(action)
        if a is None:
            raise ConstitutionViolation(
                f"{action} is not one of the three declared actions: "
                f"{list(self.c['actions'])}")
        if not a.get("enabled"):
            raise ConstitutionViolation(
                f"{action} is disabled. Risk: {a['risk']}.\n"
                f"  Enabling it needs an independent risk certificate, not a "
                f"config edit.")

    # --- ablation ----------------------------------------------------------
    def check_level(self, level):
        order = self.c["ablation"]["order"]
        done = self.c["ablation"]["implemented_through"]
        if level not in order:
            raise ConstitutionViolation(f"{level} is not in {order}")
        if order.index(level) > order.index(done):
            raise ConstitutionViolation(
                f"{level} comes after {done}, which is the last level with a "
                f"measured incremental gain.\n  {self.c['ablation']['gate']}")

    # --- oracle ------------------------------------------------------------
    def check_oracle_use(self, purpose):
        o = self.c["oracle"]
        bad = {"train": o["may_train"], "select_thresholds":
               o["may_select_thresholds"],
               "select_architecture": o["may_select_architecture"]}
        if purpose in bad and not bad[purpose]:
            raise ConstitutionViolation(
                f"the oracle may not be used to {purpose}; its use is "
                f"{o['use']!r}.\n  An upper bound that selects anything has "
                f"become a development set, and stops\n  bounding anything.")


def _self_test():
    C = Constitution()
    ok = 0

    def must_raise(fn, frag):
        nonlocal ok
        try:
            fn()
        except ConstitutionViolation as e:
            assert frag in str(e), (frag, str(e)[:120])
            ok += 1
            print(f"  raises ({frag}): {str(e).splitlines()[0][:64]}")
            return
        raise AssertionError(f"did not raise for {frag}")

    C.check_tolerance(1.0)
    must_raise(lambda: C.check_tolerance(0.5), "retired")

    C.check_disjoint({"a", "b"}, {"c"})
    must_raise(lambda: C.check_disjoint({"a"}, {"a"}), "TRAIN INTERSECT EVAL")

    C.check_features(["detector_score", "morphology_evidence"])
    must_raise(lambda: C.check_features(["detector_score",
                                         "instance_relation"]),
               "answer fields")
    must_raise(lambda: C.check_features(["is_true_boundary"]), "answer fields")

    C.check_energy_signs({"eta_point": 0.4, "eta_no_transition": 1.2})
    must_raise(lambda: C.check_energy_signs({"eta_no_transition": -0.3}),
               "went negative")

    assert C.sign_of("POINT_TRANSITION") == +1
    assert C.sign_of("NO_TRANSITION") == -1
    must_raise(lambda: C.sign_of("VIBES"), "no declared sign")

    C.check_licensing("NO_TRANSITION")
    must_raise(lambda: C.check_licensing("UNOBSERVABLE"), "may not license")
    must_raise(lambda: C.check_licensing("INTERVAL_TRANSITION"),
               "may not license")

    C.check_abstention("UNOBSERVABLE", "REVIEW")
    must_raise(lambda: C.check_abstention("UNOBSERVABLE", "SUPPRESS"),
               "different")

    C.check_report(review_rate=0.1, true_boundary_loss_rate=0.4,
                   false_boundaries_released=100)
    must_raise(lambda: C.check_report(review_rate=0.1), "without")

    must_raise(lambda: C.check_action_enabled("SUPPRESS_MODEL_CANDIDATE"),
               "disabled")
    must_raise(lambda: C.check_action_enabled("DELETE_EXISTING_ANNOTATION"),
               "disabled")

    C.check_level("C_ontology_rerank")
    must_raise(lambda: C.check_level("D_segment_dp"), "comes after")
    must_raise(lambda: C.check_level("E_structured_loss"), "comes after")

    must_raise(lambda: C.check_oracle_use("train"), "may not be used")
    must_raise(lambda: C.check_oracle_use("select_thresholds"),
               "may not be used")

    pool = [{"recording_id": f"r{i % 36}"} for i in range(3707)]
    C.check_candidate_pool(pool)
    must_raise(lambda: C.check_candidate_pool(pool[:100]), "frozen at")

    print(f"\n  {ok} violations raised, every permitted call passed.")
    print(f"  Each of these was a real failure in this project before it was "
          f"a check.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
