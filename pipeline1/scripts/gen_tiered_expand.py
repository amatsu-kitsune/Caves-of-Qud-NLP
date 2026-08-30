"""
Expand the tiered evaluation/training set from 100 to 300 texts (+50 per tier,
qt_101..qt_300), APPENDING to generated/qud_tiered_annotated.jsonl and
generated/qud_tiered_texts.txt.

Rules (aligned with the relation-only-synonymy design):
  - entities ALWAYS by canonical label (role individuals via their natural surface
    "boss-tier fighter" / "mid-tier merchant"); synonyms ONLY on relations
  - every triple HARD-VALIDATED against gold_triples.jsonl (asserted ABox facts
    only -> domain/range/disjoint compliance for free); abort otherwise
  - subject+object of every triple co-occur in the SAME sentence (checked with
    infer_pipeline.sent_split, which protects quest names containing .!?)
  - every triple must be span-markable (build_verifier_from_stories.mark_pair /
    mark_literal) so the verifier builder gets 0 object misses by construction
  - long_distance = ONE sentence with >=15 filler tokens between the mentions

Tier recipes:
  explicit      1 triple, plain template (first phrasing of OBJ_SYN pool)
  implicit      1-2 triples, idiomatic relation paraphrase, canonical entities
  long_distance 1 triple, long single sentence, entity-free filler clause
  nested        4-7 triples clustered on a hub entity, one sentence

Deterministic (seeded). Re-running when qt_101+ already exist aborts (delete the
appended block first, or use --force to regenerate exactly the same ids).

Usage: python gen_tiered_expand.py [--seed 11] [--per-tier 50]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import generate_stories as gs
import build_verifier_from_stories as bvs
import infer_pipeline as ip

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
JSONL = GEN / "qud_tiered_annotated.jsonl"
TXT = GEN / "qud_tiered_texts.txt"
NL = "\r\n"

ROLE_SURFACE = {"Fighter_Boss": "boss-tier fighter", "Fighter_NonBoss": "non-boss fighter",
                "Trader_high": "high-tier merchant", "Trader_mid": "mid-tier merchant",
                "Trader_low": "low-tier merchant"}

IMPL_SYN = {
    "memberOf": ["{s} stands with {o}", "{s} keeps faith with {o}", "{s} runs with {o}"],
    "hasMember": ["{o} is counted among {s}", "{s} claims {o} as one of its own"],
    "isHabitantOf": ["{s} calls {o} home", "{s} can be found in {o}", "{s} is holed up in {o}"],
    "hasInhabitant": ["{s} is where you'll find {o}"],
    "hates": ["{s} can't abide {o}", "{s} bears an old grudge against {o}", "{s} will never forgive {o}"],
    "likes": ["{s} holds {o} in fond regard", "{s} looks kindly on {o}"],
    "likedBy": ["{s} is held dear by {o}", "{s} enjoys the special favor of {o}"],
    "dislikes": ["{s} wants {o} gone from the world"],
    "dislikedBy": ["{s} is marked for death by {o}"],
    "giveQuest": ["{s} will saddle you with {o}", "{s} throws {o} your way", "{s} sets the player on {o}"],
    "isGivenBy": ["{s} is handed out by {o}", "{s} comes down from {o}"],
    "requiresCompleting": ["{s} stays sealed until {o} is done", "{s} must wait on {o}",
                           "wrap up {o} before you so much as touch {s}"],
    "rewardsWith": ["seeing {s} through pays out with {o}", "crack {s} and {o} is your prize"],
    "isRewardFor": ["{s} is the payoff for seeing {o} through"],
    "involvesFinding": ["{s} sends you hunting for {o}", "{s} has you track down {o}"],
    "isSearchTargetIn": ["{s} is the quarry you must track down in {o}"],
    "involvesFightWith": ["{s} pits you against {o}", "{s} throws you against {o}"],
    "foughtIn": ["{s} clashed with heroes in the thick of {o}"],
    "hostsQuest": ["deep in {s}, the trial {o} plays out"],
    "takesPlaceIn": ["{s} unfolds entirely within {o}"],
    "spawnsIn": ["{s} first opens their eyes in {o}"],
    "hasRace": ["{s} is {o} in blood and bone"],
    "hasRole": ["{s} is known around the settlements as a {o}"],
    "containsSettlement": ["{s} keeps {o} somewhere in its depths"],
    "locatedInPlace": ["{s} nests inside {o}"],
}

FILLERS = [
    "known to every water-baron and caravan hand who has ever hauled goods across the blistered stretches of the deep desert",
    "spoken of in the low careful voice that people reserve for things they respect and do not entirely understand",
    "remembered in a dozen contradictory stories that agree on almost nothing except the broad shape of what happened",
    "familiar to anyone who has spent even a single season trading news and water along the roads of the wastes",
    "regarded by the cautious as a matter best left alone and by the curious as a puzzle worth a lifetime",
    "wrapped in enough rumor and half-truth that separating the fact from the legend has become a pastime of its own",
    "the subject of more idle campfire arguments among wanderers than nearly any other matter one could care to name",
    "mentioned in the old histories only in passing and always with a caution that borders on outright superstition",
    "carrying far more weight among the survivors of the salt flats than any newcomer could reasonably be expected to guess",
    "after more years of dust and rumor and slow forgetting than any of the village elders can honestly count",
    "described by the chroniclers of the wastes at tiresome length in accounts that almost nobody alive has bothered to read",
    "trusted by some and feared by others yet impossible to leave out of any honest telling of these lands",
]


def load_gold():
    gold = bvs.load_jsonl(GEN / "gold_triples.jsonl")
    label_of = {}
    for t in gold:
        label_of[t["subject"]] = t["subject_label"]
        if t["kind"] == "object_property":
            label_of[t["object"]] = t["object_label"]
    label_of.update(ROLE_SURFACE)
    return gold, label_of


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def subj_surface(short: str, label_of) -> str:
    """Natural sentence-initial surface: 'The Barathrumites', 'The alchemist'."""
    lab = label_of[short]
    if short in ("Barathrumites", "Girsh", "Mechanimists", "Putus_Templar",
                 "Villagers_of_Joppa", "Seraphic_Covenant", "alchemist"):
        return "The " + lab
    if lab[0].islower():
        return "The " + lab
    return lab


def tail(pattern: str, o_label: str) -> str:
    """OBJ_SYN patterns all start '{s} '; strip it to get a verb tail."""
    assert pattern.startswith("{s} "), pattern
    return pattern[4:].replace("{o}", o_label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--per-tier", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold, label_of = load_gold()
    all_labels = sorted(set(label_of.values()), key=len, reverse=True)
    obj_gold = [t for t in gold if t["kind"] == "object_property"]
    dt_gold = [t for t in gold if t["kind"] == "datatype_property" and t["predicate"] in gs.DT_SYN]
    gold_obj_set = {(t["subject"], t["predicate"], t["object"]) for t in obj_gold}
    gold_dt_set = {(t["subject"], t["predicate"], t["object"]) for t in dt_gold}

    existing = [json.loads(l) for l in JSONL.open(encoding="utf-8") if l.strip()]
    existing_ids = {e["id"] for e in existing}
    seen_texts = {e["text"] for e in existing}
    if any(int(i.split("_")[1]) > 100 for i in existing_ids) and not args.force:
        raise SystemExit("qt_101+ already present - remove them first or pass --force")

    by_pred = defaultdict(list)
    for t in obj_gold:
        by_pred[t["predicate"]].append(t)
    preds = sorted(by_pred)
    by_subj = defaultdict(list)
    for t in obj_gold:
        by_subj[t["subject"]].append(t)
    dt_by_subj = defaultdict(list)
    for t in dt_gold:
        dt_by_subj[t["subject"]].append(t)

    def hit(s, lab):
        return re.search(r"(?<!\w)" + re.escape(lab) + r"(?!\w)", s, re.I)

    def validate(text, triples, tier):
        assert text.isascii(), f"non-ascii: {text!r}"
        sents = ip.sent_split(text)
        for tr in triples:
            key = (tr["subject"], tr["predicate"], tr["object"])
            if tr["kind"] == "object_property":
                assert key in gold_obj_set, f"not an ABox fact: {key}"
                ls, lo = label_of[tr["subject"]], label_of[tr["object"]]
                co = [s for s in sents if hit(s, ls) and hit(s, lo)]
                assert co, f"{tier}: no same-sentence co-occurrence {key} :: {text}"
                assert bvs.mark_pair(text, ls, lo, all_labels), f"unmarkable {key} :: {text}"
                if tier == "long_distance":
                    s = co[0]
                    sp1 = bvs.valid_spans(s, ls, all_labels)
                    sp2 = bvs.valid_spans(s, lo, all_labels)
                    d = max((len(s[min(b1, b2):max(a1, a2)].split())
                             for a1, b1 in sp1 for a2, b2 in sp2
                             if b1 <= a2 or b2 <= a1), default=0)
                    assert d >= 15, f"distance {d} < 15 :: {text}"
            else:
                assert key in gold_dt_set, f"not a gold datatype fact: {key}"
                assert bvs.mark_literal(text, label_of[tr["subject"]], tr["predicate"],
                                        str(tr["object"]), all_labels), f"unmarkable dt {key} :: {text}"

    out = []
    counters = defaultdict(int)

    def emit(tier, text, triples):
        assert text not in seen_texts, f"duplicate text: {text}"
        seen_texts.add(text)
        counters[tier] += 1
        out.append({"tier": tier, "text": text, "triples": [dict(t) for t in triples]})

    i = 0
    while counters["explicit"] < args.per_tier:
        p = preds[i % len(preds)]; i += 1
        tr = rng.choice(by_pred[p])
        ls, lo = label_of[tr["subject"]], label_of[tr["object"]]
        text = cap(gs.OBJ_SYN[p][0].replace("{s}", subj_surface(tr["subject"], label_of))
                   .replace("{o}", lo)) + "."
        if text in seen_texts:
            continue
        validate(text, [tr], "explicit")
        emit("explicit", text, [tr])

    two_subjects = [s for s, ts in by_subj.items() if len(ts) >= 2]
    i = 0
    while counters["implicit"] < args.per_tier:
        i += 1
        if i % 2 == 0 and two_subjects:
            s = rng.choice(two_subjects)
            t1, t2 = rng.sample(by_subj[s], 2)
            tl1 = tail(rng.choice(gs.OBJ_SYN[t1["predicate"]][1:] or gs.OBJ_SYN[t1["predicate"]]),
                       label_of[t1["object"]])
            tl2 = tail(rng.choice(gs.OBJ_SYN[t2["predicate"]][1:] or gs.OBJ_SYN[t2["predicate"]]),
                       label_of[t2["object"]])
            text = f"{cap(subj_surface(s, label_of))} {tl1} and {tl2}."
            triples = [t1, t2]
        else:
            p = preds[i % len(preds)]
            tr = rng.choice(by_pred[p])
            pool = IMPL_SYN.get(p) or gs.OBJ_SYN[p][1:] or gs.OBJ_SYN[p]
            patt = rng.choice(pool)
            text = cap(patt.replace("{s}", subj_surface(tr["subject"], label_of))
                       .replace("{o}", label_of[tr["object"]])) + "."
            triples = [tr]
        if text in seen_texts:
            continue
        validate(text, triples, "implicit")
        emit("implicit", text, triples)

    i = 0
    while counters["long_distance"] < args.per_tier:
        p = preds[i % len(preds)]; i += 1
        tr = rng.choice(by_pred[p])
        fill = rng.choice(FILLERS)
        text = (f"{cap(subj_surface(tr['subject'], label_of))}, {fill}, "
                f"{tail(rng.choice(gs.OBJ_SYN[p]), label_of[tr['object']])}.")
        if text in seen_texts:
            continue
        validate(text, [tr], "long_distance")
        emit("long_distance", text, [tr])

    hubs = [s for s, ts in by_subj.items() if len(ts) >= 3]
    i = 0
    while counters["nested"] < args.per_tier:
        hub = hubs[i % len(hubs)]; i += 1
        facts = by_subj[hub][:]
        rng.shuffle(facts)
        k = rng.randint(3, min(5, len(facts)))
        triples = facts[:k]
        tails = [tail(rng.choice(gs.OBJ_SYN[t["predicate"]]), label_of[t["object"]])
                 for t in triples]
        dts = dt_by_subj.get(hub, [])
        if dts and rng.random() < 0.7:
            for dtr in rng.sample(dts, min(2, len(dts))):
                patt = rng.choice(gs.DT_SYN[dtr["predicate"]])
                tails.append(patt[4:].replace("{v}", str(dtr["object"])))
                triples = triples + [dtr]
        rel = ("who" if {"NPC", "Mutant", "Truekin", "Robot", "PlayerCharacter"}
               & set(triples[0]["subject_types"]) else "which")
        body = ", ".join(tails[:-1]) + " and " + tails[-1]
        text = f"{cap(subj_surface(hub, label_of))}, {rel} {body}."
        if text in seen_texts:
            continue
        validate(text, triples, "nested")
        emit("nested", text, triples)

    order = {"explicit": 0, "implicit": 1, "long_distance": 2, "nested": 3}
    out.sort(key=lambda e: order[e["tier"]])
    next_id = 101
    jsonl_lines, txt_lines = [], []
    for e in out:
        qid = f"qt_{next_id:03d}"; next_id += 1
        rec = {"id": qid, "tier": e["tier"], "text": e["text"],
               "n_triples": len(e["triples"]), "triples": e["triples"]}
        jsonl_lines.append(json.dumps(rec, ensure_ascii=False))
        txt_lines.append(f"{qid}\t{e['tier']}\t{e['text']}")

    with JSONL.open("a", encoding="ascii", newline="") as h:
        for l in jsonl_lines:
            h.write(l + NL)
    with TXT.open("a", encoding="ascii", newline="") as h:
        for l in txt_lines:
            h.write(l + NL)

    n_trip = sum(len(e["triples"]) for e in out)
    pred_cov = sorted({t["predicate"] for e in out for t in e["triples"]})
    print(json.dumps({
        "appended_texts": len(out), "per_tier": dict(counters),
        "appended_triples": n_trip,
        "distinct_predicates": len(pred_cov), "predicates": pred_cov,
    }, indent=2))


if __name__ == "__main__":
    main()
