"""
Pipeline 1 - corpus generator: 300 short English stories (~10 lines) grounded on the
Caves of Qud ABox, with SYNONYM variation on the relation verbs (per Block 08 spec).

  - 300 stories total: 250 train / 50 val (story-level split).
  - The last 50 (the val set) are "complex": longer + compound multi-fact lines.
  - Entity NAMES are verbatim (so GLiNER grounding + verifier marker injection work);
    only the RELATION VERBS vary via synonym banks.

Outputs:
  data-input/train/story_001.txt ... story_250.txt
  data-input/val/story_251.txt   ... story_300.txt
  generated/story_corpus.jsonl   (per-line gold: text + the triple(s) it expresses)
  generated/story_corpus_stats.json

Usage: python generate_stories.py [--seed 7] [--n 300] [--val 50]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
DATA = BASE / "data-input"

SURFACE_OVERRIDES = {
    "Fighter_Boss": "boss-tier Fighter", "Fighter_NonBoss": "non-boss Fighter",
    "Trader_high": "high-tier merchant", "Trader_mid": "mid-tier merchant",
    "Trader_low": "low-tier merchant",
}

OBJ_SYN: dict[str, list[str]] = {
    "containsSettlement": ["{s} contains {o}", "{s} encloses {o}", "{s} holds {o} within it"],
    "dislikedBy": ["{s} is disliked by {o}", "{s} is scorned by {o}", "{s} is held in low regard by {o}"],
    "dislikes": ["{s} dislikes {o}", "{s} scorns {o}", "{s} looks down on {o}"],
    "foughtIn": ["{s} is fought in {o}", "{s} stands as an enemy in {o}", "{s} is battled during {o}"],
    "giveQuest": ["{s} gives the player the quest {o}", "{s} offers the quest {o}", "{s} hands out {o}",
                  "{s} entrusts the player with {o}", "{s} assigns the quest {o}"],
    "hasIndividual": ["{s} counts {o} among its members", "{s} includes {o}", "{s} numbers {o} among its kind"],
    "hasInhabitant": ["{s} is home to {o}", "{s} houses {o}", "{s} shelters {o}", "{s} is inhabited by {o}"],
    "hasMember": ["{s} counts {o} as a member", "{s} includes {o} in its ranks", "{s} numbers {o} among its members"],
    "hasRace": ["{s} is of the {o} race", "{s} belongs to the {o} race", "{s} is a {o} by lineage"],
    "hasRole": ["{s} serves as a {o}", "{s} acts as a {o}", "{s} works as a {o}", "{s} fills the role of a {o}"],
    "hates": ["{s} hates {o}", "{s} is hostile to {o}", "{s} is at war with {o}", "{s} despises {o}"],
    "hostsQuest": ["{s} hosts the quest {o}", "{s} stages {o}", "{s} is the setting of {o}", "{s} harbors {o}"],
    "involvesFightWith": ["{s} pits the player against {o}", "{s} requires fighting {o}", "{s} calls for battle with {o}"],
    "involvesFinding": ["{s} requires finding {o}", "{s} asks the player to find {o}",
                        "{s} sends the player to locate {o}", "{s} has the player seek {o}"],
    "isGivenBy": ["{s} is given by {o}", "{s} is offered by {o}", "{s} is assigned by {o}", "{s} comes from {o}"],
    "isHabitantOf": ["{s} lives in {o}", "{s} dwells in {o}", "{s} resides in {o}", "{s} makes a home in {o}"],
    "isRewardFor": ["{s} is the reward for {o}", "{s} is granted by {o}", "{s} is awarded for {o}"],
    "isRoleOf": ["{s} is the role of {o}", "{s} is the part played by {o}", "{s} is held by {o}"],
    "isSearchTargetIn": ["{s} is sought in {o}", "{s} is the target of {o}", "{s} must be found in {o}"],
    "likedBy": ["{s} is liked by {o}", "{s} is favored by {o}", "{s} enjoys the favor of {o}"],
    "likes": ["{s} likes {o}", "{s} favors {o}", "{s} is fond of {o}", "{s} holds {o} in esteem"],
    "locatedInPlace": ["{s} is located in {o}", "{s} lies within {o}", "{s} sits inside {o}"],
    "memberOf": ["{s} is a member of {o}", "{s} belongs to {o}", "{s} is part of {o}"],
    "requiresCompleting": ["{s} requires completing {o}", "{s} comes after {o}", "{s} is unlocked by {o}", "{s} follows {o}"],
    "rewardsWith": ["{s} rewards the player with {o}", "{s} grants {o}", "{s} yields {o}", "{s} awards {o}"],
    "sellsItem": ["{s} sells {o}", "{s} offers {o} for sale", "{s} trades in {o}", "{s} stocks {o}"],
    "spawnsIn": ["{s} starts in {o}", "{s} spawns in {o}", "{s} first appears in {o}"],
    "takesPlaceIn": ["{s} takes place in {o}", "{s} unfolds in {o}", "{s} is set in {o}", "{s} occurs in {o}"],
}
DT_SYN: dict[str, list[str]] = {
    "HP": ["{s} has {v} HP", "{s} has {v} hitpoints", "{s} carries {v} HP"],
    "level": ["{s} is level {v}", "{s} stands at level {v}", "{s} has reached level {v}"],
    "strata": ["{s} lies at strata {v}", "{s} sits at strata {v}", "{s} is found at strata {v}"],
    "Value": ["{s} is worth {v} drams", "{s} is valued at {v} drams", "{s} sells for {v} drams"],
    "Weight": ["{s} weighs {v}", "{s} has a weight of {v}"],
}
DT_KEEP = set(DT_SYN)


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--lines", type=int, default=10)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = load_jsonl(GEN / "gold_triples.jsonl")
    label_of: dict[str, str] = {}
    for t in gold:
        label_of[t["subject"]] = t["subject_label"]
        if t["kind"] == "object_property":
            label_of[t["object"]] = t["object_label"]
    label_of.update(SURFACE_OVERRIDES)

    facts: dict[str, list[dict]] = defaultdict(list)
    neighbors: dict[str, list[str]] = defaultdict(list)
    for t in gold:
        s, p = t["subject"], t["predicate"]
        if t["kind"] == "object_property" and p in OBJ_SYN:
            facts[s].append({"predicate": p, "object": t["object"], "kind": "object"})
            neighbors[s].append(t["object"])
        elif t["kind"] == "datatype_property" and p in DT_SYN:
            facts[s].append({"predicate": p, "value": t["object"], "kind": "datatype"})
    seeds = sorted(facts)

    def render(fact: dict, subj: str) -> tuple[str, dict]:
        s = label_of[subj]
        if fact["kind"] == "object":
            o = label_of[fact["object"]]
            line = rng.choice(OBJ_SYN[fact["predicate"]]).format(s=s, o=o)
            triple = {"subject": subj, "predicate": fact["predicate"], "object": fact["object"]}
        else:
            line = rng.choice(DT_SYN[fact["predicate"]]).format(s=s, v=fact["value"])
            triple = {"subject": subj, "predicate": fact["predicate"],
                      "object": "LiteralValue", "literal": fact["value"]}
        return line, triple

    def collect_pool(seed: str, want: int) -> list[tuple[str, dict]]:
        """BFS over the ABox from `seed`, gathering distinct facts as (subject, fact)."""
        pool: list[tuple[str, dict]] = []
        seen_keys: set = set()
        frontier = [seed]
        visited = set()
        while frontier and len(pool) < want:
            cur = frontier.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            cur_facts = facts.get(cur, [])[:]
            rng.shuffle(cur_facts)
            for f in cur_facts:
                key = (cur, f["predicate"], f.get("object"), f.get("value"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                pool.append((cur, f))
                if f["kind"] == "object" and f["object"] in facts:
                    frontier.append(f["object"])
                if len(pool) >= want:
                    break
        return pool

    stories = []
    corpus = []
    n_complex = args.val
    for i in range(args.n):
        complex_story = i >= (args.n - n_complex)
        want = (rng.randint(13, 16) if complex_story else args.lines)
        seed = seeds[i % len(seeds)]
        pool = collect_pool(seed, want + 4)
        rng.shuffle(pool)
        pool = pool[:want]
        if not pool:
            continue

        lines: list[str] = []
        line_ann: list[dict] = []
        j = 0
        while j < len(pool):
            subj, fact = pool[j]
            if (complex_story and j + 1 < len(pool) and pool[j + 1][0] == subj
                    and rng.random() < 0.45):
                l1, t1 = render(fact, subj)
                f2 = pool[j + 1][1]
                if f2["kind"] == "object":
                    o2 = label_of[f2["object"]]
                    verb2 = rng.choice(OBJ_SYN[f2["predicate"]]).format(s="who", o=o2)
                    t2 = {"subject": subj, "predicate": f2["predicate"], "object": f2["object"]}
                else:
                    verb2 = rng.choice(DT_SYN[f2["predicate"]]).format(s="who", v=f2["value"])
                    t2 = {"subject": subj, "predicate": f2["predicate"],
                          "object": "LiteralValue", "literal": f2["value"]}
                verb2 = verb2[4:].strip() if verb2.lower().startswith("who ") else verb2
                line = f"{l1}, and also {verb2}."
                lines.append(line)
                line_ann.append({"text": line, "triples": [t1, t2]})
                j += 2
            else:
                line, triple = render(fact, subj)
                line = line + "."
                lines.append(line)
                line_ann.append({"text": line, "triples": [triple]})
                j += 1

        story_id = f"story_{i + 1:03d}"
        split = "train" if i < (args.n - args.val) else "val"
        text = "\n".join(lines)
        stories.append((story_id, split, text))
        corpus.append({"story_id": story_id, "split": split, "complex": complex_story,
                       "seed": seed, "text": text, "lines": line_ann})

    for sub in ("train", "val"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
        for f in (DATA / sub).glob("story_*.txt"):
            f.unlink()
    for story_id, split, text in stories:
        (DATA / split / f"{story_id}.txt").write_text(text + "\n", encoding="utf-8")

    with (GEN / "story_corpus.jsonl").open("w", encoding="utf-8") as h:
        for row in corpus:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_lines = sum(len(c["lines"]) for c in corpus)
    n_triples = sum(len(l["triples"]) for c in corpus for l in c["lines"])
    stats = {
        "stories": len(corpus),
        "split": {"train": sum(c["split"] == "train" for c in corpus),
                  "val": sum(c["split"] == "val" for c in corpus)},
        "complex_stories": sum(c["complex"] for c in corpus),
        "total_lines": n_lines,
        "avg_lines_per_story": round(n_lines / len(corpus), 1),
        "total_triple_mentions": n_triples,
    }
    (GEN / "story_corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
