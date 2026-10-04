#!/usr/bin/env python
"""Fit the confidence calibration on the served pipeline, with the signal computed by the engine itself.

The signal is `AcisEngine.confidence_signal` — the function the API calls — for the served #1 of each query:

* `statement_like` — the final P0 pipeline artifact (APPS dev, out of fold): its served top-10 and route per query.
  Each fold's map is fitted on the other four and scored on it (held-out reliability); the shipped map uses all.
* `generic` — the APPS dev statements the router sends generic (same artifact) **and** CodeSearchNet-Python's
  human-written queries ranked live by this pipeline (`_rank_one`, served routing). Five folds by query-id hash give
  held-out reliability per source; the shipped map uses all. CosQA, used by the earlier calibration, is not in this
  fit: its corpus has no vectors under the second encoder.

The bands are fixed (`high` ≥ 0.6, `medium` ≥ 0.3). A report, never a ranking input. Dev / REG data only; the held-out
APPS labels are never read. Writes `artifacts/confidence/calibration.json` and one `dev` ledger row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from acis.appsdata import apps
from acis.core.config import load_frozen_config
from acis.core.hashing import hash_obj
from acis.core.paths import acis_root
from acis.embed.factory import build_encoder
from acis.engine import AcisEngine
from acis.eval import ledger
from acis.rank.confidence import band, isotonic, lookup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg_fusion import load_task  # noqa: E402 — the REG loader the fusion tuning used

OUT = "artifacts/confidence/calibration.json"
FOLDS = 5


def reliability(probs: list[float], hits: list[int]) -> dict[str, Any]:
    ok = [(p, h) for p, h in zip(probs, hits, strict=True) if p == p]
    out: dict[str, Any] = {
        "n": len(ok),
        "brier": float(np.mean([(p - h) ** 2 for p, h in ok])) if ok else None,
        "base_rate": float(np.mean([h for _, h in ok])) if ok else None,
    }
    for level in ("high", "medium", "low"):
        members = [h for p, h in ok if band(p) == level]
        out[level] = {"n": len(members), "top1_relevant_rate": float(np.mean(members)) if members else None}
    return out


def fold_of(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % FOLDS


def cross_fit(points: list[tuple[str, float, int, int]]) -> tuple[list[float], list[float], dict[str, list[float]]]:
    """points: (source, z, hit, fold). Returns the full-data map and the held-out probability per point, by source."""
    held: dict[str, list[float]] = {}
    held_hits: dict[str, list[int]] = {}
    for fold in sorted({p[3] for p in points}):
        train = [p for p in points if p[3] != fold]
        knots, values = isotonic([p[1] for p in train], [p[2] for p in train])
        for source, z, hit, f in points:
            if f == fold:
                held.setdefault(source, []).append(lookup(knots, values, z))
                held_hits.setdefault(source, []).append(hit)
    knots, values = isotonic([p[1] for p in points], [p[2] for p in points])
    return knots, values, {s: [held[s], held_hits[s]] for s in held}  # type: ignore[misc]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--artifact", required=True, help="per_query.jsonl of the final P0 pipeline run")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)
    root = acis_root()
    config = load_frozen_config(args.config)
    engine = AcisEngine.from_config(config, encoder=build_encoder(config))
    apps_data = engine.snapshot_data(engine.build_snapshot(apps.load_corpus(), source="calibration:apps"))
    queries = apps.load_queries()
    records = [json.loads(x) for x in (root / args.artifact).read_text("utf-8").splitlines() if x]

    statement: list[tuple[str, float, int, int]] = []
    generic: list[tuple[str, float, int, int]] = []
    for r in records:
        query, _ = engine.normalise_query(queries[r["query_id"]])
        route = r["route"]["route"]
        z = engine.confidence_signal(apps_data, query, r["top10_full"], route=route)
        point = ("apps", z, int(r["rank_full"] == 1), int(r["fold"]))
        (statement if route == "statement_like" else generic).append(point)
    print(f"apps: {len(statement)} statement_like, {len(generic)} generic", flush=True)

    corpus, splits = load_task("csn-python")
    csn = engine.snapshot_data(engine.build_snapshot(corpus, source="calibration:csn-python"))
    csn_points = 0
    for row in splits["test"]:
        query, _ = engine.normalise_query(row["text"])
        route = str(engine.route(query))
        served = [d for d, _ in engine._rank_one(csn, row["text"], top_k=10, strict=False, route=route)]
        z = engine.confidence_signal(csn, query, served, route=route)
        point = ("csn-python", z, int(served[0] in row["relevant"]), fold_of(row["id"]))
        (statement if route == "statement_like" else generic).append(point)
        csn_points += 1
    print(f"csn-python: {csn_points} queries ranked live", flush=True)

    report: dict[str, Any] = {"artifact": args.artifact, "signal": "mean of the two encoders' z", "routes": {}}
    for route, points in (("statement_like", statement), ("generic", generic)):
        points = [p for p in points if p[1] == p[1]]
        knots, values, held = cross_fit(points)
        sources = sorted({p[0] for p in points})
        report["routes"][route] = {
            "knots": knots,
            "values": values,
            "n": len(points),
            "fitted_on": f"{route}: "
            + ", ".join(f"{s} ({sum(p[0] == s for p in points)})" for s in sources)
            + " — answerable queries only (a query about something the corpus does not contain is not represented);"
            + " served pipeline with both encoders, held-out reliability by fold",
            **{f"held_out_{s}": reliability(*held[s]) for s in held},
        }
    content = hash_obj({k: {"knots": v["knots"], "values": v["values"]} for k, v in report["routes"].items()})
    run_id = ""
    if not args.no_ledger:
        row = (
            ledger.LedgerRowBuilder(kind="dev")
            .with_metrics(
                {
                    f"{route}_heldout_brier_{name.removeprefix('held_out_')}": fit[name]["brier"]
                    for route, fit in report["routes"].items()
                    for name in fit
                    if name.startswith("held_out_")
                }
            )
            .with_fields(
                rung="confidence-calibration",
                calibration_sha=content,
                source_artifact=args.artifact,
                signal=report["signal"],
                reliability={
                    k: {kk: vv for kk, vv in v.items() if kk not in ("knots", "values")}
                    for k, v in report["routes"].items()
                },
            )
            .build()
        )
        run_id = ledger.append(row).run_id
    out = root / OUT
    out.write_text(json.dumps({"ledger_run_id": run_id, "calibration_sha": content, **report}, indent=1), "utf-8")
    for route, fit in report["routes"].items():
        for name, rel in fit.items():
            if name.startswith("held_out_"):
                bands = "  ".join(
                    f"{lvl}: n={rel[lvl]['n']} top1 {rel[lvl]['top1_relevant_rate']:.2f}"
                    if rel[lvl]["top1_relevant_rate"] is not None
                    else f"{lvl}: n=0"
                    for lvl in ("high", "medium", "low")
                )
                head = f"{route:<15} {name:<22} n={rel['n']:<5} brier {rel['brier']:.3f} base {rel['base_rate']:.2f}"
                print(f"{head}  {bands}")
    print(f"written {OUT}  ledger={run_id or '(not recorded)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
