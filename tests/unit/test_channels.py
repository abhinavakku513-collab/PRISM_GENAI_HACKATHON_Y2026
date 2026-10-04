"""The heterogeneous candidate union: exact symbols, a second dense encoder, exact scores, ranker compatibility."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acis.core.config import freeze_config
from acis.core.types import Snippet
from acis.embed.hashing import HashingEncoder
from acis.engine import AcisEngine
from acis.engine.core import DEFAULT_CONFIG
from acis.lexical.symbols import SymbolIndex, query_symbols, unit_symbols
from acis.rank import candidates as cand
from acis.rank import ltr

DOCS = [
    Snippet(handle="a", text="import heapq\ndef dijkstra(g, s):\n    h = [(0, s)]\n    heapq.heappush(h, (1, s))\n"),
    Snippet(handle="b", text="def sum_intervals(xs):\n    return sum(b - a for a, b in xs)\n"),
    Snippet(handle="c", text="class UnionFind:\n    def find(self, x):\n        return x\n"),
    Snippet(handle="d", text="n = int(input())\nprint(n * 2)\n"),
    Snippet(handle="e", text="def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"),
]


def test_query_symbols_are_found_by_shape_not_by_list():
    assert query_symbols("find the shortest path in a weighted graph") == ()
    assert query_symbols("an array (sorted) of ints") == ()
    assert query_symbols("use UnionFind with path compression") == ("UnionFind",)
    assert query_symbols("Write `sum_intervals()` for the list") == ("sum_intervals",)
    assert query_symbols("call heapq.heappush then solve(n)")[:2] == ("solve", "heapq.heappush")
    assert query_symbols("dijkstra") == ("dijkstra",)  # a query that is one identifier names it


def test_the_symbol_index_ranks_units_containing_rare_symbols_first():
    index = SymbolIndex.build([d.handle for d in DOCS], [d.text for d in DOCS])
    assert "heapq" in unit_symbols(DOCS[0].text) and "heapq.heappush" in unit_symbols(DOCS[0].text)
    assert [d for d, _ in index.search(["sum_intervals"], 5)] == ["b"]
    assert index.search([], 5) == [] and index.search(["nothing_here"], 5) == []


def test_the_union_keeps_every_channel_and_cuts_by_rrf_only_when_over_cap():
    pool = cand.union(
        [("a", 0.9), ("b", 0.8)],
        [("c", 5.0)],
        dense_k=2,
        lexical_k=1,
        cap=10,
        aux=[("d", 0.7)],
        aux_k=1,
        symbol=[("e", 3.0)],
        symbol_k=1,
    )
    assert [c.doc_id for c in pool] == ["a", "b", "c", "d", "e"]
    assert pool[3].aux_rank == 1 and pool[4].symbol_rank == 1 and pool[0].dense_rank == 1
    assert len(cand.union([("a", 0.9), ("b", 0.8)], [("c", 5.0)], dense_k=2, lexical_k=1, cap=2)) == 2


def _engine(**extra: object) -> AcisEngine:
    cfg = {
        **DEFAULT_CONFIG,
        "run": {**DEFAULT_CONFIG["run"], "channel": "hybrid"},
        "retrieve": {**DEFAULT_CONFIG["retrieve"], "symbol_k": 5, **extra.pop("retrieve", {})},  # type: ignore[dict-item]
        "rank": {"generic": {"alpha": 0.9}},
        **extra,
    }
    return AcisEngine.from_config(freeze_config(cfg), encoder=HashingEncoder(dim=64))


def test_a_symbol_hit_enters_the_pool_and_every_candidate_carries_an_exact_cosine():
    engine = _engine()
    snap = engine.build_snapshot(DOCS, source="t")
    data = engine.snapshot_data(snap)
    _, pool = engine.candidate_pool(data, "UnionFind", route="generic", want=5)
    assert any(c.doc_id == "c" and c.symbol_rank for c in pool)
    assert all(not math.isnan(c.dense_score) for c in pool)  # lexical- or symbol-only units are scored too
    matrix = engine.pool_features(data, "UnionFind", pool)
    col = cand.FEATURE_NAMES.index("symbol_hits")
    hits = {c.doc_id: matrix[i, col] for i, c in enumerate(pool)}
    assert hits["c"] == 1.0 and hits.get("d", 0.0) == 0.0


def test_no_symbol_in_the_query_means_nan_symbol_features_not_zero():
    engine = _engine()
    data = engine.snapshot_data(engine.build_snapshot(DOCS, source="t"))
    _, pool = engine.candidate_pool(data, "double a number read from input", route="generic", want=5)
    matrix = engine.pool_features(data, "double a number read from input", pool)
    assert np.isnan(matrix[:, cand.FEATURE_NAMES.index("symbol_hits")]).all()


def test_a_second_dense_encoder_adds_its_own_channel_and_features():
    engine = _engine(model={"encoder": "hashing", "aux_encoder": "hashing", "dim": 64}, retrieve={"aux_k": 3})
    data = engine.snapshot_data(engine.build_snapshot(DOCS, source="t"))
    assert data.aux_vectors is not None and data.aux_vectors.shape[0] == len(DOCS)
    _, pool = engine.candidate_pool(data, "reverse the words", route="generic", want=5)
    assert any(c.aux_rank for c in pool)
    matrix = engine.pool_features(data, "reverse the words", pool)
    assert np.isfinite(matrix[:, cand.FEATURE_NAMES.index("cos2")]).all()


def test_an_older_ranker_reads_its_own_columns_from_the_wider_matrix():
    rng = np.random.default_rng(0)
    old_names = tuple(n for n in cand.FEATURE_NAMES if n not in ("cos2", "rank_dense2", "z_cos2", "n_channels"))
    groups = []
    for q in range(40):
        full = rng.standard_normal((6, len(cand.FEATURE_NAMES))).astype(np.float32)
        labels = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        groups.append((f"q{q}", full, labels))
    model = ltr.train([ltr.TrainingGroup(q, m, y, tuple("abcdef")) for q, m, y in groups], rounds=5)
    older = ltr.Ranker(booster=model.booster, feature_names=cand.FEATURE_NAMES)
    order, _ = older.rerank(list("abcdef"), groups[0][1])
    assert sorted(order) == list("abcdef")
    # A model that names a feature the pipeline no longer produces is refused, never fed shifted columns.
    stale = ltr.Ranker(booster=model.booster, feature_names=(*old_names, "gone_feature"))
    try:
        stale.score(groups[0][1])
    except Exception as exc:  # noqa: BLE001
        assert "feature width" in str(exc)
    else:
        raise AssertionError("a stale feature list must be refused")


def test_a_response_says_which_channels_ran_and_how_many_candidates_each_gave():
    from acis.core.types import SearchRequest
    from acis.engine.routing import categorize

    engine = _engine()
    engine.build_snapshot(DOCS, source="t")
    e = engine.search(SearchRequest(query="UnionFind", top_k=3, explain=True)).explanation
    assert e["category"] in ("exact_symbol", "out_of_corpus") and e["query_symbols"] == ["UnionFind"]
    assert e["candidates"]["candidates"] >= 1 and e["candidates"]["from_symbols"] >= 1
    assert "symbols" in e["channels"] and "dense" in e["channels"]
    assert categorize("x = a[i] + b[j]; y = (c * d) / e", "generic", (), weak_match=False) == "code"
    assert categorize("something with arrays", "generic", (), weak_match=False) == "vague"
    assert categorize("long statement", "statement_like", (), weak_match=True) == "statement_like"


def test_every_candidate_gets_its_whole_snapshot_rank_under_the_second_encoder():
    """`rank_dense2` must exist for every pooled unit — not only those the second channel put in its own top list."""
    engine = _engine(model={"encoder": "hashing", "aux_encoder": "hashing", "dim": 64}, retrieve={"aux_k": 0})
    data = engine.snapshot_data(engine.build_snapshot(DOCS, source="t"))
    query = "reverse the words"
    _, pool = engine.candidate_pool(data, query, route="generic", want=5)
    assert pool and all(c.aux_rank == 0 for c in pool)  # the channel added nothing to the union …
    vector = engine._aux_query_vector(data.snapshot.snapshot_id, query, route="generic")
    from acis.embed.base import exact_search

    scores = exact_search(vector.reshape(1, -1), data.aux_vectors)[0]
    for c in pool:  # … yet every candidate carries its exact rank over the whole snapshot
        assert c.aux_full_rank == int((scores > scores[data.position(c.doc_id)]).sum()) + 1
    matrix = engine.pool_features(data, query, pool)
    assert np.isfinite(matrix[:, cand.FEATURE_NAMES.index("rank_dense2")]).all()
    # Channel agreement still counts only the channels that retrieved the unit.
    assert (matrix[:, cand.FEATURE_NAMES.index("n_channels")] <= 3).all()


def test_fusion_second_encoder_weight_zero_is_exactly_the_tuned_formula():
    from acis.rank.fusion import weighted_fusion

    pool = [
        cand.Candidate("a", dense_score=0.9, aux_score=0.1, lexical_score=2.0, lexical_rank=2),
        cand.Candidate("b", dense_score=0.5, aux_score=0.8, lexical_score=4.0, lexical_rank=1),
        cand.Candidate("c", dense_score=0.1, aux_score=0.9),
    ]
    assert weighted_fusion(pool, alpha=0.9) == weighted_fusion(pool, alpha=0.9, aux_weight=0.0)
    mixed = weighted_fusion(pool, alpha=1.0, aux_weight=0.5)
    assert mixed["b"] > mixed["a"]  # the second encoder's preference moves the order
    only_second = weighted_fusion(pool, alpha=1.0, aux_weight=1.0)
    assert max(only_second, key=only_second.get) == "c"
    # No second-encoder score anywhere in the pool: the primary alone, never a silent zero.
    bare = [cand.Candidate(c.doc_id, dense_score=c.dense_score) for c in pool]
    assert weighted_fusion(bare, alpha=1.0, aux_weight=0.5) == weighted_fusion(bare, alpha=1.0)


def test_the_second_encoder_follows_the_primary_cache_policy(monkeypatch):
    """A cold official run builds the primary without the vector cache; the second encoder must not read one (D17)."""
    import acis.embed.factory as factory

    seen: list[bool] = []

    def fake_build(config, *, cache=True):
        seen.append(cache)
        return HashingEncoder(dim=64)

    monkeypatch.setattr(factory, "build_encoder", fake_build)
    for primary_cache, expected in ((None, False), (object(), True)):
        from types import SimpleNamespace

        primary = SimpleNamespace(cache=primary_cache)  # only its cache policy is read here
        cfg = {**DEFAULT_CONFIG, "model": {"encoder": "hashing", "aux_encoder": "x", "dim": 64}}
        engine = AcisEngine.from_config(freeze_config(cfg), encoder=primary)
        assert engine.aux_encoder is not None
        assert seen[-1] is expected


def test_a_strict_run_refuses_a_second_encoder_that_cannot_load(monkeypatch):
    import pytest

    import acis.embed.factory as factory
    from acis.core.errors import NotReady

    def broken(config, *, cache=True):
        raise RuntimeError("no weights")

    monkeypatch.setattr(factory, "build_encoder", broken)
    run = {**DEFAULT_CONFIG["run"], "strict": True}
    cfg = {**DEFAULT_CONFIG, "run": run, "model": {"encoder": "hashing", "aux_encoder": "x", "dim": 64}}
    with pytest.raises(NotReady):
        _ = AcisEngine.from_config(freeze_config(cfg), encoder=HashingEncoder(dim=64)).aux_encoder
    lenient = AcisEngine.from_config(
        freeze_config({**cfg, "run": DEFAULT_CONFIG["run"]}), encoder=HashingEncoder(dim=64)
    )
    assert lenient.aux_encoder is None and lenient.counters.get("dense2.unavailable") == 1


def test_encoder_agreement_features_exist_with_a_second_encoder_and_are_nan_without():
    cols = [cand.FEATURE_NAMES.index(n) for n in cand.GROUPS["mix"]]
    query = "reverse the words"
    two = _engine(model={"encoder": "hashing", "aux_encoder": "hashing", "dim": 64}, retrieve={"aux_k": 3})
    data = two.snapshot_data(two.build_snapshot(DOCS, source="t"))
    _, pool = two.candidate_pool(data, query, route="generic", want=5)
    matrix = two.pool_features(data, query, pool)
    assert np.isfinite(matrix[:, cols]).all()
    assert all(c.dense_full_rank >= 1 for c in pool)
    # Pool rank by z_cos + z_cos2 is a permutation of 1..n.
    assert sorted(matrix[:, cand.FEATURE_NAMES.index("mix_poolrank")].tolist()) == list(range(1, len(pool) + 1))
    one = _engine()
    data1 = one.snapshot_data(one.build_snapshot(DOCS, source="t"))
    _, pool1 = one.candidate_pool(data1, query, route="generic", want=5)
    assert np.isnan(one.pool_features(data1, query, pool1)[:, cols]).all()


def test_confidence_signal_is_the_mean_of_both_encoders_z_when_a_second_encoder_exists():
    from acis.embed.base import exact_search
    from acis.rank.confidence import CROWD, z_top1

    engine = _engine(model={"encoder": "hashing", "aux_encoder": "hashing", "dim": 64}, retrieve={"aux_k": 3})
    data = engine.snapshot_data(engine.build_snapshot(DOCS, source="t"))
    query, served = "reverse the words", ["e", "a"]

    def z(vec: np.ndarray, matrix: np.ndarray) -> float:
        s = exact_search(vec.reshape(1, -1), matrix)[0]
        top = np.sort(-np.partition(-s, min(CROWD, len(s)) - 1)[: min(CROWD, len(s))])[::-1]
        return z_top1(top, float(s[data.position("e")]))

    z1 = z(engine._query_vector(data.snapshot.snapshot_id, query, route="generic"), data.vectors)
    z2 = z(engine._aux_query_vector(data.snapshot.snapshot_id, query, route="generic"), data.aux_vectors)
    assert engine.confidence_signal(data, query, served, route="generic") == pytest.approx((z1 + z2) / 2)
