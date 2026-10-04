"""`AcisEngine` — the one engine behind every surface (docs/spec/06 §1, docs/spec/02 §4).

Phase 1 builds the skeleton that P0 needs and that Track B compiles against: content-addressed snapshots, an exact
dense channel, a per-snapshot BM25 channel, the degradation ladder, and the batch surface the mteb adapter uses.
Fusion, features and LTR are Phase 4 and are *absent*, not stubbed: `mode="hybrid"` raises rather than inventing an
ungated ranking.

Invariants implemented here:

* **INV-1** every `Hit.source` is re-read from the content store by `body_hash`; nothing is generated.
* **INV-2** results come only from the requested snapshot; cache keys contain `snapshot_id` and `config_hash`.
* **INV-3** a query's ranking never depends on the other queries in its batch: no per-batch statistics anywhere.
* **INV-4** external ids are opaque; only exact-duplicate ordering uses the corpus ordinal.
* **INV-6** same (snapshot, config, profile, threads) ⇒ same ranking.
* **INV-7** every fallback increments a counter and shows up in `degradations`; strict mode aborts instead.
* **INV-9** only VALID snapshots are searchable unless `allow_partial=True`.
* **INV-13** `agent_calls == 0` — there is no agent on this path.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from acis.core.config import FrozenConfig, freeze_config
from acis.core.errors import InvalidInput, NotFound, NotReady, SnapshotInvalid
from acis.core.hashing import hash_obj, sha256_text, short
from acis.core.numeric import apply_blas_threads, resolve_threads
from acis.core.types import (
    BuildReport,
    Confidence,
    Diagnostics,
    EvalReport,
    EvalSpec,
    Hit,
    Route,
    SearchRequest,
    SearchResponse,
    Snapshot,
    SnapshotRef,
    Snippet,
    Unit,
)
from acis.embed.base import Encoder, exact_search
from acis.engine.versions import VersionedEngineMixin
from acis.lexical.bm25 import Bm25Index
from acis.lexical.tokenize import corpus_text
from acis.obs.counters import Counters, degradation
from acis.obs.timing import STAGES, stage
from acis.prep.normalize import d1, is_empty_query, lexical_view, q1
from acis.prep.truncate import head_tail
from acis.prep.views import build_view, weak_marker_count

MAX_QUERY_CHARS = 16_000
MAX_TOP_K = 1000


@dataclass(slots=True)
class SnapshotData:
    """One immutable, content-addressed corpus view. Track B1 persists this; Phase 1 keeps it in memory."""

    snapshot: Snapshot
    doc_ids: tuple[str, ...]
    body_hashes: tuple[str, ...]
    store: dict[str, str]  # body_hash -> exact bytes we return as evidence (INV-1)
    ordinal: dict[str, int]
    hash_of: dict[str, str]
    lexical: Bm25Index | None = None
    vectors: np.ndarray | None = None
    missing: tuple[str, ...] = ()
    #: Exact-symbol inverted index (`acis.lexical.symbols`), built with the snapshot.
    symbols: Any = None
    #: The second dense encoder's document matrix (`model.aux_encoder`), when one is configured.
    aux_vectors: np.ndarray | None = None
    #: Parse-only document features, built once per snapshot on first use (Phase 4). Keyed by body hash, so
    #: duplicate documents — and the same body in another version — are parsed once.
    _features: dict[str, Any] = field(default_factory=dict, repr=False)
    _dup_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _positions: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def position(self, doc_id: str) -> int:
        """Row of `doc_id` in the dense matrix; built once per (immutable) snapshot."""
        if not self._positions:
            self._positions.update({d: i for i, d in enumerate(self.doc_ids)})
        return self._positions[doc_id]

    def features_of(self, doc_id: str) -> Any:
        """Document features, parsed on demand and cached for the life of the snapshot (immutable content)."""
        from acis.features.doc import extract  # noqa: PLC0415

        body_hash = self.hash_of[doc_id]
        cached = self._features.get(body_hash)
        if cached is None:
            cached = extract(self.store[body_hash])
            self._features[body_hash] = cached
        return cached

    def warm_features(self) -> int:
        """Extract every document's features now rather than on the first query that ranks it. Returns the count.

        The content is immutable, so this changes no ranking — it moves ~0.1 s of first-query parsing to start-up.
        """
        for doc_id in self.doc_ids:
            self.features_of(doc_id)
        return len(self._features)

    def duplicate_count(self, doc_id: str) -> int:
        """How many documents in this snapshot share this exact body — a corpus fact, never a query one."""
        if not self._dup_counts:
            for h in self.body_hashes:
                self._dup_counts[h] = self._dup_counts.get(h, 0) + 1
        return self._dup_counts.get(self.hash_of[doc_id], 1)

    def text_of(self, doc_id: str) -> str:
        """Evidence is always re-read from the store by hash — never carried along from the ranking (INV-1)."""
        body_hash = self.hash_of.get(doc_id)
        if body_hash is None:
            raise NotFound(f"document {doc_id!r} is not in snapshot {self.snapshot.snapshot_id}")
        return self.store[body_hash]

    def unit_of(self, doc_id: str) -> Unit:
        return Unit(
            unit_id=f"u_{short(self.hash_of[doc_id], 16)}",
            key=doc_id,
            body_hash=self.hash_of[doc_id],
            repo_id=self.snapshot.repo_id,
            version_id=self.snapshot.version_id,
            snapshot_id=self.snapshot.snapshot_id,
            n_bytes=len(self.store[self.hash_of[doc_id]].encode("utf-8")),
        )


DEFAULT_CONFIG: dict[str, object] = {
    "run": {"mode": "B", "strict": False, "device": "cpu", "threads": "auto_physical", "seed": 0},
    "prep": {
        "query": {"version": "q1", "view": "V0", "max_tokens": 1024, "head": 768, "tail": 256},
        "doc": {"version": "d1", "max_tokens": 1024, "head": 768, "tail": 256},
    },
    "lexical": {"k1": 1.5, "b": 0.75, "stemmer": "english"},
    "retrieve": {"dense_k": 100, "lexical_k": 30, "union_cap": 100, "top_k_out": 1000},
}


#: Per-request stage timings live in `acis.obs.timing` so the encoder can report its own queueing (re-exported).
_STAGES = STAGES


def query_encoder_texts(config: Any, normalised_query: str) -> tuple[str, ...]:
    """The exact text(s) the dense channel encodes for an already-normalised query.

    V0 and V1 are one text each: the view, then head+tail. V2 is two — V0's and V1's — whose vectors are averaged
    and renormalised; a query with no structure renders V1 as V0, so it is one text and one encode. An unknown view
    is an error: it used to fall back to V0, which made a V2 configuration silently run V0.

    One definition, used by the engine, by Mode B and by the training export — where a query prepared differently
    from serving is a train/serve skew nobody would see in the numbers.
    """
    prep = config.section("prep").get("query", {})
    view = str(prep.get("view", "V0"))
    if view not in ("V0", "V1", "V2"):
        raise ValueError(f"unknown query view {view!r}; known: V0, V1, V2")

    def render(v: str) -> str:
        return head_tail(
            build_view(normalised_query, v),
            max_tokens=int(prep.get("max_tokens", 1024)),
            head=int(prep.get("head", 768)),
            tail=int(prep.get("tail", 256)),
        ).text

    if view != "V2":
        return (render(view),)
    v0, v1 = render("V0"), render("V1")
    return (v0,) if v1 == v0 else (v0, v1)


def query_encoder_text(config: Any, normalised_query: str) -> str:
    """The single text for a one-text view. V2 has two; callers that need one text refuse it."""
    texts = query_encoder_texts(config, normalised_query)
    if len(texts) != 1:
        raise ValueError("this caller needs a single query text; view V2 encodes two")
    return texts[0]


def pooled_query_vector(vectors: np.ndarray) -> np.ndarray:
    """One text → its vector; two (V2) → their normalised mean."""
    if vectors.shape[0] == 1:
        return np.asarray(vectors[0], dtype=np.float32)
    total = np.asarray(vectors.sum(axis=0), dtype=np.float32)
    norm = float(np.linalg.norm(total)) or 1.0
    return total / norm


class AcisEngine(VersionedEngineMixin):
    """The engine. Construct with `AcisEngine.from_config(...)`; everything else is a method on the frozen surface."""

    def __init__(self, config: FrozenConfig, *, encoder: Encoder | None = None) -> None:
        self.config = config
        self.encoder = encoder
        self.counters = Counters()
        self.threads = resolve_threads(config.get("run.threads", "auto_physical"))
        #: The retrieval core's matrix-vector products are memory-bound: more BLAS threads only add wake-up cost.
        self.blas_threads = apply_blas_threads(self.threads)
        #: In-memory snapshots built through `build_snapshot` (the P0 path: one corpus, no versions).
        self._snapshots: dict[str, SnapshotData] = {}
        #: Snapshots loaded from the store, keyed by `(repo_id, snapshot_id)` (the P1 path, Track B1).
        self._loaded: dict[tuple[str, str], SnapshotData] = {}
        self._reports: dict[str, list[BuildReport]] = {}
        self._query_vector_cache: dict[str, np.ndarray] = {}
        #: The learned ranker, when one has been trained and the config points at it (Phase 4, gate G5).
        self._ranker: Any = None
        self._ranker_loaded = False
        #: Lineage index per repository, keyed by the versions it was built from (Track B2).
        self._lineages: dict[str, tuple[Any, Any]] = {}
        #: The TRAIN-query bank routing v1.1 reads (spec 10 §4). `None` sends every query down the generic path.
        self._bank: Any = None
        self._bank_loaded = False
        #: The confidence calibration (`confidence.calibration`); loaded once, `None` when not configured.
        self._calibration: Any = None
        self._calibration_loaded = False
        #: The second dense encoder (`model.aux_encoder`, spec 08 §3 L10), built once when configured.
        self._aux_encoder: Any = None
        self._aux_loaded = False

    # -- construction ------------------------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls, config: FrozenConfig | Mapping[str, object] | None = None, *, encoder: Encoder | None = None
    ) -> AcisEngine:
        if config is None:
            config = freeze_config(DEFAULT_CONFIG, source_path="<default>")
        elif not isinstance(config, FrozenConfig):
            config = freeze_config(config)
        return cls(config, encoder=encoder)

    @property
    def config_hash(self) -> str:
        return self.config.config_hash

    @property
    def ranker(self) -> Any:
        """The trained ranker named by `rank.model`, loaded once. `None` means fusion, never a silent identity."""
        if self._ranker_loaded:
            return self._ranker
        self._ranker_loaded = True
        path = str(self.config.get("rank.model", "") or "")
        if path:
            from acis.core.paths import acis_root  # noqa: PLC0415
            from acis.rank.ltr import Ranker  # noqa: PLC0415

            target = Path(path)
            self._ranker = Ranker.load(target if target.is_absolute() else acis_root() / target)
        return self._ranker

    @property
    def model_fingerprint(self) -> str:
        return self.encoder.fingerprint if self.encoder is not None else "none"

    def _channels(self) -> tuple[bool, bool]:
        return self.encoder is not None, True

    # -- snapshots ---------------------------------------------------------------------------------------------
    def build_snapshot(self, docs: Sequence[Snippet], *, source: str) -> Snapshot:
        """Content-address the documents, build the lexical index and (when an encoder exists) the dense matrix."""
        if not docs:
            raise InvalidInput("cannot build a snapshot from zero documents")
        started = time.perf_counter()

        doc_ids: list[str] = []
        body_hashes: list[str] = []
        store: dict[str, str] = {}
        for snippet in docs:
            text = d1(snippet.text)
            body_hash = sha256_text(text)
            doc_ids.append(str(snippet.handle))
            body_hashes.append(body_hash)
            store.setdefault(body_hash, text)
        if len(set(doc_ids)) != len(doc_ids):
            raise InvalidInput("duplicate document handles in a snapshot", n=len(doc_ids), unique=len(set(doc_ids)))

        snapshot_id = "s_" + short(
            hash_obj({"config": self.config_hash, "docs": body_hashes, "ids": doc_ids, "source": source}), 16
        )
        hash_of = dict(zip(doc_ids, body_hashes, strict=True))
        ordinal = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        lexical = Bm25Index.build(
            doc_ids,
            [corpus_text("", store[h]) for h in body_hashes],
            k1=float(self.config.get("lexical.k1", 1.5)),
            b=float(self.config.get("lexical.b", 0.75)),
            stemmer_language=self.config.get("lexical.stemmer", "english"),
            tokenizer=str(self.config.get("lexical.tokenizer", "stock")),
        )
        vectors = self._embed_documents([store[h] for h in body_hashes]) if self.encoder is not None else None
        missing: tuple[str, ...] = () if vectors is not None else ("dense",)
        aux_vectors = self._embed_aux([store[h] for h in body_hashes])
        if self.aux_encoder_name and aux_vectors is None:
            missing = (*missing, "dense2")
        from acis.lexical.symbols import SymbolIndex  # noqa: PLC0415

        symbols = SymbolIndex.build(doc_ids, [store[h] for h in body_hashes])
        if lexical.vocabulary_empty:
            # Nothing in this corpus is indexable lexically. The snapshot says so rather than pretending to have
            # a channel that can only ever return nothing.
            missing = (*missing, "lexical")

        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            repo_id="-",
            version_id="v0",
            n_units=len(doc_ids),
            config_hash=self.config_hash,
            source=source,
            state="VALID",
            missing_channels=missing,
            created_ts=time.time(),
        )
        self._snapshots[snapshot_id] = SnapshotData(
            snapshot=snapshot,
            doc_ids=tuple(doc_ids),
            body_hashes=tuple(body_hashes),
            store=store,
            ordinal=ordinal,
            hash_of=hash_of,
            lexical=lexical,
            vectors=vectors,
            missing=missing,
            symbols=symbols,
            aux_vectors=aux_vectors,
        )
        self.counters.incr("snapshot.builds")
        self.counters.incr("snapshot.build_ms", int((time.perf_counter() - started) * 1000))
        return snapshot

    def snapshot_data(self, snap: Snapshot | str) -> SnapshotData:
        snapshot_id = snap if isinstance(snap, str) else snap.snapshot_id
        data = self._snapshots.get(snapshot_id)
        if data is None:
            raise NotFound(f"unknown snapshot {snapshot_id!r}")
        return data

    def _embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        prep = self.config.section("prep").get("doc", {})
        prepared = [
            head_tail(
                t,
                max_tokens=int(prep.get("max_tokens", 1024)),
                head=int(prep.get("head", 768)),
                tail=int(prep.get("tail", 256)),
            ).text
            for t in texts
        ]
        assert self.encoder is not None
        return self.encoder.encode(prepared, is_query=False)

    @property
    def aux_encoder_name(self) -> str:
        return str(self.config.get("model.aux_encoder", "") or "")

    @property
    def aux_encoder(self) -> Any:
        """The second dense encoder named by `model.aux_encoder`, loaded once (the same pinned registry and
        vector cache as the primary). `None` when not configured, or when its weights are absent (counted)."""
        if self._aux_loaded:
            return self._aux_encoder
        self._aux_loaded = True
        name = self.aux_encoder_name
        if name and self.encoder is not None:
            from acis.embed.factory import build_encoder  # noqa: PLC0415

            # The second encoder follows the primary's cache policy: a cold official run builds the primary without
            # the vector cache (D17), and a cached second encoder would quietly turn its timing into a warm one.
            cached = getattr(self.encoder, "cache", True) is not None
            try:
                self._aux_encoder = build_encoder(self.config.with_overrides(**{"model.encoder": name}), cache=cached)
            except Exception as exc:  # noqa: BLE001 — a missing second encoder is a missing channel, never a crash
                self.counters.incr("dense2.unavailable")
                self._aux_encoder = None
                if self.config.strict:
                    # Strict runs refuse every fallback (INV-7): a configured channel that cannot load aborts.
                    raise NotReady(f"the configured second encoder {name!r} could not be loaded: {exc}") from exc
        return self._aux_encoder

    def _embed_aux(self, texts: Sequence[str]) -> np.ndarray | None:
        encoder = self.aux_encoder
        if encoder is None:
            return None
        prep = self.config.section("prep").get("doc", {})
        prepared = [
            head_tail(
                t,
                max_tokens=int(prep.get("max_tokens", 1024)),
                head=int(prep.get("head", 768)),
                tail=int(prep.get("tail", 256)),
            ).text
            for t in texts
        ]
        return np.asarray(encoder.encode(prepared, is_query=False), dtype=np.float32)

    def _aux_query_vector(self, snapshot_id: str, query: str, *, route: str) -> np.ndarray:
        """The query under the second encoder — its own instruction per route when its card has one (INV-15)."""
        encoder = self.aux_encoder
        assert encoder is not None
        if not getattr(encoder, "route_sensitive", True):
            # Same rule as the primary (`_query_vector`): no instruction format, so every route is one vector, and
            # keying it per route would only miss the cache.
            route = "generic"
        texts = query_encoder_texts(self.config, query)
        key = hash_obj(
            {
                "snapshot": snapshot_id,
                "config": self.config_hash,
                "aux": encoder.fingerprint,
                "route": route,
                "text": "\x00".join(texts),
            }
        )
        cached = self._query_vector_cache.get(key)
        if cached is not None:
            return cached
        with stage("encode2"):
            vector = pooled_query_vector(
                np.asarray(encoder.encode(list(texts), is_query=True, route=route), dtype=np.float32)
            )
        self._query_vector_cache[key] = vector
        return vector

    # -- query preparation --------------------------------------------------------------------------------------
    def normalise_query(self, text: str) -> tuple[str, bool]:
        """q1 plus the hard input rules of spec 02 §6b. Returns `(normalised, truncated)`."""
        if text is None or not isinstance(text, str):
            raise InvalidInput("query must be a string")
        if is_empty_query(text):
            raise InvalidInput("query is empty")
        truncated = len(text) > MAX_QUERY_CHARS
        if truncated:
            keep = MAX_QUERY_CHARS // 2
            text = text[:keep] + text[-keep:]
        return q1(text), truncated

    @property
    def query_bank(self) -> Any:
        """The TRAIN-query embeddings routing reads, loaded once. `None` means every query takes the generic path.

        Query-side only: it selects a pipeline and never offsets a document's score (INV-15, spec 10 §4).
        """
        if self._bank_loaded:
            return self._bank
        self._bank_loaded = True
        path = str(self.config.get("route.bank", "") or "")
        if path:
            from acis.core.paths import acis_root  # noqa: PLC0415
            from acis.engine.routing import QueryBank  # noqa: PLC0415

            target = Path(path)
            target = target if target.is_absolute() else acis_root() / target
            if target.is_file():
                self._bank = QueryBank.load(target, k=int(self.config.get("route.k", 8)))
        return self._bank

    def route(self, query: str, *, availability: float | None = None) -> Route:
        """Routing v1.1 (spec 10 §4). See `route_decision` for the signals and the reason."""
        return cast(Route, self.route_decision(query, availability=availability).route)

    def encoding_route(self, query: str) -> Route:
        """The route a query is *encoded* with — computed only when it can change the vector.

        Routing embeds the query itself (the OOD signal), so for an encoder whose input is the same on every route
        it would be a second full-length forward pass per query that changes nothing. An encoder that does not
        declare `route_sensitive` is assumed sensitive.
        """
        if self.encoder is not None and not getattr(self.encoder, "route_sensitive", True):
            return "generic"
        return self.route(query)

    def route_decision(self, query: str, *, availability: float | None = None) -> Any:
        """The full decision: route, OOD score, feature availability and why.

        Availability is passed in when the caller has already built the features (the hybrid path has), and
        computed from the query alone otherwise — a query with no numbers, no quoted output and no structural
        cues gives the ranker nothing, whatever it resembles.
        """
        with stage("route"):  # exclusive: the query's own encode inside it is timed as `route_encode`
            return self._route_decision(query, availability=availability)

    def _route_decision(self, query: str, *, availability: float | None) -> Any:
        from acis.engine.routing import DEFAULT_RHO, DEFAULT_TAU, decide  # noqa: PLC0415
        from acis.features.query import BRIDGE_FEATURES  # noqa: PLC0415
        from acis.features.query import extract as query_features

        _ = weak_marker_count(query)  # a signal the decision may use later; never a filter on its own
        bank = self.query_bank
        if availability is None:
            qf = query_features(query)
            fired = sum(
                1
                for name, present in (
                    ("out_literal_recall", bool(qf.expected_outputs)),
                    ("numeric_literal_overlap", bool(qf.numeric_literals)),
                    ("const_jaccard", bool(qf.numeric_literals)),
                    ("io_shape_compat", qf.mentions_input),
                    ("tc_loop_expected", qf.mentions_testcases),
                )
                if present
            )
            availability = fired / len(BRIDGE_FEATURES)

        vector = None
        if bank is not None and self.encoder is not None:
            try:
                # The dense channel's own input (view + head/tail), so the dense stage's encode is a cache hit:
                # encoding the raw text here cost a second full-length forward pass on every query over 1,024
                # tokens (5.5 s for a 16k-character query). Identical text for anything shorter.
                with stage("route_encode"):
                    texts = query_encoder_texts(self.config, query)
                    vector = pooled_query_vector(
                        np.asarray(self.encoder.encode(list(texts), is_query=True), dtype=np.float32)
                    )
            except Exception:  # noqa: BLE001 — a routing failure routes down, it never fails a search
                vector = None
        return decide(
            vector,
            float(availability),
            bank,
            tau=float(self.config.get("route.tau", DEFAULT_TAU)),
            rho=float(self.config.get("route.rho", DEFAULT_RHO)),
        )

    def _query_vector(self, snapshot_id: str, query: str, *, route: str = "generic") -> np.ndarray:
        assert self.encoder is not None
        # The route changes a vector only through the instruction it selects. For an encoder whose input is the
        # same on every route, encoding under the pipeline's route would miss the vector routing just computed
        # for the same text — a second full forward pass on every fresh interactive query.
        if not getattr(self.encoder, "route_sensitive", True):
            route = "generic"
        texts = query_encoder_texts(self.config, query)
        prepared = "\x00".join(texts)  # the cache key must distinguish a V2 pair from either of its texts
        # INV-2: the cache key carries the snapshot and the config, so a vector can never cross either boundary.
        key = hash_obj(
            {
                "snapshot": snapshot_id,
                "config": self.config_hash,
                "model": self.model_fingerprint,
                "profile": self.config.numeric_profile,
                # The route chooses the instruction the query is encoded with (INV-15), so two routes are two
                # different vectors of the same text. Leaving it out of the key would serve one for the other.
                "route": route,
                "text": prepared,
            }
        )
        cached = self._query_vector_cache.get(key)
        if cached is not None:
            self.counters.incr("cache.qemb.hit")
            return cached
        self.counters.incr("cache.qemb.miss")
        with stage("encode"):
            vector = pooled_query_vector(
                np.asarray(self.encoder.encode(list(texts), is_query=True, route=route), dtype=np.float32)
            )
        self._query_vector_cache[key] = vector
        return vector

    # -- channels ------------------------------------------------------------------------------------------------
    def _dense_ranking(
        self, data: SnapshotData, query: str, *, route: str = "generic", k: int | None = None
    ) -> list[tuple[str, float]]:
        """The dense channel's top-`k`, already in final order. `k=None` ranks the whole snapshot."""
        if data.vectors is None or self.encoder is None:
            raise NotReady("the dense channel is not available for this snapshot")
        vector = self._query_vector(data.snapshot.snapshot_id, query, route=route)
        with stage("dense"):
            scores = exact_search(vector.reshape(1, -1), data.vectors)[0]
            return self._stable_top_k(data, scores, data.size if k is None else k)

    @classmethod
    def _stable_top_k(cls, data: SnapshotData, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        """The `k` best documents in `_stable_order` order, without ordering the other 8,755 of them.

        Sorting the whole corpus per query is the single most expensive thing the retrieval core did, and almost
        all of it was thrown away: a request for 10 results does not need ranks 11 to 8,765 in order. The cut is
        exact rather than approximate — the threshold is the k-th best score and *every* document that matches it
        is sorted, so a tie at the boundary cannot be dropped on a technicality and the answer is identical to
        the full sort (pinned by a metamorphic test, not by inspection).
        """
        n = int(scores.shape[0])
        k = max(0, min(int(k), n))
        if k == 0:
            return []
        if k < n:
            window = np.argpartition(-scores, k - 1)[:k]
            threshold = float(scores[window].min())
            indices = np.flatnonzero(scores >= threshold)
        else:
            indices = np.arange(n)
        ranking = [(data.doc_ids[i], float(scores[i])) for i in indices.tolist()]
        return cls._stable_order(data, ranking)[:k]

    @staticmethod
    def _stable_order(data: SnapshotData, ranking: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
        """Deterministic order that does not depend on how the corpus happens to be arranged.

        Equal scores are broken by **content hash** — derived from the document's bytes, not from its id or its
        position (INV-4) — so shuffling the corpus or relabelling documents cannot move a result. Only documents
        with *identical* content reach the last key, the corpus ordinal, which is the one place order may be used
        (D9): it gives exact duplicates distinct adjacent ranks.
        """
        return sorted(
            ranking,
            key=lambda item: (-item[1], data.hash_of.get(item[0], item[0]), data.ordinal.get(item[0], 1 << 30)),
        )

    def candidate_pool(
        self,
        data: SnapshotData,
        query: str,
        *,
        route: str,
        want: int,
        counters: Counters | None = None,
        strict: bool = False,
    ) -> tuple[list[tuple[str, float]], list[Any]]:
        """Stages 3–6 of spec 02 §4: the dense prefix and the candidate union, exactly as serving builds them.

        Public so the offline tuning of the generic fusion reads the same pools the engine ranks — one code path,
        not a re-implementation that could drift from it.
        """
        from acis.rank import candidates as cand  # noqa: PLC0415

        retrieve = self.config.section("retrieve")
        lexical_k = int(retrieve.get("lexical_k", 30))
        union_cap = int(retrieve.get("union_cap", 100))
        dense_k = int(retrieve.get("dense_k", 100))
        # Only the top `dense_k` feed the pool and the tail only has to fill `want` past the head (≤ union_cap), so
        # this prefix is all that is ever read. `_stable_top_k` is exact, so it is the same prefix a full sort gives
        # — without building and tie-sorting all 8,765 documents on every query.
        dense_order = self._dense_ranking(data, query, route=route, k=min(data.size, want + union_cap + dense_k))

        lexical_order: list[tuple[str, float]] = []
        if data.lexical is not None and "lexical" not in data.missing:
            lexical_order = self._lexical_ranking(data, query, lexical_k)
        else:
            degradation("lexical_unavailable", "dense-only fusion", strict=strict, counters=counters or self.counters)

        aux_k = int(retrieve.get("aux_k", 0))
        aux_order: list[tuple[str, float]] = []
        if aux_k and data.aux_vectors is not None and self.aux_encoder is not None:
            vector2 = self._aux_query_vector(data.snapshot.snapshot_id, query, route=route)
            with stage("dense2"):
                aux_order = self._stable_top_k(data, exact_search(vector2.reshape(1, -1), data.aux_vectors)[0], aux_k)
        elif aux_k and self.aux_encoder_name:
            # Configured but absent for this snapshot (e.g. a P1 snapshot built with one encoder): counted, visible.
            (counters or self.counters).incr("dense2.missing")

        symbol_k = int(retrieve.get("symbol_k", 0))
        symbol_order: list[tuple[str, float]] = []
        if symbol_k and data.symbols is not None:
            from acis.lexical.symbols import query_symbols  # noqa: PLC0415

            with stage("symbols"):
                symbol_order = data.symbols.search(query_symbols(query), symbol_k)
                if route != "statement_like" and float(self.config.get("rank.generic.symbol_beta", 0.0) or 0.0):
                    # Generic route only: units defining a name the query uses join the pool (spec 10 R-Q2 keeps
                    # the statement route's pools exactly as its ranker was trained on them).
                    seen = {d for d, _ in symbol_order}
                    extra = data.symbols.defined_search(data.symbols.defined_in_query(query), symbol_k)
                    symbol_order += [(d, s) for d, s in extra if d not in seen][: max(0, symbol_k - len(symbol_order))]

        sink = counters or self.counters
        sink.incr("pool.from_dense", min(dense_k, len(dense_order)))
        sink.incr("pool.from_bm25", len(lexical_order))
        sink.incr("pool.from_dense2", len(aux_order))
        sink.incr("pool.from_symbols", len(symbol_order))
        pool = cand.union(
            dense_order,
            lexical_order,
            dense_k=dense_k,
            lexical_k=lexical_k,
            cap=union_cap,
            aux=aux_order,
            aux_k=aux_k,
            symbol=symbol_order,
            symbol_k=symbol_k,
        )
        sink.incr("pool.candidates", len(pool))
        return dense_order, self._with_exact_scores(data, query, pool, route=route)

    def _with_exact_scores(self, data: SnapshotData, query: str, pool: list[Any], *, route: str) -> list[Any]:
        """Every candidate gets its exact cosine under each dense encoder, whichever channel retrieved it.

        A unit only BM25 or the symbol channel found used to reach the ranker with no cosine at all: the one signal
        that could confirm or refute the lexical match was missing exactly where it mattered. Ranks stay those of
        the channels' own top lists (0 = not retrieved by that channel).
        """
        from dataclasses import replace as _replace  # noqa: PLC0415

        if not pool or data.vectors is None or self.encoder is None:
            return pool
        rows = [data.position(c.doc_id) for c in pool]
        qvec = self._query_vector(data.snapshot.snapshot_id, query, route=route)
        # The primary's whole-snapshot scores (one exact pass, as the dense channel does): every candidate's cosine
        # and its rank over the snapshot, which the encoder-agreement features compare with the second encoder's.
        whole = exact_search(qvec.reshape(1, -1), data.vectors)[0]
        primary = whole[rows]
        primary_rank = whole.shape[0] - np.searchsorted(np.sort(whole), primary, side="right") + 1
        secondary = None
        full_rank = None
        if data.aux_vectors is not None and self.aux_encoder is not None:
            vector2 = self._aux_query_vector(data.snapshot.snapshot_id, query, route=route)
            with stage("dense2"):
                # One exact pass over the snapshot gives every candidate its cosine *and* its whole-snapshot rank
                # under the second encoder (rank = 1 + documents scoring strictly higher), whichever channel found it.
                everything = exact_search(vector2.reshape(1, -1), data.aux_vectors)[0]
                secondary = everything[rows]
                ordered = np.sort(everything)
                full_rank = everything.shape[0] - np.searchsorted(ordered, secondary, side="right") + 1
        out = []
        for i, c in enumerate(pool):
            fields: dict[str, float | int] = {}
            if c.dense_score != c.dense_score:
                fields["dense_score"] = float(primary[i])
            if secondary is not None:
                fields["aux_score"] = float(secondary[i])
                fields["aux_full_rank"] = int(full_rank[i])  # type: ignore[index]
                fields["dense_full_rank"] = int(primary_rank[i])
            out.append(_replace(c, **fields) if fields else c)
        return out

    @property
    def generic_alpha(self) -> float | None:
        """The generic route's fusion weight (`rank.generic.alpha`), or `None` when none has been tuned."""
        value = self.config.get("rank.generic.alpha", None)
        return None if value in (None, "") else float(value)

    def fused_order(
        self, data: SnapshotData, query: str, pool: Sequence[Any], *, route: str, counters: Counters | None = None
    ) -> list[str]:
        """Weighted dense + lexical fusion over the pool (spec 02 §4 stage 8, generic route).

        A lexical-only candidate has no cosine yet; it gets its exact one from the snapshot's matrix (one dot
        product each), so both channels are compared on the same documents. Ties fall to the content hash (INV-4).
        """
        from dataclasses import replace as _replace  # noqa: PLC0415

        from acis.rank.fusion import identifier_query, mentions, weighted_fusion  # noqa: PLC0415

        alpha = self.generic_alpha
        assert alpha is not None
        missing = [c for c in pool if c.dense_score != c.dense_score]
        if missing and data.vectors is not None:
            vector = self._query_vector(data.snapshot.snapshot_id, query, route=route)
            index = {doc_id: i for i, doc_id in enumerate(data.doc_ids)}
            rows = data.vectors[[index[c.doc_id] for c in missing]]
            filled = {c.doc_id: float(s) for c, s in zip(missing, rows @ vector, strict=True)}
            pool = [_replace(c, dense_score=filled[c.doc_id]) if c.doc_id in filled else c for c in pool]
        beta = float(self.config.get("rank.generic.symbol_beta", 0.0) or 0.0)
        defined: dict[str, float] = {}
        if beta and data.symbols is not None:
            names = data.symbols.defined_in_query(query)
            if names:
                defined = {c.doc_id: data.symbols.defined_score(data.position(c.doc_id), names) for c in pool}
                if counters is not None and any(defined.values()):
                    counters.incr("fusion.defined_symbols")
        with stage("fusion"):
            aux_weight = float(self.config.get("rank.generic.aux_weight", 0.0) or 0.0)
            if aux_weight and not any(c.aux_score == c.aux_score for c in pool):
                # Configured, but this snapshot has no second-encoder vectors: counted, never silent (INV-7).
                (counters or self.counters).incr("fusion.aux_missing")
            fused = weighted_fusion(pool, alpha=alpha, beta=beta, defined=defined, aux_weight=aux_weight)
            order = [doc for doc, _ in self._stable_order(data, list(fused.items()))]
            # Spec 02 §6b: for an identifier-like query, units containing that exact identifier come first (still
            # in fused order), then the rest. A shape rule, not a name list (INV-15); nothing matches -> unchanged.
            identifier = identifier_query(query)
            if identifier is not None:
                exact = [d for d in order if mentions(identifier, data.text_of(d))]
                if exact:
                    sink = counters or self.counters
                    sink.incr("fusion.identifier_first")
                    sink.incr("fusion.identifier_matches", len(exact))
                    order = exact + [d for d in order if d not in set(exact)]
            return order

    def pool_features(self, data: SnapshotData, query: str, pool: Sequence[Any]) -> np.ndarray:
        """Appendix A features for a candidate pool — the matrix the ranker reads (spec 02 §4 stage 7).

        Public so an offline experiment builds exactly the features serving builds.
        """
        from acis.features.query import bridge as bridge_features  # noqa: PLC0415
        from acis.features.query import extract as query_features  # noqa: PLC0415
        from acis.rank import candidates as cand  # noqa: PLC0415

        qf = query_features(query)
        bridge = {c.doc_id: bridge_features(qf, data.features_of(c.doc_id)) for c in pool}
        doc_meta = {
            c.doc_id: {
                "n_tokens": data.features_of(c.doc_id).n_tokens,
                "parse_ok": data.features_of(c.doc_id).parse_ok,
                "dup_cluster_size": data.duplicate_count(c.doc_id),
            }
            for c in pool
        }
        return cand.feature_matrix(
            pool,
            bridge=bridge,
            query_tokens=qf.n_tokens,
            doc_meta=doc_meta,
            symbols=self._symbol_features(data, query, pool),
        )

    @staticmethod
    def _symbol_features(data: SnapshotData, query: str, pool: Sequence[Any]) -> dict[str, dict[str, float]]:
        """Per candidate: how many of the query's symbols it contains, their share, and their summed IDF.

        Empty (so NaN in the matrix) when the query names no symbol: "nothing to compare" is not "no match".
        """
        from acis.lexical.symbols import query_symbols  # noqa: PLC0415

        symbols = query_symbols(query)
        if not symbols or data.symbols is None:
            return {}
        idf = {s: data.symbols.idf(s) for s in symbols}
        out = {}
        for c in pool:
            own = data.symbols.symbols_of[data.position(c.doc_id)]
            matched = [s for s in symbols if s in own]
            out[c.doc_id] = {
                "symbol_hits": float(len(matched)),
                "symbol_coverage": len(matched) / len(symbols),
                "symbol_idf": float(sum(idf[s] for s in matched)),
            }
        return out

    def _hybrid_ranking(
        self,
        data: SnapshotData,
        query: str,
        *,
        route: str,
        want: int,
        counters: Counters,
        strict: bool,
    ) -> list[tuple[str, float]]:
        """Dense ∪ lexical → features → ranker (statements) or weighted fusion (everything else) → dense tail.

        The fallback chain is explicit and counted (INV-7): with no ranker loaded, weighted fusion if it has been
        tuned and the dense order otherwise (spec 02 §4 stage 8). Every step down increments a counter and appears
        in `degradations`, and strict mode refuses all of them.
        """

        dense_order, pool = self.candidate_pool(data, query, route=route, want=want, counters=counters, strict=strict)
        if not pool:
            return dense_order[:want]

        ranker = self.ranker
        head: list[str]
        if route != "statement_like":
            # R-Q2 (spec 10 §2): the learned ranker is trained on problem statements; on anything else it is
            # measurably worse than the frozen dense order, so the generic route is ranked by weighted fusion of the
            # two channels that already ran. Counted, and **not** a degradation: it is the specified ranking for a
            # non-statement query, not a fallback from something better.
            if ranker is not None:
                counters.incr(f"route.{route}.ltr_skipped")
            if self.generic_alpha is not None:
                counters.incr("fusion.applied")
                head = self.fused_order(data, query, pool, route=route, counters=counters)
            else:
                # The specified ranking for this route is missing, so the dense order serves it: a fallback,
                # counted and refused by strict mode like every other (INV-7).
                counters.incr("fusion.untuned")
                degradation("fusion_untuned", "dense order", strict=strict, counters=counters)
                head = []
        elif ranker is not None:
            pool = self._with_prf(data, query, route=route, pool=pool)
            with stage("features"):
                matrix = self.pool_features(data, query, pool)
            with stage("ranker"):
                head, abstained = ranker.rerank([c.doc_id for c in pool], matrix)
            if abstained:
                # Thin evidence: the ranker's contract is "the dense order stands". Not fusion (G2 measured
                # rank fusion harmful on statements), and not a fallback: abstaining is the designed behaviour.
                counters.incr("ltr.abstained")
                head = []
            else:
                counters.incr("ltr.applied")
        elif self.generic_alpha is not None:
            degradation("ltr_unavailable", "weighted fusion", strict=strict, counters=counters)
            head = self.fused_order(data, query, pool, route=route, counters=counters)
        else:
            degradation("ltr_unavailable", "dense order", strict=strict, counters=counters)
            head = []

        # The head is re-scored by position, then the dense tail follows it to `want` (spec 02 §4 stage 9). The
        # tail's own order is the dense one, which is why it is taken from `dense_order` rather than recomputed.
        seen = set(head)
        tail = [doc for doc, _ in dense_order if doc not in seen]
        ordered = head + tail
        floor = -float(len(ordered))
        return [(doc, floor + float(len(ordered) - i)) for i, doc in enumerate(ordered[:want])]

    def _with_prf(self, data: SnapshotData, query: str, *, route: str, pool: Sequence[Any]) -> list[Any]:
        """Add the second-pass score and rank as **features**, never as the ranking itself (spec 02 §4 stage 5).

        Feedback helps when the top of the first pass is right and hurts when it is wrong, so it is off until
        gate G4 says otherwise — and even then it feeds the ranker rather than replacing it, which is the whole
        reason the second-pass score is a feature in the first place.
        """
        from dataclasses import replace as _replace  # noqa: PLC0415

        settings = self.config.section("retrieve").get("prf", {})
        if not settings or not settings.get("enabled") or data.vectors is None:
            return list(pool)

        from acis.rank.prf import DEFAULT_ALPHA, DEFAULT_M, expand  # noqa: PLC0415

        vector = self._query_vector(data.snapshot.snapshot_id, query, route=route)
        index_of = {doc_id: i for i, doc_id in enumerate(data.doc_ids)}
        first = [(index_of[c.doc_id], float(c.dense_score)) for c in pool if c.dense_score == c.dense_score]
        first.sort(key=lambda item: -item[1])
        _, scores = expand(
            vector,
            data.vectors,
            first,
            m=int(settings.get("m", DEFAULT_M)),
            alpha=float(settings.get("alpha", DEFAULT_ALPHA)),
        )
        if scores.size == 0:
            return list(pool)

        order = {
            doc: rank
            for rank, doc in enumerate(
                sorted((c.doc_id for c in pool), key=lambda d: -float(scores[index_of[d]])), start=1
            )
        }
        return [_replace(c, prf_score=float(scores[index_of[c.doc_id]]), prf_rank=order[c.doc_id]) for c in pool]

    @staticmethod
    def _rrf_order(pool: Sequence[Any]) -> list[str]:
        """Reciprocal-rank fusion over the union — parameter-free, and the honest default until G2 tunes weights.

        Ties are broken by dense rank, so fusion can never reorder two documents it has no reason to separate.
        """
        from acis.rank.candidates import reciprocal_rank  # noqa: PLC0415

        return [
            c.doc_id
            for c in sorted(
                pool,
                key=lambda c: (
                    -(reciprocal_rank(c.dense_rank) + reciprocal_rank(c.lexical_rank)),
                    c.dense_rank or 1 << 30,
                    c.doc_id,
                ),
            )
        ]

    def _lexical_ranking(self, data: SnapshotData, query: str, k: int) -> list[tuple[str, float]]:
        if data.lexical is None:
            raise NotReady("the lexical channel is not available for this snapshot")
        with stage("bm25"):
            return data.lexical.search_one(lexical_view(query), k)

    # -- the batch surface the adapter uses -----------------------------------------------------------------------
    def search_batch(
        self,
        snap: Snapshot,
        ids: Sequence[str],
        texts: Sequence[str],
        *,
        top_k: int,
        restrict_to: Mapping[str, Sequence[str]] | None = None,
        strict: bool = False,
    ) -> dict[str, list[tuple[str, float]]]:
        """Rank every query independently. Batch composition never changes a single ranking (INV-3)."""
        if len(ids) != len(texts):
            raise InvalidInput("ids and texts must have the same length", n_ids=len(ids), n_texts=len(texts))
        data = self.snapshot_data(snap)
        if data.snapshot.state != "VALID":
            raise SnapshotInvalid(f"snapshot {data.snapshot.snapshot_id} is {data.snapshot.state}")
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        out: dict[str, list[tuple[str, float]]] = {}
        for qid, text in zip(ids, texts, strict=True):
            allowed = restrict_to.get(str(qid)) if restrict_to else None
            if allowed is None:
                out[str(qid)] = self._rank_one(data, text, top_k=top_k, strict=strict)
                continue
            # A reranking task hands us the candidate set: rank the whole corpus, then keep the candidates, so the
            # result still holds min(top_k, |candidates|) entries rather than whatever survived an early cut.
            allowed_set = {str(a) for a in allowed}
            full = self._rank_one(data, text, top_k=data.size, strict=strict)
            out[str(qid)] = [(d, s) for d, s in full if d in allowed_set][:top_k]
        return out

    def _rank_one(
        self,
        data: SnapshotData,
        text: str,
        *,
        top_k: int,
        strict: bool,
        counters: Counters | None = None,
        route: str | None = None,
        channel: str = "auto",
    ) -> list[tuple[str, float]]:
        query, _truncated = self.normalise_query(text)
        # The route chooses the instruction the query is encoded with (spec 02 §4, stage 2). `search` has already
        # computed it; the batch surface has not, and routing is a pure function of the query, so it is safe here.
        # A request that names a channel gets that channel; "auto" means the configured one.
        mode = channel if channel != "auto" else str(self.config.get("run.channel", "auto"))
        if route is None:
            # On the hybrid channel the route picks the ranker or the generic path, so it always matters there.
            route = self.route(query) if mode == "hybrid" else self.encoding_route(query)
        dense_available, _ = self._channels()
        sink = counters if counters is not None else self.counters

        want = min(top_k, data.size)
        ordered = False
        if mode in ("auto", "dense") and dense_available:
            # The dense channel returns its top-`want` already in final order, so nothing below re-sorts it.
            ranking = self._dense_ranking(data, query, route=route, k=want)
            ordered = True
        elif mode == "dense":
            raise NotReady("dense channel requested but no encoder is configured")
        elif mode == "lexical":
            ranking = self._lexical_ranking(data, query, data.size)
        elif mode == "hybrid":
            ranking = self._hybrid_ranking(data, query, route=route, want=want, counters=sink, strict=strict)
            ordered = True
        else:
            degradation("dense_unavailable", "lexical-only ranking", strict=strict, counters=sink)
            ranking = self._lexical_ranking(data, query, data.size)

        if len(ranking) < want:
            ranking = self._extend_with_unretrieved(data, ranking, want)
            ordered = False
        return (ranking if ordered else self._stable_order(data, ranking))[:top_k]

    @staticmethod
    def _extend_with_unretrieved(
        data: SnapshotData, ranking: Sequence[tuple[str, float]], want: int
    ) -> list[tuple[str, float]]:
        """Pad a short ranking so INV-10 still returns `min(top_k, N)` entries.

        A channel can return fewer hits than asked for — BM25 with no matching term is the ordinary case. The
        documents it did not retrieve are all equally unranked, so they all get **one** score below every retrieved
        document and `_stable_order` sorts them by content hash. Giving them descending scores in corpus order
        instead would make the tail depend on how the corpus happens to be arranged, which is exactly what parity
        P5 forbids.
        """
        seen = {d for d, _ in ranking}
        floor = min((s for _, s in ranking), default=0.0) - 1.0
        wanted = max(0, want - len(ranking))
        if not wanted:
            return list(ranking)
        # All padding entries share one score, so the order among them is the content-hash tie-break alone —
        # `nsmallest` on that key gives the same answer as sorting the whole corpus, without doing so per query.
        chosen = heapq.nsmallest(
            wanted,
            (doc_id for doc_id in data.doc_ids if doc_id not in seen),
            key=lambda doc_id: (data.hash_of.get(doc_id, doc_id), data.ordinal.get(doc_id, 1 << 30)),
        )
        return [*ranking, *((doc_id, floor) for doc_id in chosen)]

    # -- the single-query surface ---------------------------------------------------------------------------------
    def search(self, req: SearchRequest) -> SearchResponse:
        """One query against one pinned snapshot, with evidence re-read from the content store (INV-1)."""
        started = time.perf_counter()
        # Resolution decides where the snapshot comes from — the store for a repository, memory for the P0 batch
        # surface — so the "is there anything to search" check belongs there, not here.
        data = self._resolve_snapshot(req)
        if data.snapshot.state != "VALID" and not req.allow_partial:
            raise SnapshotInvalid(f"snapshot {data.snapshot.snapshot_id} is {data.snapshot.state} (INV-9)")

        stages: dict[str, float] = {}
        token = _STAGES.set(stages)
        t_norm = time.perf_counter()
        query, truncated = self.normalise_query(req.query)
        decision = self.route_decision(query)
        route = cast(Route, decision.route)
        top_k = max(1, min(int(req.top_k), MAX_TOP_K))
        strict = bool(self.config.strict)

        t_rank = time.perf_counter()
        # Per-request counters: `SearchResponse.degradations` must describe *this* request, not everything the
        # process has degraded since start-up. They are merged into the engine's totals for the run manifest.
        request_counters = Counters()
        ranked = self._rank_one(
            data, req.query, top_k=top_k, strict=strict, counters=request_counters, route=route, channel=str(req.mode)
        )
        for name, count in request_counters.snapshot().items():
            self.counters.incr(name, count)
        for event in request_counters.degradations():
            self.counters.note(event)
        signals = self._explain_signals(data, query, route, [d for d, _ in ranked]) if req.explain else {}
        hits = [
            Hit(
                rank=rank,
                score=float(score),
                unit=data.unit_of(doc_id),
                source=data.text_of(doc_id),  # INV-1: re-read by hash
                # Never publish the corpus ordinal: on this corpus `ordinal < 5000` is an exact train-partition
                # detector, and a Phase 4 feature builder reading `hit.signals` would learn it (docs/DESIGN_RULES.md).
                signals={"channel_score": float(score), **signals.get(doc_id, {})},
            )
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        ]
        explanation = self._explain_order(str(req.mode), request_counters) if req.explain else {}
        if req.explain:
            explanation["route_decision"] = decision.as_dict()
        z = self.confidence_signal(data, query, [d for d, _ in ranked], route=route) if ranked else float("nan")
        confidence, no_strong_match, confidence_facts = self._confidence(z, route)
        exact = request_counters.get("fusion.identifier_matches")
        if exact:
            # A verbatim identifier match is a fact about the text, not an estimate: it is never a weak match. The
            # calibrated level is kept, with the basis saying the calibration was not fitted on such queries.
            from acis.rank.fusion import identifier_query  # noqa: PLC0415

            no_strong_match = False
            confidence_facts["basis"] = (
                f"{exact} unit(s) contain `{identifier_query(query)}` verbatim and are listed first (exact match). "
                "The calibrated estimate was fitted on natural-language queries, not identifiers: "
                + str(confidence_facts["basis"])
            )
            confidence_facts["identifier_matches"] = exact
        if req.explain:
            from acis.engine.routing import categorize  # noqa: PLC0415
            from acis.lexical.symbols import query_symbols  # noqa: PLC0415

            explanation["confidence"] = confidence_facts
            explanation["confidence_basis"] = confidence_facts["basis"]
            symbols = query_symbols(query)
            explanation["query_symbols"] = list(symbols)
            explanation["category"] = categorize(query, route, symbols, weak_match=no_strong_match)
            seen = request_counters.snapshot()
            explanation["candidates"] = {
                k.removeprefix("pool."): int(v) for k, v in seen.items() if k.startswith("pool.")
            }
            explanation["channels"] = [
                name
                for name, key in (
                    ("dense", "from_dense"),
                    ("bm25", "from_bm25"),
                    ("dense2", "from_dense2"),
                    ("symbols", "from_symbols"),
                )
                if explanation["candidates"].get(key)
            ]
            explanation["second_encoder"] = getattr(self.aux_encoder, "name", None) if self.aux_encoder_name else None
            explanation["hit_symbols"] = self.matched_symbols(data, query, [d for d, _ in ranked])
        _STAGES.reset(token)
        total_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(
            snapshot=SnapshotRef(
                id=data.snapshot.snapshot_id,
                version=data.snapshot.version_id,
                complete=data.snapshot.complete,
                missing_channels=data.missing,
                repo_id=data.snapshot.repo_id,
                n_units=data.size,
            ),
            results=hits,
            confidence=confidence,
            no_strong_match=no_strong_match,
            route=route,
            interpreted_intent={"query_truncated": truncated},
            timings_ms={
                "normalize": round((t_rank - t_norm) * 1000, 3),
                "rank": round((time.perf_counter() - t_rank) * 1000, 3),
                "total": round(total_ms, 3),
                **{f"stage.{k}": round(v, 3) for k, v in stages.items()},
            },
            degradations=request_counters.degradations(),
            explanation=explanation,
        )

    def _explain_signals(
        self, data: SnapshotData, query: str, route: str, doc_ids: Sequence[str]
    ) -> dict[str, dict[str, float]]:
        """Per hit, what each channel said: dense cosine and rank over the whole snapshot, BM25 score and rank.

        Computed for display only, after the ranking is fixed — nothing here can change an order. A channel that
        did not retrieve a document simply has no entry for it; nothing is filled in.
        """
        out: dict[str, dict[str, float]] = {d: {} for d in doc_ids}
        if data.vectors is not None and self.encoder is not None:
            vector = self._query_vector(data.snapshot.snapshot_id, query, route=route)
            scores = exact_search(vector.reshape(1, -1), data.vectors)[0]
            index = {d: i for i, d in enumerate(data.doc_ids)}
            for d in doc_ids:
                s = float(scores[index[d]])
                out[d]["similarity"] = s
                out[d]["dense_rank"] = float(int((scores > s).sum()) + 1)
        retrieve = self.config.section("retrieve")
        dense_k = int(retrieve.get("dense_k", 100))
        lexical_k = int(retrieve.get("lexical_k", 30))
        for d in doc_ids:
            rank = out[d].get("dense_rank")
            out[d]["found_dense"] = float(rank is not None and rank <= dense_k)
        if data.lexical is not None and "lexical" not in data.missing:
            for rank, (d, s) in enumerate(self._lexical_ranking(data, query, max(100, lexical_k)), start=1):
                if d in out:
                    out[d]["bm25"] = float(s)
                    out[d]["bm25_rank"] = float(rank)
                    out[d]["found_bm25"] = float(rank <= lexical_k)
        from acis.lexical.symbols import query_symbols  # noqa: PLC0415

        symbols = query_symbols(query)
        symbol_k = int(retrieve.get("symbol_k", 0))
        if symbols and data.symbols is not None:
            ranked = {d: r for r, (d, _) in enumerate(data.symbols.search(symbols, max(symbol_k, 1)), start=1)}
            for d in doc_ids:
                hits = data.symbols.hits(data.position(d), symbols)
                if hits:
                    out[d]["symbol_hits"] = float(hits)
                out[d]["found_symbols"] = float(symbol_k > 0 and d in ranked)
        aux_k = int(retrieve.get("aux_k", 0))
        if data.aux_vectors is not None and self.aux_encoder is not None:
            vector2 = self._aux_query_vector(data.snapshot.snapshot_id, query, route=route)
            scores2 = exact_search(vector2.reshape(1, -1), data.aux_vectors)[0]
            for d in doc_ids:
                s2 = float(scores2[data.position(d)])
                out[d]["similarity2"] = s2
                out[d]["dense2_rank"] = float(int((scores2 > s2).sum()) + 1)
                out[d]["found_dense2"] = float(aux_k > 0 and out[d]["dense2_rank"] <= aux_k)
        for d in doc_ids:
            out[d]["n_channels"] = float(sum(v for k, v in out[d].items() if k.startswith("found_")))
        return out

    def matched_symbols(self, data: SnapshotData, query: str, doc_ids: Sequence[str]) -> list[list[str]]:
        """Per hit, the query's code-shaped symbols the unit contains — display only, after the order is fixed."""
        from acis.lexical.symbols import query_symbols  # noqa: PLC0415

        symbols = query_symbols(query)
        if not symbols or data.symbols is None:
            return [[] for _ in doc_ids]
        return [[s for s in symbols if s in data.symbols.symbols_of[data.position(d)]] for d in doc_ids]

    def _explain_order(self, requested: str, counters: Counters) -> dict[str, Any]:
        """Which stage produced the final order, from this request's own counters."""
        channel = requested if requested != "auto" else str(self.config.get("run.channel", "auto"))
        seen = counters.snapshot()
        # `ordering` is the machine-readable name of the stage that produced the order; `ordered_by` says it in words.
        second = getattr(self.aux_encoder, "name", None) if self.aux_encoder_name else None
        dense_desc = f"dense ({getattr(self.encoder, 'name', 'encoder')} + {second})" if second else "dense"
        aux_weight = float(self.config.get("rank.generic.aux_weight", 0.0) or 0.0)
        mix = f", second-encoder share {aux_weight:g}" if (second and aux_weight) else ""
        if channel == "hybrid":
            alpha = self.generic_alpha
            if seen.get("ltr.applied"):
                ordering, ordered_by = "ltr", f"learned ranker over {dense_desc} + BM25 + identifier candidates"
            elif seen.get("ltr.abstained"):
                ordering, ordered_by = "ltr_abstained", "dense (the ranker abstained: too little evidence fired)"
            elif seen.get("fusion.identifier_first"):
                ordering = "identifier_first"
                ordered_by = (
                    f"exact identifier matches first, then weighted fusion of {dense_desc} + BM25 "
                    f"(α = {alpha:g}{mix}; generic route)"
                )
            elif seen.get("fusion.applied"):
                ordering = "fusion"
                ordered_by = (
                    f"weighted fusion of {dense_desc} + BM25 (α = {alpha:g}{mix}; generic route, the ranker stays off)"
                )
            elif seen.get("fusion.untuned"):
                ordering, ordered_by = "dense", "dense (generic route; no fusion weight has been tuned)"
            else:
                ordering, ordered_by = "dense", "dense (no ranker loaded)"
        elif channel == "lexical":
            ordering, ordered_by = "lexical", "BM25"
        else:
            ordering, ordered_by = "dense", "dense"
        return {
            "channel": channel,
            "ordering": ordering,
            "ordered_by": ordered_by,
            "encoder": getattr(self.encoder, "name", "none"),
            "second_encoder": second,
            "generic_alpha": self.generic_alpha,
            "generic_aux_weight": aux_weight if second else None,
        }

    def _resolve_snapshot(self, req: SearchRequest) -> SnapshotData:
        """Resolve the request's version **once**, and pin it for the whole request (INV-2).

        A request that names a repository is answered from the store (Track B1); one that does not is answered
        from the in-memory snapshots the P0 batch surface builds. The two never mix: a repository id is how a
        caller says "this is a versioned corpus".
        """
        if req.repo_id not in ("-", "", None):
            return self.open_version(req.repo_id, req.version)

        if req.version not in ("latest", "", None):
            for data in self._snapshots.values():
                if req.version in (data.snapshot.version_id, data.snapshot.snapshot_id):
                    return data
            raise NotFound(f"no snapshot for version {req.version!r}")
        if not self._snapshots:
            raise NotReady("no snapshot has been built yet")
        return next(reversed(list(self._snapshots.values())))

    @property
    def calibration(self) -> Any:
        """The confidence calibration named by `confidence.calibration`, loaded once; `None` when not configured."""
        if self._calibration_loaded:
            return self._calibration
        self._calibration_loaded = True
        path = str(self.config.get("confidence.calibration", "") or "")
        if path:
            from acis.core.paths import acis_root  # noqa: PLC0415
            from acis.rank.confidence import Calibration  # noqa: PLC0415

            target = Path(path) if Path(path).is_absolute() else acis_root() / path
            if target.is_file():
                self._calibration = Calibration.load(target)
        return self._calibration

    def confidence_signal(self, data: SnapshotData, query: str, served: Sequence[str], *, route: str) -> float:
        """z of the served #1 against this query's own top-100 dense cosines (`acis.rank.confidence`); with a second
        encoder, the mean of its z and the primary's."""
        from acis.rank.confidence import CROWD, z_top1  # noqa: PLC0415

        if not served or data.vectors is None or self.encoder is None:
            return float("nan")

        def z_under(scores: np.ndarray) -> float:
            crowd = min(CROWD, int(scores.shape[0]))
            top = -np.partition(-scores, crowd - 1)[:crowd]
            return z_top1(np.sort(top)[::-1], float(scores[data.position(served[0])]))

        vector = self._query_vector(data.snapshot.snapshot_id, query, route=route)
        z = z_under(exact_search(vector.reshape(1, -1), data.vectors)[0])
        if data.aux_vectors is not None and self.aux_encoder is not None:
            # With a second encoder the signal is the mean of the two z's: on the 5,000 dev queries it predicts a
            # correct #1 with AUC 0.876 against 0.832 for the primary alone (scripts/bench/calibrate_confidence.py).
            vector2 = self._aux_query_vector(data.snapshot.snapshot_id, query, route=route)
            z2 = z_under(exact_search(vector2.reshape(1, -1), data.aux_vectors)[0])
            if z2 == z2 and z == z:
                return (z + z2) / 2.0
        return z

    def _confidence(self, z: float, route: str) -> tuple[Confidence, bool, dict[str, Any]]:
        """A report, never a ranking input (spec 02 §4 stage 11): calibrated P(served #1 relevant) → band."""
        from acis.rank.confidence import band  # noqa: PLC0415

        calibration = self.calibration
        if calibration is None:
            return "low", False, {"calibrated": False, "z": z, "basis": "confidence is not calibrated"}
        p = calibration.probability(route, z)
        level = cast(Confidence, band(p))
        basis = (
            f"estimated P(top result relevant) = {p:.2f} for the {route} route, from z = {z:.2f} "
            f"(isotonic fit on {calibration.source(route)}) [ledger:{calibration.ledger_run_id}]"
        )
        return level, level == "low", {"calibrated": True, "z": z, "p": p, "basis": basis}

    # -- introspection ---------------------------------------------------------------------------------------------
    def diagnostics(self, repo_id: str | None = None) -> Diagnostics:
        from acis.cli.doctor import collect  # noqa: PLC0415 — hardware probing is not on the hot path

        hardware = collect(with_gemm=False)["hardware"]
        return Diagnostics(
            config_hash=self.config_hash,
            model_fingerprint=self.model_fingerprint,
            numeric_profile=self.config.numeric_profile,
            hardware=hardware,
            tier=str(collect(with_gemm=False)["tier"]),
            degradations=dict(self.counters.snapshot()),
            snapshots=tuple(
                {
                    "id": d.snapshot.snapshot_id,
                    "version": d.snapshot.version_id,
                    "units": d.size,
                    "state": d.snapshot.state,
                    "missing_channels": list(d.missing),
                }
                for d in self._snapshots.values()
                if repo_id is None or d.snapshot.repo_id == repo_id
            ),
            agent_calls=0,  # INV-13: there is no agent on this path
        )

    def evaluate(self, spec: EvalSpec) -> EvalReport:
        """CLI-only, never exposed on the network API (docs/spec/06 §1)."""
        from acis.eval import ladder  # noqa: PLC0415

        return ladder.run_spec(self, spec)

    # -- Track B surface -------------------------------------------------------------------------------------------
    # `ingest`, `index`, `update_version`, `compare_versions`, `rollback` and `open_version` come from
    # `VersionedEngineMixin` (Track B1, `acis.engine.versions`).
    def search_version(self, repo_id: str, version: str, query: str, **kw: object) -> SearchResponse:
        return self.search(SearchRequest(query=query, repo_id=repo_id, version=version, **kw))  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_CONFIG",
    "MAX_QUERY_CHARS",
    "MAX_TOP_K",
    "AcisEngine",
    "SnapshotData",
    "pooled_query_vector",
    "query_encoder_text",
    "query_encoder_texts",
]
