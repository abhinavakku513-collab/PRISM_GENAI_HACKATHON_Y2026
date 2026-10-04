# Changelog

## v1.0.0 — 2026-10-02 (final submission)

**P0 retrieval accuracy.** Qwen3-Embedding-0.6B joins gte-modernbert-base as a second dense encoder (ADR-0009): its
top 300 enter the 500-candidate union, its cosine and whole-corpus rank feed the LambdaRank ranker (refit on the new
pools), and it carries 0.75 of the generic route's dense term. DEV (APPS train split, all 5,000 queries, out of fold):
NDCG@10 74.13 → 86.70, MRR@10 70.93 → 84.19 `[ledger:dev-f93aeb265be7]`; then five gte/Qwen agreement features in the
ranker: **87.07 / 84.72**, Recall@100 98.76 `[ledger:dev-839fba9f81a6]`; candidate recall 97.16 % → 99.38 %. The
architecture is frozen at this point; the official TEST run is the final evaluation. Measured and rejected: a third encoder (granite-small-r2, +0.11), a 700-candidate union,
pseudo-relevance feedback, the statement-only query view, pool-context features, larger rankers, the xendcg
objective, the ranker on every route. MTEB Mode A and the engine rank DEV queries identically (300/300 top-100 lists).

**Official TEST result (AppsRetrieval test split, MTEB, cold, Mode A):** NDCG@10 **78.283**, MRR@10 **74.393**,
evaluation_time 27,849 s `[ledger:rc-bd2285335a86]`; `verify-submission` PASS including the held-out re-score.

**Correctness and honesty fixes.** The second encoder follows the primary's cache policy, so a cold official run stays
cold; a strict run refuses a configured channel that cannot load; a cold run encodes each query once instead of twice;
`rank_dense2` is the whole-corpus rank for every candidate; the CLI, API and UI name both encoders and the fusion mix;
Qwen's forward batches are capped at 4,096 tokens (the cold run was OOM-killed at the default); the submission
verifier allows exactly MTEB's own 5-decimal rounding when re-scoring against the JSON.

**Confidence report (after the official run; no ranking change).** The calibrated confidence was still fitted on
the pre-Qwen pipeline, so short questions read "weak match" even when the top result was right. The signal is now the
mean of both encoders' z (AUC for a correct #1: 0.876 vs 0.832), recalibrated on the served pipeline — APPS dev out of
fold and live CodeSearchNet-Python queries `[ledger:dev-528ef76633cd]`; held out, "high" means the top result was
right 92–94 % of the time. The UI's ranking card now describes the stage that actually ran.

**Submission tooling.** `make models` (pinned download + verification), `make rc-smoke` (fills the sealed dataset cache
through MTEB's own loader, then proves the offline load), per-encoder prebuilt vector packs (`acis demo-index
--encoder`), Mode A as the primary official surface, P0 evidence published in `docs/evidence/p0/`.

**Repository.** Beginner README; internal AI-assistant tooling and planning notes removed from the published tree;
development reports kept in `docs/history/`; design rules published as `docs/DESIGN_RULES.md`.

## Earlier development

Phase reports and the previous acceptance report are in `docs/history/`. The previous served pipeline (gte + BM25 +
exact symbols + LambdaRank) measured NDCG@10 74.13 · MRR@10 70.93 on DEV `[ledger:dev-299a3010be5a]`.
