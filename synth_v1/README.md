# synth_v1 — synthetic Persian pretraining corpus (recipe)

Goal: **40–60B tokens of high-quality Persian educational text** for IbnSina 3B midtraining, released under
**Apache-2.0**. The text is **generated in Persian** (never translated from English); English is used only for
seed-topic inventories, because the underlying knowledge is densest in English. Universal knowledge only.

## Domain selection

Every subdomain in `taxonomy.yaml` is scored 1–5 on four criteria (recorded per subdomain):
**universal** across countries / **dense in English** sources / **scarce in Persian** / **non-political**.
Domains: STEM foundations (biggest slice, 40%), medicine science-half (16%), engineering & applied science
(14%), academic explainer prose in the FineWeb-Edu register (13%), reasoning/CoT in Persian (17%).
Excluded outright: law, tax, civics, history, politics, religion, current events, business/social norms,
health policy, country-specific practice — anything jurisdiction-bound or contested.

## Method

1. **Seed topics** (English): the teacher model itself expands curriculum lists / textbook-TOC-style /
   arXiv-category-style topic inventories per subdomain (`prompts/topics_user.md`). Nothing is scraped.
2. **Style anchors** (Persian): ~200 real high-scoring passages (educational register, classifier
   `edu_score >= 3.0`, low news probability, 500–1500 chars) sampled from our scored Persian web corpus by
   `sample_anchors.py`. Each generation prompt carries one anchor for **register only**; reusing its content
   is forbidden. Anchor texts live only on the pipeline VM (`/data/synth_v1/anchors/`) and are never
   committed or redistributed.
3. **Generation** (`prompts/gen_system.md` + `gen_user.md`): one standalone Persian document per call —
   doc types: explainer article, lecture notes, worked problems, Q&A, glossary; 800–2500 tokens each.
   Teachers on Vertex (`location="global"`): `gemini-3.7-flash` primary, `gemini-3.5-flash-lite` for bulk,
   thinking disabled, temperature 0.9, 3x oversample. Translationese is explicitly forbidden with concrete
   bad→good examples; a scope guard bans excluded domains even as examples.
   Literal/faithful rendering of canonical proofs and precise definitions is allowed as a labelled minority
   (<10% of tokens).
4. **Judge filter** (`prompts/judge.md`): `gemini-3.7-flash` (calibration lesson from sft_v2: Flash-Lite
   misses confidently-wrong math), 0–10 on correctness, natural Persian, translationese-free, informational
   density; overall capped by the weakest critical criterion; keep the top third. Keep threshold is
   calibrated at STOP SY-B against human spot-checks. Reasoning docs additionally get a machine
   verification pass over their «جواب: …» answer lines.
5. **Dedup + decontamination**: MinHash (same params as `pipeline/p1_dedup.py`) plus decontamination against
   ParsiNLU, Khayyam, PersianMedQA, PerCoR and TARAZ (exact question match + normalized 13-gram overlap,
   same method as sft_v2 assembly).
6. **Output**: pretraining-format parquet shards + manifest (counts, teacher split, prompt hashes, license
   tags), Apache-2.0.

## Files

- `taxonomy.yaml` — domains, subdomains, weights, 4-criteria scores, per-subdomain seed-topic strategy,
  doc-type mixes, length targets, teachers + verified model ids + prices, judge config, budget ladder.
- `prompts/gen_system.md`, `prompts/gen_user.md` — generation (style-anchor slot, anti-translationese rules).
- `prompts/topics_user.md` — teacher-side seed-topic expansion.
- `prompts/judge.md` — judge rubric, JSON output.
- `sample_anchors.py` — anchor sampler (runs on the pipeline VM; output stays there).
- `pilot.py` — SY-A pilot: topics → ~50 docs/domain (Flash + labelled Flash-Lite slice) → judge → readable
  per-domain `.md` tree + `_pilot_report.md` with measured $/B projections. Resumable; hard spend cap.

## House patterns reused from sft_v2 / pipeline (and reused again by the full pipeline)

- token/cost **Ledger** with per-model usage (sft_v2/gen.py), extended with thought-token tracking;
- **bounded worker pool over a queue** (a flat gather over thousands of coroutines starves completion);
- **tolerant JSONL readers + append-only done-sets** → any stage survives VM reboots and resumes;
- judge **calibration procedure** and keep-selection logic (sft_v2/judge.py + CALIBRATION.md);
- decontamination method + eval-set fetching (sft_v2/fetch_evalsets.py), MinHash from pipeline/p1_dedup.py,
  `normalize_fa` normalization at assembly time.

## Bulk pipeline (SY-A approved 2026-08-31; target 3-5B kept tokens, STOP SY-B at ~1B)

Wave-based driver on the pipeline VM, everything resumable and reboot-safe:

- `bulk_common.py` — shared plumbing: batch-priced cost ledger, Vertex batch submit/poll/parse (per-line
  `id` matching), prompt building, tolerant JSONL IO, online async fallback runner.
- `gen_bulk.py` — per wave: teacher-expanded topic inventory (once), scenario sampling
  (subdomain x doc-type x length x aspect x variation), Vertex **batch** jobs chunked per teacher
  (routing: Flash for reasoning_cot + medicine_science, Flash-Lite elsewhere), poll, parse to docs.jsonl.
  Wave 1 is deliberately small (~9k docs) as an end-to-end validation; later waves default to 120k.
- `judge_bulk.py` — judges EVERY candidate with gemini-3.7-flash via batch (chunked), no sampling shortcuts.
- `assemble_shards.py` — the "kept" gate: auto-checks, top third per domain with overall >= 7,
  MinHash dedup (5-gram, 14x8, persistent index), decontamination against the eval sets
  (normalized 13-gram overlap), normalize_fa, parquet shard -> `gs://.../synth_v1/shards/` + Apache-2.0
  manifest; updates kept-token state; exits 42 at STOP SY-B (~1B kept), 43 at the $20k hard stop,
  44 on a keep-rate floor breach (< 0.15).
- `run_synth.sh` — serial-wave driver (gen -> judge -> assemble), detached via setsid nohup; markers
  SYB_READY / BUDGET_STOP / KEEPRATE_STOP; ledger at `/data/synth_v1/bulk/ledger.json`, per-wave reports
  under `/data/synth_v1/bulk/reports/`.
- `calibration/` — 22 planted-flaw judge-calibration cases + runner (see calibration/README.md).

## Budget ladder (also in taxonomy.yaml)

Mandate estimate $8–15k for 40–60B tokens; **notify at each $5k; hard stop $20k; pilot capped at ~$20.**
The pilot report (`_pilot_report.md`) carries the measured $/B projection — read it before approving bulk:
at 2026-08-31 Vertex list prices the mandate estimate does not cover 40–60B at 3x oversample on Flash-class
models, so the SY-A decision includes teacher mix, batch mode (-50%), oversample rate and/or a smaller
token target.

## STOP gates

- **SY-A** (now): taxonomy + prompts + pilot docs + cost report → Sina signs off scope, teacher mix, budget.
- **SY-B**: judge calibration on pilot + first 1% of bulk reviewed → keep threshold frozen, bulk continues.
