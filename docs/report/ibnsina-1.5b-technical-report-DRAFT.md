# IbnSina-1.5B: an open Persian-first large language model trained from scratch

**Technical report — DRAFT skeleton (2026-08-30).** This report is written in the first person; everything not marked `[[TBD Sept 1: …]]` is verified against the repository, build manifests or run logs (Appendix C). The two evaluation tables are placeholders to be filled on Sept 1.

Sina Meraji · ORCID 0009-0002-8028-1932 · github.com/sinameraji

Project home: `github.com/ibnsina-llm` · `huggingface.co/ibnsina-llm`

---

## Abstract

IbnSina-1.5B is a 1.48-billion-parameter, Llama-compatible large language model pretrained from scratch on 36.7 B tokens — a planned 46 B-token run completed at 80 % after two optimisation instabilities (Section 5) — 63 % of them Persian, and instruction-tuned on a 51 k-conversation Persian recipe. Unlike most Persian models, which adapt an English-first base, it never knew English first. I release the weights (Apache-2.0) as standard GGUF for llama.cpp, ollama and LM Studio, together with the complete recipe: a 39-source corpus pipeline (28 of the sources are used in this build; the rest are excluded by licence or absent) with per-source licence gating, a 32 k Persian-dominant tokenizer that needs 21–29 % fewer tokens on Persian than the Qwen3.5 and Gemma 3 tokenizers, a training stack built on nanochat's loop with a drop-in Llama-3-shaped model, and a published supervised-fine-tuning taxonomy with every prompt in the repository. I report the model on ParsiNLU and PersianMedQA, and place PersianMedQA results for the 2026 frontier (Claude, GPT, Gemini, Kimi, GLM, DeepSeek, Qwen, Grok, Hunyuan) alongside it: on the same 5,235 questions the strongest frontier systems score 88–89 % (Gemini 3.1 Pro 88.8 %, Claude Fable 5 88.1 %, Claude Opus 5 87.0 %), where IbnSina-1.5B reaches `[[TBD Sept 1: IbnSina PersianMedQA]]`. I also document a mid-run stability episode caused by attention-logit growth in a QK-norm-free architecture, how it was diagnosed, and how it was contained.

## 1. Introduction

**Goals.** (1) *Persian-first*: Persian is the majority language of pretraining and of the SFT conversations (the SFT mixture also carries an English multiple-choice auxiliary that teaches answer formatting, Section 6.6), so fluency and register come from the base model, not from adaptation. (2) *Open*: weights, tokenizer, corpus pipeline, licence table, mix recipe, SFT prompts and evaluation code are public; the assembled corpus itself is not redistributed but can be rebuilt from the original sources. (3) *Runs everywhere on day one*: the architecture is Llama-compatible so the checkpoint exports to standard GGUF with no custom inference code.

**Naming.** The family is *IbnSina*; models publish as `ibnsina-llm/ibnsina-1.5b` (base + chat). A 360 M research pilot (`ibnsina-pilot-360m`, nanochat-native, no GGUF) preceded it and is reported here only as context.

**Scope of this report.** Sections 2–4 describe data, tokenizer and pretraining; Section 5 the stability episode; Section 6 SFT; Section 7 evaluation; Section 8 release. Appendices give hyperparameters, compute and cost, and the provenance of every number.

## 2. Pretraining data

### 2.1 Pipeline

Five deterministic, rerunnable stages (`pipeline/`):

| stage | script | what it does |
|---|---|---|
| raw → raw_filtered | `filter_parallel.py` | per-shard domain blocklist on the web crawls; audited counts in `_manifest.json` |
| Phase 0: extract / normalise | `p0_run.py` | readers for the pipeline's 39 sources (28 used in this build) (PDF via `pdftotext` with OCR fallback, Wikipedia via `wikiextractor`, Stack Overflow dumps, parquet row-group sharding); common doc schema `{id, text, source, url, lang, meta}`; rejects kept with a reason |
| Phase 1: dedup | `p1_dedup.py` | exact (xxh64) + MinHash near-dedup (datatrove, 5-grams, 14×8 buckets, 64-bit) with a *min-priority* cluster rule so curated documents are never dropped in favour of web copies |
| Phase 2: quality | `p2_quality.py` | a Persian *educational-value* classifier: Gemini labels 10 k documents on a FineWeb-Edu-style 0–5 rubric, a fastText regressor is distilled from them (accuracy 0.74, within-one-point 0.91) and scores every web document; a news-probability head down-weights news |
| Phase 3: mix | `p3_mix.py` | deterministic hash sampling into slices with per-source licence gating (`licenses.json`), epochs, ≈1 G-char parquet shards (132 row groups × 2000 docs, sized for 8-GPU striding), a 0.5 ‰ × 5 validation split and a `mix_manifest.json` recording exactly what went in |

Dedup removed 28.5 % of Persian web documents (170.6 M → 122.0 M; 3.2 M exact + 45.4 M MinHash duplicates); mC4-fa lost 39 % and FineWeb-2-fa 32 % of their documents to overlap with CulturaX. After scoring, the web slice keeps every document with educational score ≥ 1.5 (21.5 B tokens) plus a sampled fill from the [1.0, 1.5) band (non-news at p ≈ 0.61, news at p ≈ 0.11); documents below 0.75 are dropped.

### 2.2 The `train_v1_1_open` mix

The public mix recipe is `docs/data/mix_manifest_public_v1_1.json` (category level: per-slice share, estimated tokens, documents, number of sources, licence classes and epoch ranges; 28 of the pipeline's 39 sources are used in this build, 46.35 B tokens). Token counts below are the build's estimate with a 64 k BPE proxy tokenizer; the run consumed 70,000 × 524,288 ≈ 36.7 B tokens of the final 32 k tokenizer (≈0.8 epoch — the run was shortened to 70,000 of a planned 88,000 steps, Sections 4.2 and 5).

| slice | tokens (est.) | share | documents | sources | main sources (licence) |
|---|---:|---:|---:|---:|---|
| Persian web | 29.40 B | 63.4 % | 29.4 M | 3 | CulturaX-fa (ODC-BY/CC0), mC4-fa (ODC-BY), FineWeb-2-fa (ODC-By), classifier-filtered, news down-weighted |
| English educational | 7.16 B | 15.5 % | 6.7 M | 2 | FineWeb-Edu (ODC-By), OpenStax (CC-BY), Project Gutenberg / pre-1929 books, peS2o (ODC-By) |
| Code: Python + TypeScript | 2.39 B | 5.2 % | 2.8 M | 2 | StarCoderData (permissive per-file, The Stack v1) |
| Code: other | 2.39 B | 5.2 % | 4.0 M | 3 | Stack Overflow (CC-BY-SA), Persian-NLP GitHub repositories (per-repo licences) |
| Math & textbooks (×3 epochs) | 2.39 B | 5.2 % | 1.1 M | 3 | OpenWebMath (ODC-By); v1.1 adds 429 Iranian school textbooks from chap.sch.ir (14.5 M tokens × 3 epochs; proprietary/curated class) |
| Wikipedia (×1–4 epochs) | 1.43 B | 3.1 % | 4.3 M | 2 | fa-Wikipedia × 4 epochs (0.95 B), en-Wikipedia sampled (CC-BY-SA) |
| Parallel fa–en | 0.96 B | 2.1 % | 0.9 M | 8 | OPUS: OPUS-100, GlobalVoices, HPLT, WikiMatrix, XLEnt, CCMatrix/CCAligned, OpenSubtitles (see licence notes) |
| Persian literature (×3–4 epochs) | 0.24 B | 0.5 % | 0.6 M | 5 | Ganjoor classical poetry, fa-Wikisource, a curator-provided history volume — 2 licence classes (public-domain, proprietary/curated) |
| **total** | **46.35 B** | | **49.7 M** | **28** | |

The literature slice was planned at 5 % of the mix; only 0.24 B tokens of licence-clean text existed after deduplication, and the shortfall was reallocated to Persian web. Persian sources (web, Wikipedia, literature, half of parallel) make up roughly 68 % of tokens; the rest is English and code, included so the model can read documentation, code and the English half of translation tasks.

**Licence policy.** Included: public-domain, permissive and share-alike sources. Curated material without an explicit open licence (official school textbooks, curator-provided volumes) and one research-use parallel corpus (OPUS OpenSubtitles) were admitted per source by curator decision on 2026-08-28 and are recorded as such; NC/ND-licensed sources were excluded. No source text is redistributed. The parallel-data slice includes OPUS OpenSubtitles (research-use terms), retained for training only by curator decision; roughly 0.3 B of 46 B tokens. No source text is redistributed. Excluded on licence grounds: TED2020 (CC-BY-NC-ND), MIZAN and TEP (non-commercial), NCERT, Kanoon exam booklets, CodeParrot, Matina (CC-BY-NC-ND), FLORES-200 (evaluation only). The per-source decisions are in `pipeline/licenses.json`. Some gated sources could not be obtained in time for this build and are absent.

**Decontamination.** Evaluation data is never in the training mix (ParsiNLU is CC-BY-NC-SA and evaluation-only). The SFT set is additionally decontaminated against every evaluation set (Section 6.5).

### 2.3 What is and is not public

Public: pipeline code, the per-source licence table, the category-level mix manifest (`docs/data/mix_manifest_public_v1_1.json`), and every generation/judging prompt for SFT. Not public: the assembled corpus (it can be rebuilt from the cited sources with the pipeline).

## 3. Tokenizer

A 32,768-token byte-level BPE trained with nanochat's tokenizer tooling on a ≈10 GB stratified sample dominated by Persian. Version 2 replaces the pre-tokenizer with Llama-3's regex so that the exported GGUF uses `tokenizer.ggml.pre = llama-bpe` and needs no custom llama.cpp code; fertility is identical to v1 within ±0.005 tokens/word. Special tokens follow nanochat's chat scaffolding (`<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, tool-call tokens).

Fertility on held-out text (the validation split, never trained on), tokens per whitespace word — lower is better:

| text | words | IbnSina 32k | Qwen3.5 (248 k vocab) | Gemma 3 (262 k vocab) |
|---|---:|---:|---:|---:|
| Persian web | 3.93 M | **1.288** | 1.728 | 1.635 |
| Persian Wikipedia | 0.80 M | **1.447** | 2.026 | 1.825 |
| Persian literature / poetry | 0.24 M | **1.370** | 1.766 | 1.671 |
| Parallel fa–en | 3.19 M | **1.477** | 1.679 | 1.626 |
| English edu | 3.25 M | 1.490 | 1.351 | 1.330 |
| English Wikipedia | 1.68 M | 1.630 | 1.462 | 1.435 |
| Math (EN) | 3.22 M | 1.745 | 1.666 | 1.650 |
| Code (Stack Overflow) | 2.70 M | 2.058 | 1.995 | 2.093 |
| Code (py/ts) | 1.86 M | 3.203 | 3.135 | 3.363 |

On Persian web text the 32 k vocabulary needs 25 % fewer tokens than Qwen3.5's 248 k vocabulary and 21 % fewer than Gemma 3's 262 k (6.97 bytes/token vs 5.20 and 5.49); on Persian Wikipedia the gap is 29 % and 21 %. The price is paid in English (≈10 % more tokens than the large vocabularies) and code, which is the intended trade for a Persian-first model with a small, fast embedding table.

## 4. Model and pretraining setup

### 4.1 Architecture

Llama-3-shaped, chosen for ecosystem reach over the last few percent of training efficiency (ruling of 2026-08-29): `nanochat_patches/nanochat/llama.py` implements exact llama.cpp `llama` semantics inside nanochat's training loop.

| | IbnSina-1.5B | pilot-360m (nanochat arch, context only) |
|---|---|---|
| layers / width | 28 / 2048 | 18 / 1152 |
| attention | 16 heads, 4 KV heads (GQA), head dim 128, full causal context | 9 / 9 |
| FFN | SwiGLU, hidden 6144 | ReLU², 4× |
| norms / positions | RMSNorm (ε = 1e-5, learnable gains), RoPE θ = 500 000 | nanochat defaults (incl. QK-norm) |
| head | untied, no biases | |
| vocab / context | 32,768 / 2048 | 32,768 / 2048 |
| parameters | ≈1.48 B (0.13 B embeddings) | 0.36 B |

Relative to nanochat's own GPT the Llama variant gives up value embeddings, residual lambdas, QK-norm, logit soft-capping and smear/backout; the toy A/B at d12 (300 steps, 39 M tokens, same data and tokenizer) measured val bpb 1.072 for the Llama arch vs 1.102 for the nanochat arch (−2.7 %) at +4 % tokens/s, so no efficiency price was visible at that scale. The absence of QK-norm turned out to matter at 1.5 B scale (Section 5).

### 4.2 Optimisation and schedule

nanochat's loop (commit `92d63d4`): Muon for the 2-D matrices (momentum ramped linearly 0.85 → 0.97 over the first 400 steps, then constant at 0.97, warming down to 0.90 during the LR warmdown — the same override applies if an early warmdown is pulled; Newton–Schulz 5 steps, NorMuon-style second-moment normalisation), AdamW for embeddings, unembedding and norm gains; weight decay on cosine decay to zero; bf16 with on-the-fly tokenisation; batch 524,288 tokens (8 GPUs × 8 sequences × 2048 × 4 micro-steps); planned 88,000 iterations = 46.1 B tokens (≈31 tokens per parameter), completed at a shortened 70,000 iterations = 36.7 B tokens (≈25 tokens per parameter). **The learning-rate schedule as it actually ran:** 500 warm-up steps; constant at the original LRs to 20,000; from the 20,000 resume all group LRs ×0.5 after an attention-logit-driven instability (Section 5); the scheduled linear warmdown began at 52,800 (`warmdown_ratio` 0.4 of 88,000); after a second instability at 61,106 the run resumed from 60,000 with LRs ×0.35 and a compressed linear warmdown to the shortened 70,000 total (`warmdown_ratio` 0.18 of 70,000, LR-continuous at the splice: multiplier 0.79 → `final_lr_frac`). The Muon momentum warmdown mirrored the LR warmdown throughout.

Full arguments are in `training/config_1p5b.md` and the run's recorded argument file (Appendix A).

### 4.3 Hardware and throughput

One `a3-highgpu-8g` spot instance (8 × H100 80 GB, NVLink) on Google Cloud with a 24-hour spot lifetime, zone-rotating launcher, checkpoints every 1000 steps mirrored to object storage and automatic resume after preemption. Measured: 432–438 k tokens/s, 54–55 % bf16 MFU, 1.21–1.23 s per 524 k-token step at device batch 8 (device batch 16 does not fit in 80 GB). Base pretraining wall-clock ≈ 30 h of stepping plus restarts; total compute and cost in Appendix B.

**fp8 trial abandoned.** A 200-step fp8 vs bf16 A/B was stable, but the fp8 run diverged after warm-up at full learning rate (loss minimum 3.18 at step 1000 rising to 4.66 by 4200; val bpb 0.871 → 1.153). The run was rolled back to bf16 from checkpoint 2000. Lesson: an fp8 trial must cover ≥ 1500 full-LR steps; bf16 is the default for release runs.

## 5. Training stability episode

*An engineering narrative, kept because the diagnosis changed twice and the false leads are as useful as the answer.*

**Symptom.** Twenty-seven thousand clean steps (val bpb 0.6118 at 26 k, improving ≈0.002 per 2 k steps), then at step 27,008 the logged training loss went 2.55 → 7.45 within 20 steps and needed ≈500 steps to recover. Rolling back to the clean step-27,000 checkpoint and resuming reproduced a jump within 8–35 steps on every attempt.

**Ruled out, and how.**
- *Data.* The token ids of the batches being processed at the moment of a jump were dumped from every rank and decoded: ordinary Persian and English web text, sane id ranges and BOS counts.
- *Hardware.* GPU Xid, ECC and NVLink error counters were clean; the VM was stopped and started to land on a different host and the jump recurred within 20 steps.
- *Communication.* nanochat's fused Muon/AdamW optimiser overlaps NCCL `reduce_scatter`/`all_gather` with compute. A genuine hygiene issue was found (the gather's input tensor is a local dropped before `wait()`), and the plateau shape — flat while updates were skipped, "repaired" by the next optimiser step — fitted a torn gather. But keeping the tensors alive did not stop it, running every collective synchronously did not stop it, and a direct test at the moment of a plateau found **0 of 255 parameter tensors differing across ranks**. The weights were bit-identical everywhere. (The hygiene fix is worth upstreaming; it is explicitly *not* the cause.)

**Root cause.** Attention-logit growth. Measured on validation batches, the maximum |q·k/√d| over layers, heads and positions was 53 at step 10 k, 51 at 20 k and **102 at 27 k**, concentrated in layer 3 (99.9th percentile 67) and layer 0. At that magnitude the bf16 softmax saturates and the gradient through it becomes spiky: the all-rank gradient norm, normally ≈0.23, spiked to 3, 10, 35, 55 and 90 within a single 60-step window. Each spike is a consistent, oversized update on every rank — a jump of 2–8 nats in one step, flat while updates are withheld, and undone within a handful of normal updates. nanochat's own GPT carries QK-norm precisely to prevent this; the Llama-faithful architecture chosen for GGUF compatibility does not, and the optimiser's aggressive updates let the q/k norms drift upward until the softmax saturated.

**Mitigation.** (1) Rollback to step 20,000, where logits were still ≈51 (2.4 h of compute discarded rather than taming a saturated state). (2) All learning rates halved for the remainder. (3) A gradient-norm guard before every optimiser step: the update is skipped when the all-rank gradient norm exceeds 6× its running average; plus a loss guard as backstop (skip when the all-rank step loss exceeds the running average by more than 1 nat, at most 12 consecutive skips). (4) Collectives kept synchronous (≈2 % cost) for the rest of this run. (5) Checkpoints from the discarded trajectory quarantined so that a preemption could not rewind onto them.

**Outcome.** On the new trajectory, val bpb was 0.5909 at 22 k and 0.5893 at 26 k — already better than the old trajectory's best (0.6118 at 26 k), because halving the learning rate removed a large amount of weight noise immediately — then 0.5522 at 60 k and **0.5221 at the final 70,000**. The gradient guard skipped ≈213 anomalous updates between the 20 k resume and a second instability at step 61,106, at which point the run was resumed from the 60,000 checkpoint with learning rates ×0.35 and a compressed warmdown to a shortened 70,000 steps (the "ship it" rung of the pre-agreed escape ladder). After that resume: one gradient-guard skip in ≈10,000 steps, zero loss trips, and attention logits back at the 20 k baseline (max ≈51, p99.9 ≈27) through 70,000. The residue sanity check confirms the clean finish: re-measured on a fixed 20 M-token evaluation set, val bpb is 0.5502 at checkpoint 60,000 and 0.5206 at the final 70,000 (consistent with the training-time values of 0.5522 and 0.5221, which use a larger eval sample); greedy generations at both checkpoints are fluent, coherent Persian and English across factual, biographical and procedural prompts, with one repetition loop on an under-specified date prompt at 70 k — ordinary base-model behaviour (an apparent token-separator artefact in the probe's decoded output came from the probe script, not the model). The logged validation trajectory across the whole post-20 k run improves monotonically (0.5909 → 0.5206) with no discontinuity at any resume point. Conclusion: the 27 k–61 k turbulence left no measurable residue; the shipped weights come from the 20 k rollback line with the guards clean throughout.

**Lessons.** QK-norm (or attention-logit soft-capping) is not optional at this scale with Muon-class optimisers — the 3 B model will use an architecture that has it and that llama.cpp can still export (Qwen3-style). Pre-update gradient-norm and loss guards with forensics (per-rank losses, batch dumps, cross-rank consistency) are now standard in this training stack. Measure attention logits at every checkpoint; a slow creep precedes the blow-up by thousands of steps. The tooling ships with the repository so the story comes with reusable parts: the gradient-norm/loss guard with its forensics (`training/nanochat_patches/spike_guard_patch.py`), the attention-logit probe (`training/attn_probe.py`) and the post-run sanity check (`training/sanity_check.py`). The async-collective lifetime hardening of the optimiser — the side finding, not the cause — has been submitted upstream to nanochat ([PR #843](https://github.com/karpathy/nanochat/pull/843)).

## 6. Supervised fine-tuning (`sft_v2`)

### 6.1 Taxonomy

Seventeen categories in three groups, every target reached exactly; 51,400 kept conversations (50,997 train + 502 held-out `sft_eval`, after one decontamination drop):

| group | category | kept |
|---|---|---:|
| bulk | general_instruction (incl. 1,944 SmolTalk multi-turn conversations translated by Gemini and judged like everything else) | 24,999 |
| bulk | reasoning_math_cot_fa (step-by-step Persian, konkur-style) | 5,000 |
| bulk | persian_native (FarsInstruct permissive subsets recast as natural conversation) | 5,000 |
| bulk | formatting_control (lists, tables, JSON, length limits, one-line, letter/yes-no/three-way answers) | 5,000 |
| differentiator | persian_writing (official and personal letters, e-mail, proofreading, classical-form poems) | 2,000 |
| differentiator | translation_fa_en (both directions, idioms, Finglish → Persian script) | 2,000 |
| differentiator | iran_knowledge (literature, history, geography, calendar conversions; grounded on corpus passages) | 2,000 |
| differentiator | everyday_advice | 1,500 |
| differentiator | taarof_register | 500 |
| control | respect_and_contested (symmetric respect; "supporters say / critics say") | 600 |
| control | multiturn_repair | 1,000 |
| control | uncertainty («نمی‌دانم» over invention) | 500 |
| control | refusals · language_discipline | 300 · 300 |
| control | toolcall (calculator, Jalali↔Gregorian conversion, search — executed during judging) | 500 |
| control | anti_sycophancy | 200 |
| control | identity (human-written canonical set, expanded verbatim) | 100 |

Every category has 5–10 hand-written gold seed conversations (136 total, reviewed before generation) and a rubric; each generation prompt samples a scenario from persona × register × length × turns × region axes and shows three rotating seeds.

### 6.2 Generation

Two teachers, same prompts: Gemini 2.5 Flash-Lite for bulk categories and Gemini 2.5 Flash for differentiators and controls (Vertex AI), plus Kimi K3 via OpenRouter on differentiators and controls within a $30 cap. Three candidates per target (two for general_instruction), user and assistant turns generated together, multi-turn where the category calls for it. Teacher split of the kept set: Flash-Lite 38,055 · Flash 11,086 · Flash (SmolTalk translation) 1,944 · Kimi K3 314 · human 100.

### 6.3 Judge and calibration

Gemini 2.5 Flash scores each candidate 1–10 on correctness, natural Persian (no translationese), instruction adherence, register match, and the respect/safety rules, after automatic checks (assistant-turn language ID, repetition loops, length sanity, teacher self-reference, and *execution* of tool calls with a Jalali↔Gregorian converter and a safe evaluator). Calibration on 100 identical candidates showed Flash-Lite missing wrong arithmetic (scored 9 and 7 where Flash scored 1 and 1; Pearson 0.72, ≥2-point disagreement 15 %), so judging runs on Flash regardless of cost. Selection is best-first up to each category's target with overall score ≥ 7. Candidates judged: ≈141 k; kept: 51,400; best-vs-worst sibling pairs persisted for a later DPO pass: 13,453.

### 6.4 Identity and behaviour policy

The identity set is human-written (ten canonical answers, expanded verbatim to 100 rows). The behaviour policy the data teaches: natural, register-matching Persian; honest uncertainty; medical and legal questions answered as information with a pointer to a professional and Iranian emergency numbers where acute; and *symmetric respect* — it declines to mock or insult any person or group on any side while answering factual and theological questions normally, recounting documented history honestly and mapping contested political questions without crowning a side.

### 6.5 Assembly and decontamination

Persian normalisation identical to pretraining (digits, kaf/yeh, ZWNJ), then exact-question and normalised 13-gram overlap decontamination against ParsiNLU (19,493 strings), PerCoR (96,217), TARAZ (6,447), a GhazalBench proxy (Hafez ghazals 1–100 couplets, 1,680) and PersianMedQA (39,137). Output in nanochat chat format with tokenizer-v2 special tokens; manifest with per-category counts, teacher split, seed and prompt hashes, licence tags (generated data Apache-2.0; persian_native sources MIT / CC-BY-SA / Apache-2.0).

### 6.6 The SFT training run

85 steps (1.52 minutes on 8×H100) over a 100,997-row mixture: the 50,997 Persian `sft/v2` conversations plus a 50,000-row English MMLU auxiliary set kept from the v1 recipe, whose job is teaching option-letter answering for the categorical evaluations (a later pass may shrink this auxiliary). Optimizer momentum is carried over from the base checkpoint with learning rates reset. Held-out SFT-eval bpb: 0.2825.

**Auxiliary-mixture ablation.** Two SFT passes from the same base checkpoint differ only in the English MMLU auxiliary: 50,000 rows (aux50k, the shipped model) vs 8,000 rows (aux8k). Under the letter protocol, aux50k scores 31.81 / 33.65 / 50.73 / 26.76 (ParsiNLU-MC / entailment / QQP / PersianMedQA) against aux8k's 29.52 / 33.47 / 45.35 / 25.33 — the full auxiliary wins on every task. The interpretation is that the English multiple-choice auxiliary teaches the option-letter answer format the categorical evaluations require, and does not measurably displace Persian capability at this scale. The final ship decision awaits a qualitative side-by-side; the quantitative verdict is unambiguous.

### 6.7 Cost

≈$228 in teacher and judge calls (Gemini Flash $148, Flash-Lite $49, Kimi K3 $30), plus $5 for the SmolTalk translation; CPU only, in parallel with pretraining.

## 7. Evaluation

### 7.1 Protocols

*ParsiNLU* multiple-choice (1,050 questions, three categories), textual entailment and question paraphrase, in nanochat's categorical format (answer letter). *PersianMedQA* (Ranjbar Kalahroodi et al., 2025; CC-BY-4.0): 5,235 test questions across 23 medical fields, evaluated with the paper's zero-shot prompt at temperature 0, answer = option number. Validation bits-per-byte on the held-out split of the mix. English CORE was not run (Persian-first model; `--core-metric-every=-1`).

### 7.2 Table A — IbnSina on Persian benchmarks `[[TBD Sept 1]]`

| model | val bpb | ParsiNLU MC | ParsiNLU entailment | ParsiNLU QQP | PersianMedQA |
|---|---:|---:|---:|---:|---:|
| ibnsina-1.5b base | 0.5221 | — | — | — | — |
| ibnsina-1.5b chat (sft_v2) | 0.2825* | 31.81 % | 33.65 % | 50.73 % | 26.76 % |
| ibnsina-pilot-360m chat (sft_v1; context) | 0.428* | 29.2 % | 36.5 % | 43.5 % | — |
| chance | — | 25 % | 33.3 % | 50 % | 25 % |

\* chat rows report bpb on the held-out SFT conversations (a different distribution from the pretraining validation split); categorical evaluations use the letter-answer protocol (`eval_fa`) and were run on the chat models only, since letter-format answers come from the SFT. Sample sizes: MC 1,050 (math & logic 28.0 %, common knowledge 36.86 %, literature 30.57 %), entailment 1,673, QQP 1,916, PersianMedQA 5,235.

Read honestly: at 1.5 B parameters and 36.7 B tokens the model is at or barely above chance on knowledge-heavy multiple choice — entailment and QQP are at chance, PersianMedQA is 1.8 points above it — which is in line with the small-model band under this harness protocol (Llama-3.2-3B 30.91 %, Phi-4-14B 30.45 % on PersianMedQA). Its best category is Persian common knowledge (36.86 %). MCQ knowledge is not where a model this size differentiates; the model's case rests on Persian fluency, register and tokenizer efficiency, which these numbers do not measure.

### 7.3 Table B — PersianMedQA, IbnSina alongside the 2026 frontier

IbnSina appears in Table B not as a competitor to 400B-class systems but to place an open, laptop-scale, from-scratch Persian model on the same ruler. Same protocol for every row: the paper's zero-shot prompt, temperature 0, answer = option number, 5,235 test questions with a complete option set. Models are each vendor's newest flagship on the run date (2026-08-30; ids from the OpenRouter catalogue, Gemini from Vertex AI). "Truncated / unparsed" = no option number in the reply (truncated thinking or refusal); such rows count as wrong in the first accuracy column and are excluded in the second. Thinking-mandatory endpoints were retried once on unparsed rows with a 4,096-token cap³.

| model (vendor, release) | id | accuracy, all 5,235 | accuracy, answered | truncated / unparsed | protocol notes |
|---|---|---:|---:|---:|---|
| Gemini 3.1 Pro (preview) (Google, 2026-06) | `gemini-3.1-pro-preview` | **88.77 %** | 88.82 % | 3 (0.1 %) | Vertex AI, thinking on |
| Gemini 3.7 Flash (Google, 2026-08) | `gemini-3.7-flash` | **88.65 %** | 88.65 % | 0 (0.0 %) | Vertex AI, thinking on |
| Grok 4.6 (xAI, 2026-08) | `x-ai/grok-4.6` | **88.18 %** | 88.18 % | 0 (0.0 %) | thinking on (mandatory) |
| Claude Fable 5 (Anthropic, 2026-06)¹ | `anthropic/claude-fable-5` | **88.08 %** | 89.15 % | 63 (1.2 %) | thinking on (mandatory) |
| Claude Opus 5 (Anthropic, 2026-07) | `anthropic/claude-opus-5` | **86.99 %** | 86.99 % | 0 (0.0 %) | answer-only |
| Gemini 2.5 Pro (prior generation) (Google, 2025-06) | `gemini-2.5-pro` | **86.40 %** | 86.45 % | 3 (0.1 %) | Vertex AI, thinking on |
| GPT-5.6 Terra (OpenAI, 2026-07) | `openai/gpt-5.6-terra` | **84.53 %** | 84.53 % | 0 (0.0 %) | answer-only |
| Qwen3.8 2.4T-A95B (Alibaba, 2026-08) | `qwen/qwen3.8-2.4t-a95b` | **84.13 %** | 89.40 % | 309 (5.9 %) | thinking on (mandatory) |
| GLM-5.3 (Z.ai, 2026-08)² | `z-ai/glm-5.3` | **83.97 %** | 87.57 % | 215 (4.1 %) | thinking on (mandatory) |
| Kimi K3 (Moonshot AI, 2026-07) | `moonshotai/kimi-k3` | **83.19 %** | 83.19 % | 0 (0.0 %) | answer-only |
| Qwen3.8 Max (Alibaba, 2026-08) | `qwen/qwen3.8-max` | **82.14 %** | 90.34 % | 475 (9.1 %) | thinking on (mandatory) |
| Hunyuan Hy4 (preview) (Tencent, 2026-08) | `tencent/hy4-preview` | **76.18 %** | 76.44 % | 18 (0.3 %) | answer-only |
| DeepSeek V4 Pro (0813) (DeepSeek, 2026-08) | `deepseek/deepseek-v4-pro-0813` | **72.51 %** | 72.53 % | 1 (0.0 %) | answer-only |
| **IbnSina-1.5B chat** | this work | **22.64 %** | 24.45 % | 388 (7.4 %) | 1.48 B parameters, from scratch; local llama.cpp, same retry policy |
| *PersianMedQA paper (2025): GPT-4.1 83.1, Gemini 2.5 Flash 82.4, Claude 3.7 75.2, Llama-3.1-405B 67.0, Dorna2-8B 34.9, human 75* | | | | | |

¹ Claude Fable 5 was also run through Claude Code subagents with 219 questions per prompt (not the paper protocol): 88.90 % on 5,234/5,235 answered — not comparable, reported for transparency.
² An earlier pass used the superseded id `z-ai/glm-5` (73.49 %); the table reports GLM-5.3, the current flagship at run time.
³ Thinking-mandatory endpoints were retried once on unparsed rows with a 4,096-token cap; remaining truncations are counted as wrong in the first accuracy column and excluded in the second.

Reading the table: the strongest 2026 systems land at 88–89 % on all 5,235 questions (Gemini 3.1 Pro 88.77, Gemini 3.7 Flash 88.65, Grok 4.6 88.18, Claude Fable 5 88.08), against 83.1 % for GPT-4.1 in the 2025 paper and a 75 % human reference; the two accuracy columns separate knowledge from answer discipline — Qwen3.8 Max answers 90.3 % of what it finishes but truncates 9 % of the time. IbnSina-1.5B lands at 22.64 % — 1.8 points above the 20.5 % random baseline for this option mix, in line with the small-model band under this protocol (Llama 3.2 1B 24.85 %, Gemma 3 1B 27.43 %), and consistent with the thesis that MCQ knowledge is not where a 1.5 B model differentiates.

### 7.4 Table B-2 — PersianMedQA per field (top 8 models, fields with n ≥ 100)

| Field (n) | Gemini 3.1 Pro | Gemini 3.7 Flash | Grok 4.6 | Claude Fable 5 | Claude Opus 5 | Gemini 2.5 Pro | GPT-5.6 Terra | Qwen3.8 2.4T-A95B |
|---|---|---|---|---|---|---|---|---|
| کودکان (631) | 91.6 | 91.9 | 90.6 | 90.2 | 89.7 | 88.3 | 87.3 | 85.7 |
| جراحی (619) | 83.0 | 82.2 | 82.6 | 84.0 | 81.4 | 81.9 | 78.8 | 77.7 |
| زنان (440) | 88.0 | 86.8 | 87.0 | 88.0 | 85.2 | 85.5 | 82.5 | 80.0 |
| عفونی (270) | 88.1 | 89.6 | 87.4 | 85.2 | 87.4 | 84.8 | 81.9 | 83.3 |
| پاتولوژی (243) | 95.1 | 94.2 | 95.1 | 93.0 | 93.8 | 94.2 | 93.4 | 93.0 |
| نورولوژی (219) | 91.3 | 90.4 | 90.4 | 90.0 | 87.7 | 87.7 | 86.3 | 89.5 |
| ارتوپدی (214) | 90.2 | 89.3 | 89.7 | 91.1 | 88.3 | 86.0 | 83.2 | 85.5 |
| روانپزشکی (190) | 91.6 | 92.1 | 90.5 | 93.2 | 91.1 | 88.4 | 84.7 | 87.4 |
| غدد (184) | 88.6 | 90.8 | 89.1 | 88.0 | 85.9 | 84.8 | 85.3 | 83.2 |
| فارماکولوژی (184) | 94.0 | 92.9 | 92.9 | 83.7 | 92.9 | 91.3 | 91.3 | 91.8 |
| اورولوژی (182) | 84.1 | 83.5 | 81.3 | 83.5 | 83.5 | 83.0 | 80.8 | 76.9 |
| قلب (180) | 85.6 | 86.7 | 86.7 | 87.8 | 86.1 | 85.0 | 81.7 | 82.8 |
| رادیولوژی (175) | 94.3 | 93.7 | 93.7 | 92.6 | 92.6 | 94.3 | 90.3 | 89.7 |
| ریه (175) | 86.9 | 86.3 | 86.9 | 86.3 | 85.1 | 82.9 | 82.9 | 81.7 |
| نفرولوژی (175) | 89.7 | 90.9 | 89.7 | 86.3 | 86.3 | 88.6 | 83.4 | 86.3 |
| پوست (169) | 92.3 | 92.9 | 92.3 | 92.9 | 92.9 | 89.9 | 90.5 | 89.9 |

Accuracy in percent on all questions of the field (truncations counted wrong). The comparison chart: `[[TBD Sept 1]]`.

## 8. Release and usage

- **Weights:** `huggingface.co/ibnsina-llm/ibnsina-1.5b` — a 5.08 GiB bundle: GGUF F16 (2.97 GB), Q8_0 (1.58 GB, near-lossless), Q4_K_M (0.90 GB, laptop/phone), an ollama `Modelfile`, the model card, tokenizer files and the evaluation results. `[[TBD Sept 1: safetensors mirror yes/no]]`
- **Export path:** `training/export_gguf.py` writes architecture `llama` with a GPT-2-style BPE vocabulary derived from the tokenizer's byte ranks, `tokenizer.ggml.pre = llama-bpe`, the chat template in metadata, and the Q/K permutation llama.cpp expects. Validated end to end on a toy model before the 1.5 B run: tokenizer parity 6/6 probes, F16 greedy decoding identical to PyTorch for 24/24 tokens (server and Apple Metal), Q8_0 identical, Q4_K_M runs.
- **Chat format:** `<|user_start|>…<|user_end|><|assistant_start|>…<|assistant_end|>`, BOS added by the runtime, a system message folded into the first user turn. Tool calls (calculator, date conversion, search) use nanochat's format and work only in a host that executes them; llama.cpp and ollama do not.
- **Code:** `github.com/ibnsina-llm/ibnsina` — pipeline, tokenizer, training patches, SFT recipe with all prompts, evaluation and release tooling. Apache-2.0.
- **Not released:** the assembled corpus; raw SFT candidates that the judge rejected.

## 9. Limitations

A 1.5 B model on 46 B tokens: fluent Persian, weak on precise facts, no memory across conversations, English as a second language, knowledge frozen at training time. No built-in internet access. Not for medical, legal or financial decisions. The evaluation is exam-style multiple choice plus validation perplexity; open-ended generation quality is judged only informally. The stability episode cost a rollback and a learning-rate change mid-run whose effect on the final model is documented but not ablated. The frontier comparison covers one benchmark and the specific model versions available on the run date.

## 10. Acknowledgments

nanochat (Andrej Karpathy) for the training loop, tokenizer tooling and chat scaffolding; the Muon optimizer; llama.cpp for inference and distribution; datatrove for deduplication and the FineWeb-Edu rubric my classifier adapts; FineWeb-2, CulturaX and mC4; OpenWebMath, StarCoderData, peS2o, OPUS; the Ganjoor project; ParsiNLU, PerCoR, TARAZ, FarsInstruct and PersianMedQA; the EMNLP 2025 taarof study (arXiv 2509.01035); and the Persian models that came before — PersianMind, Dorna, PersianLLaMA, Maral, gpt2-fa and ParsBERT. The pipeline, training runs, evaluations and the drafts of this report were executed by AI coding agents (Claude Code) under my direction; the design decisions, reviews and mistakes are mine.

---

## Appendix A — Hyperparameters

| | value |
|---|---|
| depth / width / heads / KV heads / head dim | 28 / 2048 / 16 / 4 / 128 |
| FFN hidden | 6144 (SwiGLU) |
| context | 2048 |
| vocab | 32,768 (tokenizer v2, Llama-3 regex) |
| batch | 524,288 tokens (device batch 8 × 8 GPUs × 2048 × 4 micro-steps) |
| iterations | 70,000 completed (planned 88,000; shortened after the second §5 event) |
| warm-up / warmdown | 500 steps warm-up; scheduled linear warmdown from 52,800; from the 60,000 resume a compressed linear warmdown to 70,000 (multiplier 0.79 at the splice → `final_lr_frac` = 0.05, nanochat's default — not overridden in the run args) |
| optimiser | Muon (matrices; momentum 0.85 → 0.97 linear over the first 400 steps, 0.97 constant, warmdown to 0.90 mirroring the LR warmdown (incl. the compressed 60,000→70,000 segment); 5 Newton–Schulz steps, β₂ 0.9) + AdamW (embeddings, unembedding, norm gains; nanochat LRs scaled by 1/√(d/768)) |
| learning-rate scale | ×1.0 to 20,000 · ×0.5 from the 20,000 resume · ×0.35 from the 60,000 resume |
| weight decay | nanochat scaled default, cosine to zero |
| precision | bf16 (fp8 trial abandoned) |
| guards | gradient-norm guard 6× running average; loss guard +1 nat, ≤12 consecutive skips |
| checkpoints | every 1000 steps, mirrored to object storage; eval every 2000 steps on 41.9 M validation tokens |
| SFT | `chat_sft_fa.py`: 85 steps (~1.5 min, 8×H100), mixture 100,997 rows (`sft/v2` 50,997 + 50,000 MMLU option-letter auxiliary), momentum carried from the base checkpoint with LRs reset, SFT-eval bpb 0.2825 |

## Appendix B — Compute and cost

| item | amount |
|---|---|
| Pretraining hardware | 1 × a3-highgpu-8g spot (8 × H100 80 GB), Google Cloud |
| Throughput | 432–438 k tok/s, 54–55 % MFU, 1.21–1.23 s/step |
| Pretraining stepping time | ≈23.7 h of productive 8×H100 time for the final line, plus roughly 8 h consumed by the instability episode across rollbacks (estimate) |
| Pretraining cost | `[[TBD Sept 1: from cloud billing; config estimate was $800–900 spot incl. restarts]]` |
| Pilot (360 M, 15 k steps) | 81 min on 8 × H100 spot, ≈$45 |
| Tokenizer, corpus pipeline | CPU VMs (n2-highmem-32/64) `[[TBD Sept 1: VM-hours]]`; Gemini labels for the quality classifier (10 k docs) |
| SFT data | ≈$228 teacher/judge + $5 translation |
| Frontier PersianMedQA sweep | `[[TBD Sept 1: OpenRouter total; Gemini rows on Vertex credits]]` |

## Appendix C — Sources of numbers

Citation labels below resolve to exact locations in a private mapping kept outside the published report (`docs/report/private/sources-map.json`, git-ignored).

| claim | source (citation label) |
|---|---|
| 39-source pipeline, 28 sources in this build; slice shares; licence exclusions | `README.md` (Data table); licence table `pipeline/licenses.json`; public mix manifest |
| per-slice tokens / docs / shares / source counts / epochs; literature shortfall | public mix manifest, `train_v1_1_open`, category level: `docs/data/mix_manifest_public_v1_1.json` (generated by `pipeline/manifest_public.py`; the full build manifest is **[MIX-MANIFEST-FULL]**, private) |
| dedup statistics (170.6 M → 122.0 M; mC4 −39 %, FineWeb-2 −32 %) | **[STOP-2]** Phase 1 dedup report (overlap matrix) |
| classifier accuracy 0.74 / within-1 0.91; web slice bands | **[STOP-3]** Phase 2 quality report; band table `scored/_bands.json` (pipeline artefact) |
| chap.sch.ir: 429 textbooks, 14.5 M tokens × 3 epochs | **[V11-BUILD]** v1.1 build note |
| tokenizer fertility table | **[FERTILITY-V1]** fertility report, tokenizer v1_32k (v2 identical ±0.005; `training/fertility.py`) |
| architecture, parameter count, GQA, RoPE | `training/config_1p5b.md`, `release/MODEL_CARD_ibnsina-1.5b.md` |
| toy A/B (1.102 vs 1.072, +4 % tok/s) | **[TOY-AB]** `training/toy_gguf_loop.sh` result record |
| throughput / MFU / step time; device batch 8 | **[TRAIN-LOG]** 1.5B training log, step lines; launch notes |
| fp8 divergence numbers | **[FP8-AB]** fp8 A/B logs and decision record |
| stability episode: loss 2.55→7.45; val 0.6118@26k; 0/255 rank-consistent; logits 53/51/102; grad-norm spikes; new-trajectory vals 0.5909/0.5893; logits 51–56 | **[TRAIN-LOG]** guard lines; **[ATTN-PROBE]** attention-probe output (`training/attn_probe.py`); engineering note **[STABILITY-NOTE]** |
| SFT counts, teacher split, decontamination sets, licences | **[SFT-MANIFEST]** SFT manifest, `sft/v2` build; `sft_v2/sft_taxonomy.yaml` |
| judge calibration | `sft_v2/CALIBRATION.md` |
| candidates judged ≈141 k, kept 51,400, DPO 13,453, spend $228 | **[SFT-SB]** S-B judge report |
| pilot results | **[PILOT]** pilot result record and bundle |
| GGUF validation (6/6, 24/24) | **[TOY-GGUF]** toy GGUF loop record |
| PersianMedQA paper numbers | arXiv 2506.00250 |
| frontier accuracies, truncation counts, per-field table (Tables B, B-2) | committed results: `docs/eval/frontier_persianmedqa_2026-08-30.md`, `.json`, `_by_field.md` (harness `eval/frontier_persianmedqa.py`) |
| frontier model ids and dates | **[MODEL-CATALOGUE]** OpenRouter model listing of 2026-08-30; Vertex AI model list |

### Placeholders to fill on Sept 1

- Table B IbnSina row (identical-harness run pending) and the chart refresh · abstract IbnSina sentence (same run) · §8 safetensors mirror · Appendix B pretraining cost, VM-hours, sweep total.

### Numbers not verifiable tonight (kept out or hedged)

- Exact per-source token counts inside each slice (the manifest's per-source block is not reproduced here; slice totals are used instead).
- Pretraining dollar cost (no billing export yet; the config's $800–900 estimate is quoted as an estimate).
- The README's "Persian literature 5 %" differs from the manifest's 0.5 % (planned vs achieved); the report uses the manifest.
