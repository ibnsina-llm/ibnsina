---
license: apache-2.0
language:
- fa
- en
pipeline_tag: text-generation
library_name: llama.cpp
tags:
- persian
- farsi
- ibnsina
- gguf
- llama
- chat
---

# ibnsina-1.5b

> ⚠️ **ابن‌سینا یک مدل کوچک است برای نوشتن، خلاصه، ترجمه و گفت‌وگو به فارسی — نه منبع اطلاعات درباره‌ی افراد، سیاست یا اخبار.** برای مشاوره، پاسخ به سؤال‌های دانشی، حل ریاضی یا نوشتن کد ساخته نشده است؛ برای آن کارها از مدل‌های بزرگ استفاده کنید. کارش تولید متن فارسی، آفلاین و روی دستگاه خودتان است — و ممکن است جمله‌های روان اما نادرست بسازد؛ هر چیز مهم را خودتان راستی‌آزمایی کنید.
>
> ⚠️ **IbnSina is a small model for writing, summarizing, translating and conversing in Persian — not a source of facts about people, politics, or news.** It is not built for advice, knowledge questions, math, or code — use a large model for those. What it is for: offline Persian text generation on your own device. It can produce fluent but wrong sentences — verify anything that matters.

**اولین مدل زبانی فارسی متن‌باز در این مقیاس که از صفر با فارسی آموزش دیده است.** بیشتر مدل‌های فارسی روی یک پایهٔ انگلیسی‌زبان ساخته شده‌اند؛ این مدل هرگز اول انگلیسی یاد نگرفت. ابن‌سینا ۱٫۵B با ۴۶ میلیارد توکن — عمدتاً فارسی — آموزش دیده و برای گفت‌وگو تنظیم شده است؛ با llama.cpp، ollama و LM Studio روی لپ‌تاپ و گوشی اجرا می‌شود. وزن‌ها، کد و دستور ساخت آزادند (Apache-2.0). [راهنمای فارسی](https://github.com/ibnsina-llm/ibnsina/blob/main/README_FA.md)

**The first open-source Persian LLM at modern scale pretrained from scratch** — most Persian models adapt an English-first base; this one never knew English first. IbnSina-1.5B (1.48 B parameters, Llama-compatible architecture) was trained on 46 B tokens, then instruction-tuned on a 51.5 k-conversation Persian recipe. Code, data recipe and weights: [github.com/ibnsina-llm](https://github.com/ibnsina-llm) · [Persian README](https://github.com/ibnsina-llm/ibnsina/blob/main/README_FA.md). Author: Sina Meraji · ORCID 0009-0002-8028-1932 · github.com/sinameraji.

![PersianMedQA: IbnSina alongside the 2026 frontier and today's small models — identical protocol](persianmedqa_chart_en.svg)

## Files

| file | use |
|---|---|
| `ibnsina-1.5b-Q4_K_M.gguf` | laptop / phone (≈0.9 GB) |
| `ibnsina-1.5b-Q8_0.gguf` | near-lossless (≈1.6 GB) |
| `ibnsina-1.5b-f16.gguf` | full precision (≈3 GB) |
| `Modelfile` | ollama |

## Quickstart (Mac / Windows / Linux)

Install [ollama](https://ollama.com/download) for your OS, then:
```bash
ollama run hf.co/ibnsina-llm/ibnsina-1.5b
```
Or in [LM Studio](https://lmstudio.ai): search **ibnsina-llm/ibnsina-1.5b**. For llama.cpp, download a GGUF from this repo: `llama-cli -m ibnsina-1.5b-Q4_K_M.gguf`.

## Run

```bash
llama-cli -m ibnsina-1.5b-Q4_K_M.gguf                      # llama.cpp
ollama create ibnsina-1.5b -f Modelfile && ollama run ibnsina-1.5b
```
The GGUF carries the chat template. Format: `<|user_start|>…<|user_end|><|assistant_start|>…<|assistant_end|>`; the runtime adds `<|bos|>`. A system message is folded into the first user turn.

## Model

| | |
|---|---|
| architecture | Llama-compatible: 28 layers, d=2048, 16 heads / 4 KV heads (GQA), SwiGLU 6144, RMSNorm, RoPE θ=500k, untied head |
| parameters | 1.48 B |
| context | 2048 tokens |
| tokenizer | 32,768-token byte-level BPE, Persian-dominant training sample, Llama-3 pre-tokenizer regex (`llama-bpe`) |
| pretraining | {{TRAIN_TOKENS}} tokens of `train_v1_1_open`, {{TRAIN_STEPS}} steps × 524k tokens, bf16, Muon + AdamW (nanochat loop), 8×H100 |
| fine-tuning | `sft_v2`: 51.5 k judged conversations across 17 categories (+ canonical identity set) |

Tokenizer efficiency on held-out Persian: 1.29 tokens/word vs 1.73 (Qwen3.5) and 1.64 (Gemma 3).

## Data

Persian web (CulturaX, mC4, FineWeb-2; classifier-filtered) 63 %, English educational 15 %, code 10 %, math & Iranian school textbooks 5 %, Persian literature 0.5 % (planned at 5 %; the shortfall of licence-clean text was reallocated to Persian web), Wikipedia 3 %, fa–en parallel 2 %. Only sources whose licences permit an Apache-2.0 release are included; the per-source licence table and the category-level mix manifest are in the GitHub repo.

## Evaluation

{{EVAL_TABLE}}

Small-model caveat: multiple-choice accuracies near 25–40 % are close to chance; treat these as a baseline.

A technical report is forthcoming: IbnSina evaluated alongside the 2026 frontier (Claude Opus 5, GPT-5.6, Gemini, Kimi K3, GLM, DeepSeek, Qwen) on Persian exam benchmarks such as [PersianMedQA](https://arxiv.org/abs/2506.00250) — the first such comparison for this model generation. *[link placeholder]*

## Intended use and limitations

Persian conversation, writing, summarisation, translation and everyday questions. Not for medical, legal or financial decisions; it states general information and points to professionals. It has no memory between conversations and knowledge frozen at training time, and makes factual mistakes more often than large models — verify anything that matters. No built-in internet access: it can emit calculator, date-conversion and search *tool calls* in nanochat's format; these work only in a host that executes them (the reference runtime in the GitHub repo does; llama.cpp and ollama do not).

## Behaviour policy · سیاست رفتاری

Trained to answer in natural register-matching Persian, to say «نمی‌دانم» rather than invent, and to be **symmetrically respectful**: it declines to mock or insult any person or group — political or religious figures, ethnicities, genders, nationalities, on every side — while answering factual and theological questions normally, recounting documented history honestly, and presenting contested political questions as "supporters say / critics say" without a verdict.

این مدل آموزش دیده است که به فارسیِ طبیعی و هم‌سطح با لحن کاربر پاسخ دهد، به‌جای ساختن پاسخ بگوید «نمی‌دانم»، و **احترام را متقارن** رعایت کند: درخواست تمسخر یا توهین به هیچ شخص یا گروهی را نمی‌پذیرد — چهره‌های سیاسی و مذهبی، اقوام، جنسیت‌ها و ملیت‌ها، از هر طرف که باشند — اما به پرسش‌های واقعی و الهیاتی عادی پاسخ می‌دهد، تاریخ مستند را صادقانه روایت می‌کند، و پرسش‌های سیاسیِ مورد مناقشه را به شکل «موافقان می‌گویند / منتقدان می‌گویند» ارائه می‌کند، بی‌آنکه حکم بدهد. در موضوعات پزشکی و حقوقی اطلاعات عمومی می‌دهد و برای تصمیم‌ها به متخصص ارجاع می‌دهد؛ در شرایط بحرانی کوتاه و گرم پاسخ می‌دهد و شماره‌های امداد ایران را می‌گوید.

## Licence

Apache-2.0 for weights and code. Training-data licences are per source (see repo). Included: public-domain, permissive and share-alike sources. Curated material without an explicit open licence (official school textbooks, curator-provided volumes) and one research-use parallel corpus (OPUS OpenSubtitles) were admitted per source by curator decision on 2026-08-28 and are recorded as such; NC/ND-licensed sources were excluded. The parallel-data slice includes OPUS OpenSubtitles (research-use terms), retained for training only by curator decision; roughly 0.3 B of 46 B tokens. No source text is redistributed.

## Acknowledgments

Built with [nanochat](https://github.com/karpathy/nanochat) (Andrej Karpathy) and the [Muon](https://github.com/KellerJordan/Muon) optimizer; distributed via [llama.cpp](https://github.com/ggml-org/llama.cpp). Data methods from [datatrove](https://github.com/huggingface/datatrove) and the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) rubric; Persian poetry from [Ganjoor](https://ganjoor.net). Persian NLP we build on or evaluate against: [ParsiNLU](https://github.com/persiannlp/parsinlu), [FarsInstruct](https://huggingface.co/datasets/ParsiAI/FarsInstruct), [PerCoR](https://huggingface.co/datasets/MCINext/PerCoR), [TARAZ](https://github.com/Georgetown-IR-Lab/TARAZ), the EMNLP 2025 [taarof study](https://arxiv.org/abs/2509.01035); and the Persian models that came before — [PersianMind](https://huggingface.co/universitytehran/PersianMind-v1.0), [Dorna](https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct), [gpt2-fa](https://huggingface.co/HooshvareLab/gpt2-fa). The pipeline, training runs and evaluations were executed by AI coding agents (Claude Code) under Sina Meraji's direction.

## Citation

```
@software{ibnsina2026, title={IbnSina: an open Persian-first language model family}, author={Meraji, Sina}, year={2026}, url={https://github.com/ibnsina-llm}, note={ORCID 0009-0002-8028-1932}}
```
