# PersianMedQA — frontier comparison (rows as of 2026-08-30; Gemini 3.8 Flash added 2026-09-09)

PersianMedQA test split (Ranjbar Kalahroodi et al. 2025), 5,235 questions with a complete option set

Protocol: paper zero-shot prompt, temperature 0, answer = option number; unparsed = no option number in the reply (truncated thinking or refusal), counted as wrong in acc_all; thinking models retried once with a 4,096-token cap

Frontier rows were NOT re-run for the 3B launch (owner decision 2026-09-09); Gemini 3.8 Flash is an extra row run on Vertex (GCP credits). Newer flagships not shown: Claude Fable 5.1 (Sept 1), GPT-6 Astra Pro (Sept 4), Qwen3.8 Max 0902 (Sept 3).

| Model (vendor, release) | Accuracy, all 5,235 | Accuracy, answered | Truncated / unparsed | Protocol notes |
|---|---|---|---|---|
| Gemini 3.8 Flash (Google, 2026-09) | **88.92 %** | 88.94 % | 1 (0.0 %) | Vertex AI, thinking on; added 2026-09-09 (GCP credits, no OpenRouter spend) |
| Gemini 3.1 Pro (preview) (Google, 2026-06) | **88.77 %** | 88.82 % | 3 (0.06 %) | Vertex AI, thinking on |
| Gemini 3.7 Flash (Google, 2026-08) | **88.65 %** | 88.65 % | 0 (0.0 %) | Vertex AI, thinking on |
| Grok 4.6 (xAI, 2026-08) | **88.18 %** | 88.18 % | 0 (0.0 %) | thinking on (mandatory) |
| Claude Fable 5 (Anthropic, 2026-06) | **88.08 %** | 89.15 % | 63 (1.2 %) | thinking on (mandatory) |
| Claude Opus 5 (Anthropic, 2026-07) | **86.99 %** | 86.99 % | 0 (0.0 %) |  |
| Gemini 2.5 Pro (prior generation) (Google, 2025-06) | **86.40 %** | 86.45 % | 3 (0.06 %) | Vertex AI, thinking on |
| GPT-5.6 Terra (OpenAI, 2026-07) | **84.53 %** | 84.53 % | 0 (0.0 %) |  |
| Qwen3.8 2.4T-A95B (Alibaba, 2026-08) | **84.13 %** | 89.40 % | 309 (5.9 %) | thinking on (mandatory) |
| GLM-5.3 (Z.ai, 2026-08) | **83.97 %** | 87.57 % | 215 (4.11 %) | thinking on (mandatory) |
| Kimi K3 (Moonshot AI, 2026-07) | **83.19 %** | 83.19 % | 0 (0.0 %) |  |
| Qwen3.8 Max (Alibaba, 2026-08) | **82.14 %** | 90.34 % | 475 (9.07 %) | thinking on (mandatory) |
| Hunyuan Hy4 (preview) (Tencent, 2026-08) | **76.18 %** | 76.44 % | 18 (0.34 %) |  |
| DeepSeek V4 Pro (0813) (DeepSeek, 2026-08) | **72.51 %** | 72.53 % | 1 (0.02 %) |  |
