# Persian LLM pilot — run it on your Mac

This bundle is a nanochat-native checkpoint (nanochat commit `92d63d4`). llama.cpp has no nanochat architecture yet
(value embeddings, residual/x0 lambdas, sliding-window pattern), so there is no GGUF/ollama build of this pilot — it runs
through nanochat itself on Apple silicon (MPS). A 360M model answers at a comfortable pace on an M-series laptop.

## 1. Get the bundle (≈3 GB)
```bash
mkdir -p ~/persian-pilot && gcloud storage rsync -r ${CORPUS_BUCKET:-gs://YOUR-BUCKET}/pilot/pilot ~/persian-pilot
```
Layout (this is a complete `NANOCHAT_BASE_DIR`):
```
~/persian-pilot/tokenizer/{tokenizer.pkl, token_bytes.pt}
~/persian-pilot/chatsft_checkpoints/pilot/{model_XXXXXX.pt, meta_XXXXXX.json}   # chat model (after SFT)
~/persian-pilot/base_checkpoints/pilot/{model_015000.pt, meta_015000.json}       # raw pretrained model
~/persian-pilot/{results_pilot.json, loss.csv, README.md}
```

## 2. nanochat on the Mac (once)
```bash
git clone https://github.com/karpathy/nanochat ~/nanochat && cd ~/nanochat && git checkout 92d63d4
uv sync --extra cpu          # macOS wheels include MPS
```

## 3. Talk to it
```bash
cd ~/nanochat
NANOCHAT_BASE_DIR=~/persian-pilot uv run python -m scripts.chat_cli --device-type mps -i sft -g pilot
# one-shot:
NANOCHAT_BASE_DIR=~/persian-pilot uv run python -m scripts.chat_cli --device-type mps -i sft -g pilot -p "پایتخت ایران کجاست؟" -t 0.6
```
`-i base -g pilot` loads the raw pretrained model instead (completion-style, no chat format).

## Eval snapshot
`results_pilot.json` — ParsiNLU multiple-choice (1,050 exam questions: math & logic / common knowledge / literature),
entailment and question-paraphrase accuracy of the SFT model, with random baselines and centered scores.
