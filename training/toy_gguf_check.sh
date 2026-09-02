#!/bin/bash
# Re-runnable post-build checks of the toy GGUF loop: quantize -> tokenizer parity -> greedy parity -> upload. Run as root on the toy box.
set -u; export HOME=/root PATH=/root/.local/bin:$PATH NANOCHAT_BASE_DIR=/data/nc2; B=${CORPUS_BUCKET:-gs://YOUR-BUCKET}; NC=/data/nanochat; LOG=/data/logs/toy_gguf_check.log; cd $NC; exec > >(tee -a $LOG) 2>&1
echo "== $(date -u +%FT%TZ) toy check start"
LC=/data/llama.cpp/build/bin; ls $LC | tr "\n" " "; echo
$LC/llama-quantize /data/export/toy-llama-f16.gguf /data/export/toy-llama-Q4_K_M.gguf Q4_K_M 2>&1 | tail -n 2
ls -la /data/export/*.gguf
# 5) tokenizer parity: llama.cpp ids vs tiktoken ids on Persian / English / code / digits / newlines
uv run --no-sync python - <<'PY'
import pickle, subprocess, json, sys
enc = pickle.load(open("/data/nc2/tokenizer/tokenizer.pkl", "rb"))
tests = ["زبان فارسی یکی از کهن‌ترین زبان‌های زنده جهان است.", "Persian is one of the oldest living languages in the world.", "def area(r):\n    return 3.14159 * r ** 2\n", "در سال 1403 حدود 12345678 نفر", "خط اول\n\n\nخط دوم  ", "Hello world! 😀 سلام"]
ok = True
for s in tests:
    ref = enc.encode_ordinary(s)
    out = subprocess.run(["/data/llama.cpp/build/bin/llama-tokenize", "-m", "/data/export/toy-llama-f16.gguf", "-p", s, "--ids", "--no-bos", "--log-disable"], capture_output=True, text=True).stdout
    got = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    same = got == ref; ok &= same
    print(("OK  " if same else "DIFF") + f" {s[:40]!r}: tiktoken {ref[:12]} llama.cpp {got[:12] if got else out[-200:]}")
print("TOKENIZER PARITY:", "PASS" if ok else "FAIL")
PY
# 6) greedy-decode parity: torch (bf16) vs llama.cpp F16 and Q4_K_M
uv run --no-sync python - <<'PY'
import subprocess, torch, sys
sys.argv = ["x"]
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
ddp, rank, lrank, world, device = compute_init(autodetect_device_type())
model, tok, meta = load_model("base", device, phase="eval", model_tag="toy-llama")
prompt = "زبان فارسی"
ids = [tok.get_bos_token_id()] + tok.encode(prompt)
with torch.no_grad():
    for _ in range(16):
        logits = model(torch.tensor([ids], device=device))
        ids.append(int(logits[0, -1].argmax()))
torch_text = tok.decode(ids[1 + len(tok.encode(prompt)):])
print("torch greedy   :", repr(torch_text))
for q in ("f16", "Q4_K_M"):
    r = subprocess.run(["/data/llama.cpp/build/bin/llama-simple", "-m", f"/data/export/toy-llama-{q}.gguf", "-n", "16", prompt], capture_output=True, text=True)
    out = (r.stdout or "").strip().splitlines(); gen = out[-1] if out else r.stderr[-300:]
    print(f"llama.cpp {q:6s}:", repr(gen[len(prompt):] if gen.startswith(prompt) else gen))
PY
gcloud --no-user-output-enabled storage cp /data/export/toy-llama-f16.gguf /data/export/toy-llama-Q4_K_M.gguf $LOG $B/toy_gguf/ && echo "uploaded to $B/toy_gguf/"
echo "== $(date -u +%FT%TZ) toy check done"
