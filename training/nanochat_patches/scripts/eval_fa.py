"""Persian categorical evals (ParsiNLU MC / entailment / QQP) on a nanochat checkpoint. Single GPU is fine.
usage: uv run --no-sync python -m scripts.eval_fa -i sft -g pilot -o /data/eval/results.json [-x 500]"""
import argparse, json, time
from functools import partial
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model
from scripts.chat_eval import run_categorical_eval
from tasks.parsinlu import ParsiNLUMC, ParsiNLUEntailment, ParsiNLUQQP
from tasks.persianmedqa import PersianMedQA

TASKS = {"ParsiNLU-MC": (ParsiNLUMC, 0.25),
         "ParsiNLU-MC/math_and_logic": (partial(ParsiNLUMC, category="math_and_logic"), 0.25),
         "ParsiNLU-MC/common_knowledge": (partial(ParsiNLUMC, category="common_knowledge"), 0.25),
         "ParsiNLU-MC/literature": (partial(ParsiNLUMC, category="literature"), 0.25),
         "ParsiNLU-Entailment": (ParsiNLUEntailment, 1 / 3), "ParsiNLU-QQP": (ParsiNLUQQP, 0.5),
         "PersianMedQA": (PersianMedQA, 0.25)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--source", default="sft"); ap.add_argument("-g", "--model-tag", default=None); ap.add_argument("-s", "--step", type=int, default=None)
    ap.add_argument("-b", "--batch-size", type=int, default=16); ap.add_argument("-x", "--max-problems", type=int, default=None); ap.add_argument("-o", "--out", default="/data/eval/results.json")
    a = ap.parse_args()
    ddp, rank, local_rank, world, device = compute_init(autodetect_device_type())
    model, tokenizer, meta = load_model(a.source, device, phase="eval", model_tag=a.model_tag, step=a.step)
    res = {"source": a.source, "model_tag": a.model_tag, "step": meta.get("step") if isinstance(meta, dict) else a.step, "tasks": {}}
    for name, (ctor, base) in TASKS.items():
        t0 = time.time()
        try:
            task = ctor()
        except Exception as e:                      # data not fetched (e.g. gated set without a token) -> skip, keep the run going
            print0(f"{name:32s} SKIPPED ({type(e).__name__}: {str(e)[:80]})"); continue
        n = len(task) if a.max_problems is None else min(len(task), a.max_problems)
        acc = run_categorical_eval(task, tokenizer, model, a.batch_size, max_problems=a.max_problems)
        res["tasks"][name] = {"acc": acc, "n": n, "random": base, "centered": (acc - base) / (1 - base)}
        print0(f"{name:32s} acc {100*acc:5.1f}%  (random {100*base:.0f}%, centered {100*(acc-base)/(1-base):+.1f})  n={n}  {time.time()-t0:.0f}s")
    if rank == 0:
        json.dump(res, open(a.out, "w"), indent=1); print0(f"wrote {a.out}")
    compute_cleanup()


if __name__ == "__main__":
    main()
