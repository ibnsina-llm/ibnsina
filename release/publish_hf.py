#!/opt/pipe/bin/python3
"""Release day: upload a release bundle (GGUFs, Modelfile, README/model card, tokenizer) to huggingface.co/ibnsina-llm/<name>.
Runs on the pipeline VM (HF token + huggingface_hub live there). DOES NOTHING without --yes (Sina's sign-off).
  publish_hf.py --name ibnsina-1.5b --bundle gs://.../release/ibnsina-1.5b --yes"""
import argparse, os, subprocess, sys, tempfile
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--bundle", required=True); ap.add_argument("--org", default="ibnsina-llm"); ap.add_argument("--yes", action="store_true"); a = ap.parse_args()
ap.add_argument("--private", action="store_true", help="create the HF repo private (beta)")
repo = f"{a.org}/{a.name}"
if not a.yes:
    print(f"dry run: would create/upload model repo {repo} from {a.bundle} (README.md = model card, *.gguf, Modelfile, tokenizer files). Re-run with --yes after sign-off."); sys.exit(0)
from huggingface_hub import HfApi, create_repo
tmp = tempfile.mkdtemp(); subprocess.run(["gcloud", "--no-user-output-enabled", "storage", "rsync", "-r", a.bundle, tmp], check=True)
for f in os.listdir(tmp):
    if f.endswith((".pt", ".log", ".json")) and f not in ("results_pilot.json",): pass
api = HfApi(); create_repo(repo, repo_type="model", exist_ok=True, private=a.private)
api.upload_folder(repo_id=repo, folder_path=tmp, repo_type="model", allow_patterns=["*.gguf", "Modelfile", "README.md", "tokenizer.pkl", "token_bytes.pt", "results_*.json"], commit_message=f"{a.name}: initial release")
print(f"uploaded -> https://huggingface.co/{repo}")
