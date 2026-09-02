"""Architecture switch: NANOCHAT_ARCH=gpt (nanochat's own, default), llama (nanochat/llama.py) or qwen3 (nanochat/qwen3.py,
+per-head QK-norm) — llama and qwen3 are GGUF-exportable. Import sites use `from nanochat.arch import GPT, GPTConfig, Linear`;
checkpoint loading auto-detects from meta["model_config"]["arch"]."""
import os
from nanochat.gpt import Linear
_ARCH = os.environ.get("NANOCHAT_ARCH", "gpt")
if _ARCH == "llama":
    from nanochat.llama import Llama as GPT, LlamaConfig as GPTConfig
elif _ARCH == "qwen3":
    from nanochat.qwen3 import Qwen3 as GPT, Qwen3Config as GPTConfig
else:
    from nanochat.gpt import GPT, GPTConfig


def model_classes_for(model_config_kwargs):
    """Pick (GPT, GPTConfig) for a checkpoint's saved model_config (Llama/Qwen3 checkpoints carry arch="llama"/"qwen3")."""
    arch = model_config_kwargs.get("arch") if isinstance(model_config_kwargs, dict) else None
    if arch == "llama":
        from nanochat.llama import Llama, LlamaConfig
        return Llama, LlamaConfig
    if arch == "qwen3":
        from nanochat.qwen3 import Qwen3, Qwen3Config
        return Qwen3, Qwen3Config
    from nanochat.gpt import GPT as _GPT, GPTConfig as _GPTConfig
    return _GPT, _GPTConfig
