"""
Llama-3-shaped model for nanochat's training loop. Select with NANOCHAT_ARCH=llama (see nanochat/arch.py).
Exactly the llama.cpp / HF `llama` architecture so checkpoints export to standard GGUF:
- pre-norm RMSNorm with learnable gain (attn_norm / ffn_norm / output_norm)
- RoPE, textbook rotation (+theta) on half-split channels (HF layout; permuted to llama.cpp's interleaved layout at export)
- MHA/GQA (n_kv_head), scale 1/sqrt(head_dim), full causal context
- SwiGLU MLP (gate/up/down), hidden = round_up(8/3 * n_embd, 256)
- untied token embedding and lm_head; no QK-norm, no logit softcap, no value embeddings, no lambdas, no sliding windows
Same public interface as nanochat.gpt.GPT (init_weights, setup_optimizer, forward, estimate_flops, ...). Trained with nanochat's Muon+AdamW.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.gpt import Linear  # fp32 master weights, matmul in the activation dtype


def ffn_dim(n_embd, multiple_of=256):
    return ((int(8 * n_embd / 3) + multiple_of - 1) // multiple_of) * multiple_of


@dataclass
class LlamaConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "L"   # accepted for CLI/meta compatibility; a Llama block always attends to the full context
    arch: str = "llama"
    rope_base: float = 500000.0
    norm_eps: float = 1e-5
    ffn_hidden: int = 0         # SwiGLU width; 0 = round_up(8/3 * n_embd, 256)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.weight.to(x.dtype), eps=self.eps)


def apply_rotary_emb(x, cos, sin):
    """Textbook RoPE on half-split channels (HF Llama convention): (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], 3)


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0 and config.n_head % config.n_kv_head == 0
        self.c_q = Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.ffn_hidden or ffn_dim(config.n_embd)
        self.c_gate = Linear(config.n_embd, h, bias=False)
        self.c_up = Linear(config.n_embd, h, bias=False)
        self.c_down = Linear(h, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_down(F.silu(self.c_gate(x)) * self.c_up(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config, layer_idx)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn(self.attn_norm(x), cos_sin, window_size, kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Llama(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """NOTE: runs under a meta device context (shapes only); real init happens in init_weights()."""
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.norm_f = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.window_sizes = [(config.sequence_len, 0)] * config.n_layer  # full context everywhere (interface compatibility)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5  # uniform with the std of N(0, 1/sqrt(n_embd))
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)   # RMS ~1 residual stream (nanochat normalises embeddings; Llama has no post-embedding norm)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_up.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_down.weight)
            torch.nn.init.ones_(block.attn_norm.weight)
            torch.nn.init.ones_(block.mlp_norm.weight)
        torch.nn.init.ones_(self.norm_f.weight)
        head_dim = n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
        base = self.config.rope_base if base is None else base
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().to(COMPUTE_DTYPE), freqs.sin().to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_matmul_params(self):
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def estimate_flops(self):
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * self.num_matmul_params() + self.config.n_layer * 12 * h * q * t

    def estimate_decode_flops(self, context_len):
        h, q = self.config.n_head, self.config.n_embd // self.config.n_head
        return 2 * self.num_matmul_params() + self.config.n_layer * 4 * h * q * min(context_len, self.config.sequence_len)

    def estimate_prefill_flops(self, num_tokens):
        h, q = self.config.n_head, self.config.n_embd // self.config.n_head
        w = min(self.config.sequence_len, num_tokens)
        attended = w * (w + 1) // 2 + (num_tokens - w) * w
        return 2 * self.num_matmul_params() * num_tokens + self.config.n_layer * 4 * h * q * attended

    def kv_bytes_per_token(self):
        head_dim = self.config.n_embd // self.config.n_head
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * COMPUTE_DTYPE.itemsize

    def kv_read_bytes(self, context_len):
        head_dim = self.config.n_embd // self.config.n_head
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * COMPUTE_DTYPE.itemsize * min(context_len, self.config.sequence_len)

    def _norm_params(self):
        return [self.norm_f.weight] + [p for b in self.transformer.h for p in (b.attn_norm.weight, b.mlp_norm.weight)]

    def num_scaling_params(self):
        wte = self.transformer.wte.weight.numel()
        lm_head = self.lm_head.weight.numel()
        scalars = sum(p.numel() for p in self._norm_params())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters()) - sum(p.numel() for b in self.transformer.h for p in (b.attn_norm.weight, b.mlp_norm.weight))
        total = wte + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {"wte": wte, "value_embeds": 0, "lm_head": lm_head, "transformer_matrices": transformer_matrices, "scalars": scalars, "total": total}

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        norm_params = self._norm_params()
        norm_ids = {id(p) for p in norm_params}
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in norm_ids]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(norm_params)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=norm_params, lr=scalar_lr * 0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),  # RMSNorm gains
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device and self.cos.dtype == COMPUTE_DTYPE
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)
        x = self.norm_f(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return logits
