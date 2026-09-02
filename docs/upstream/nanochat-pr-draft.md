# APPROVED by Sina 2026-08-31 — first-person body below is final. Submission = Sina's click on the prefilled PR URL.

**Target:** `karpathy/nanochat` `master`, base `92d63d4`. Branch: `sinameraji/nanochat:fix/async-collective-tensor-lifetime` (from worktree `~/nanochat-pr`, one commit, `nanochat/optim.py` +11/−10).

## Title

optim: keep async collective inputs and Work handles alive until wait()

## Body (final, Sina's wording, markdown-styled)

I hit this while pretraining a 1.5B Persian model on nanochat (8×H100, bf16), so the Muon path has had a hard workout. Sharing a latent lifetime bug I found in it.

`MuonAdamW.step()` overlaps communication with compute by launching `reduce_scatter_tensor` / `all_gather_into_tensor` with `async_op=True`, keeping only the returned Future. Two lifetime issues:

1. In `_compute_muon`, the input of the final all_gather, `updated_params`, is a local tensor that goes out of scope as soon as the method returns — before `_finish_gathers` waits on the collective. Nothing else references it, so the CUDA caching allocator can hand its block to the next allocation (usually the next group's `updated_params`, immediately overwritten) while NCCL is still reading it.
2. The `Work` objects are dropped everywhere (`.get_future()` only). On current PyTorch (checked on 2.9.1 + NCCL 2.27.5) the NCCL backend doesn't `record_stream` collective arguments by default, so correctness relies on the caller keeping tensors referenced until `wait()`.

If the timing lines up, the gather can deliver torn data: every rank keeps its own shard of each Muon matrix correct and stale bytes for the other `world_size - 1` shards.

**Fix:** keep each async collective's input tensor and its `Work` referenced in the info dicts that already flow to the waits, and have `_finish_gathers` call `work.wait()` alongside `future.wait()` before the copy-back. No behaviour change on the happy path, no extra memory beyond what was already in flight, no extra synchronisation beyond the existing waits.

**Honest scope:** I first suspected this path when my run hit repeated loss plateaus around step 27,000, but forensics exonerated it for my actual incident — with updates frozen at a plateau, all-rank parameter checks showed 0/255 tensors differing, and the plateaus reproduced under fully synchronous collectives too. My real root cause was attention-logit growth in a custom no-QK-norm model variant, unrelated to this optimizer. Submitting anyway: the lifetime hazard is real under current PyTorch semantics and cheap to close. I'm presenting it as a hardening fix with a consistency test, not a field-failure fix — `repro_torn_gather.py` (in a comment below, not the diff) churns the allocator between steps and asserts cross-rank parameter equality, but I did not observe divergence I could attribute to this bug in production.

**Environment:** torch 2.9.1+cu128, NCCL 2.27.5, CUDA 12.8, 8×H100 (GCE a3-highgpu-8g), nanochat `92d63d4`.

## After creation

Post `scratchpad/nanochat_pr/repro_torn_gather.py` as the first comment (or a gist link).
