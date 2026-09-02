"""Spike guard for scripts/base_train.py (idempotent). NANOCHAT_SPIKE_GUARD=<margin nats> (default 1.0; 0 disables):
when a step's mean loss (all-reduced, so every rank agrees) exceeds the debiased running-average loss by more than <margin>,
the optimizer update is skipped (grads zeroed, EMA left untouched) and the batch is simply consumed. Motivation: three
garbage batches around steps 27008-27397 of the 1.5B run (single-step loss 8-15 nats vs 2.6 average) blew the run up once."""
import sys
p = "scripts/base_train.py"; s = open(p).read()
if "SPIKE_GUARD" in s:
    print("base_train: spike guard already present"); sys.exit(0)

old1 = "    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)\n        train_loss = loss.detach() # for logging\n"
new1 = ("    step_loss_acc = None; _gx = []; _gl = []  # SPIKE_GUARD: mean loss over this step's micro-batches (+ the batches, for forensics)\n"
        "    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)\n        train_loss = loss.detach() # for logging\n"
        "        step_loss_acc = train_loss / grad_accum_steps if step_loss_acc is None else step_loss_acc + train_loss / grad_accum_steps\n"
        "        _gx.append(x); _gl.append(train_loss)\n")
assert s.count(old1) == 1, "micro-step loop not found"; s = s.replace(old1, new1)

old2 = "    if scaler is not None:\n        scaler.unscale_(optimizer)\n"
new2 = ("    # SPIKE_GUARD: skip the update on a garbage batch. Reference = the guard's own EMA of the same all-reduced step-mean loss\n"
        "    # (self-consistent), warmed for 5 steps after (re)start; at most 3 consecutive skips so a broken guard can never starve the run.\n"
        "    _skip_update = False\n"
        "    if SPIKE_GUARD_MARGIN > 0 and step >= args.warmup_steps:\n"
        "        _sl = step_loss_acc.clone(); _ll = train_loss.clone()\n"
        "        if is_ddp_initialized(): dist.all_reduce(_sl, op=dist.ReduceOp.AVG); dist.all_reduce(_ll, op=dist.ReduceOp.AVG)\n"
        "        _slf = _sl.item(); _llf = _ll.item()\n"
        "        _gsq = torch.zeros((), device=step_loss_acc.device, dtype=torch.float32)\n"
        "        for _p in orig_model.parameters():\n"
        "            if _p.grad is not None: _gsq += _p.grad.float().pow(2).sum()\n"
        "        if is_ddp_initialized(): dist.all_reduce(_gsq, op=dist.ReduceOp.AVG)\n"
        "        _gn = _gsq.sqrt().item()   # RMS over ranks of the per-rank (pre-reduction) gradient norm\n"
        "        _ref = guard_ema / (1 - 0.9 ** guard_n) if guard_n > 0 else _slf   # guard_n counts EMA updates only, so a skip streak cannot decay the reference\n"
        "        _gref = grad_ema / (1 - 0.9 ** guard_n) if guard_n > 0 else _gn\n"
        "        _loss_trip = _slf > _ref + SPIKE_GUARD_MARGIN; _grad_trip = GRAD_GUARD_FACTOR > 0 and _gn > GRAD_GUARD_FACTOR * _gref\n"
        "        if guard_n < 5 or step % 200 == 0: print0(f\"SPIKE_GUARD diag step {step}: rank0_last_micro {train_loss.item():.3f} | rank0_step_mean {step_loss_acc.item():.3f} | global_step_mean {_slf:.3f} | global_last_micro {_llf:.3f} | ref {_ref:.3f} | grad_norm {_gn:.3f} (ref {_gref:.3f})\")\n"
        "        if guard_n >= 5 and (_loss_trip or _grad_trip) and guard_consec < GUARD_VALVE:\n"
        "            _skip_update = True; spike_skips += 1; guard_consec += 1\n"
        "            _pr = [torch.zeros_like(step_loss_acc) for _ in range(dist.get_world_size())] if is_ddp_initialized() else [step_loss_acc]\n"
        "            if is_ddp_initialized(): dist.all_gather(_pr, step_loss_acc)\n"
        "            print0(f\"SPIKE_GUARD: step {step} {'GRAD' if _grad_trip else 'LOSS'} trip: loss {_slf:.3f} (ref {_ref:.3f}) grad_norm {_gn:.3f} (ref {_gref:.3f}) -> update skipped ({spike_skips} skipped so far) | per-rank {[round(t.item(), 2) for t in _pr]} | rank0 micro {[round(t.item(), 2) for t in _gl]} | pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}\")\n"
        "            if guard_consec == 1:\n"
        "                _diffs = []\n"
        "                for _n, _p in orig_model.named_parameters():\n"
        "                    _mx = _p.detach().float().clone(); _mn = _p.detach().float().clone()\n"
        "                    if is_ddp_initialized(): dist.all_reduce(_mx, op=dist.ReduceOp.MAX); dist.all_reduce(_mn, op=dist.ReduceOp.MIN)\n"
        "                    _d = (_mx - _mn).abs(); _diffs.append((_d.max().item(), (_d > 0).float().mean().item(), _n))\n"
        "                _diffs.sort(reverse=True); _nd = sum(1 for d in _diffs if d[0] > 0)\n"
        "                print0(f\"SPIKE_GUARD consistency at step {step}: {_nd}/{len(_diffs)} params differ across ranks; worst: {[(n, round(m, 4), round(f, 3)) for m, f, n in _diffs[:6]]}\")\n"
        "            if guard_consec <= 4:\n"
        "                _r = dist.get_rank() if is_ddp_initialized() else 0; os.makedirs('/data/logs/spike_batches', exist_ok=True)\n"
        "                torch.save({'step': step, 'rank': _r, 'micro_losses': [t.item() for t in _gl], 'x': torch.stack(_gx).cpu(), 'loader': dict(dataloader_state_dict)}, f'/data/logs/spike_batches/step{step}_rank{_r}.pt')\n"
        "        else:\n"
        "            if guard_consec >= GUARD_VALVE and (_loss_trip or _grad_trip): print0(f\"SPIKE_GUARD: step {step} loss {_slf:.3f} grad_norm {_gn:.3f} still anomalous after {GUARD_VALVE} skips -> accepting the update (valve)\")\n"
        "            guard_consec = 0; guard_ema = 0.9 * guard_ema + 0.1 * _slf; grad_ema = 0.9 * grad_ema + 0.1 * _gn; guard_n += 1\n"
        "    if _skip_update:\n"
        "        pass\n"
        "    elif scaler is not None:\n        scaler.unscale_(optimizer)\n")
assert s.count(old2) == 1, "optimizer block not found"; s = s.replace(old2, new2)

old3 = "    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point\n"
new3 = old3 + "    if _skip_update: train_loss_f = smooth_train_loss / (1 - 0.9 ** step)  # SPIKE_GUARD: keep the EMA (and the guard's reference) clean\n"
assert s.count(old3) == 1, "train_loss_f line not found"; s = s.replace(old3, new3)

old4 = "# Training loop\n"
new4 = ("# Training loop\n"
        "SPIKE_GUARD_MARGIN = float(os.environ.get('NANOCHAT_SPIKE_GUARD', '1.0'))  # nats above the running average that marks a garbage batch; 0 disables\n"
        "GRAD_GUARD_FACTOR = float(os.environ.get('NANOCHAT_GRAD_GUARD', '3.0'))  # skip the update when the all-rank grad norm exceeds this multiple of its running average; 0 disables\n"
        "GUARD_VALVE = int(os.environ.get('NANOCHAT_GUARD_VALVE', '12'))  # consecutive skips before an update is accepted anyway\n"
        "spike_skips = 0; guard_ema = 0.0; grad_ema = 0.0; guard_n = 0; guard_consec = 0\n"
        "print0(f'SPIKE_GUARD margin: {SPIKE_GUARD_MARGIN} | GRAD_GUARD factor: {GRAD_GUARD_FACTOR}')\n")
assert s.count(old4) == 1, "training loop header not found"; s = s.replace(old4, new4)
if "\nimport os\n" not in s and not s.startswith("import os\n"): s = "import os\n" + s
open(p, "w").write(s); print("base_train: spike guard added")
