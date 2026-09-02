"""Keep async-collective inputs and Work objects alive until wait() in nanochat/optim.py (MuonAdamW). Idempotent.
Bug: _compute_muon launched all_gather_into_tensor(stacked_params, updated_params, async_op=True) and dropped the local
`updated_params` (the gather INPUT) on return; the caching allocator could reuse that block before NCCL finished reading it,
so the gather delivered torn parameters (each rank keeps only its own shard correct) -> loss plateaus at 4-11 nats until the
next gather repairs it. Seen as five 'loss spikes' around step 27000 of the 1.5B run on two different hosts."""
import sys
p = "nanochat/optim.py"; s = open(p).read()
if "gather_inputs_alive" in s:
    print("optim: async-collective lifetime fix already present"); sys.exit(0)
reps = [
    # muon gather: keep work + input alive
    ('        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()\n'
     '        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))\n',
     '        work = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True); future = work.get_future()\n'
     '        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params, work=work, gather_inputs_alive=updated_params))\n'),
    # adamw gather: keep work alive (input p_slice is a view of p, output is p)
    ('                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()\n'
     '                gather_list.append(dict(future=future, params=None))\n',
     '                work = dist.all_gather_into_tensor(p, p_slice, async_op=True); future = work.get_future()\n'
     '                gather_list.append(dict(future=future, params=None, work=work, gather_inputs_alive=p_slice))\n'),
    # reduce ops: keep work objects alive until their waits
    ('                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()\n'
     '                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)\n',
     '                work = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True); future = work.get_future()\n'
     '                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True, work=work)\n'),
    ('                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()\n'
     '                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)\n',
     '                work = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True); future = work.get_future()\n'
     '                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False, work=work)\n'),
    ('        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()\n'
     '\n'
     '        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)\n',
     '        work = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True); future = work.get_future()\n'
     '\n'
     '        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size, work=work, grad_stack_alive=grad_stack)\n'),
]
for old, new in reps:
    assert s.count(old) == 1, "pattern not found:\n" + old
    s = s.replace(old, new)
# and make the final copy-back wait on the Work object too (device-side sync on the current stream), not only the future
old_fin = ('            if info["future"] is not None:\n'
           '                info["future"].wait()\n'
           '            if info["params"] is not None:\n')
new_fin = ('            if info["future"] is not None:\n'
           '                info["future"].wait()\n'
           '                if info.get("work") is not None: info["work"].wait()\n'
           '            if info["params"] is not None:\n')
assert s.count(old_fin) == 1; s = s.replace(old_fin, new_fin)
# v2: optionally make every collective synchronous (default ON via NANOCHAT_SYNC_COLLECTIVES=1). With async_op=False the
# call returns None, and every consumer already guards `if future is not None`, so no other change is needed.
s = s.replace("import torch.distributed as dist\n", "import os\nimport torch.distributed as dist\n_ASYNC = os.environ.get('NANOCHAT_SYNC_COLLECTIVES', '1') != '1'   # IbnSina patch: sync collectives by default\n", 1)
n_async = s.count("async_op=True)")
s = s.replace("async_op=True)", "async_op=_ASYNC)")
s = s.replace("; future = work.get_future()", "; future = work.get_future() if work is not None else None")
open(p, "w").write(s); print(f"optim: async-collective lifetime fix applied; {n_async} collectives switched to async_op=_ASYNC (sync by default)")
