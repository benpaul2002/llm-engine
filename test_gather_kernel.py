# Temporary check: does the CUDA gather kernel match KVCache.gather_decode exactly?
import os
import types

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

import torch
from torch.utils.cpp_extension import load_inline

from cache import KVCache, Sequence

mod = load_inline(
    name="gather_ext",
    cpp_sources="void gather(torch::Tensor pool, torch::Tensor table, torch::Tensor lengths, torch::Tensor out);",
    cuda_sources=open("gather_kernel.cu").read(),
    functions=["gather"],
)

# 1. A real cache: one layer, real shapes (128 blocks, 16 slots, 2 KV heads, 64 dims).
config = types.SimpleNamespace(num_hidden_layers=1, num_key_value_heads=2, head_dim=64)
cache = KVCache.preallocate(config, "cuda")

# 2. Random contents. preallocate uses torch.empty, which may hold NaN (and NaN != NaN).
cache.k[0].copy_(torch.randn_like(cache.k[0]))
cache.v[0].copy_(torch.randn_like(cache.v[0]))

# 3. Grow sequences one token at a time, taking turns, so their blocks interleave in the pool.
targets = [5, 40, 17, 33]
seqs = [Sequence() for _ in targets]
while any(s.length < t for s, t in zip(seqs, targets)):
    for s, t in zip(seqs, targets):
        if s.length < t:
            cache.ensure_capacity(s, 1)
            s.length += 1

for s in seqs:
    print(f"length {s.length:3d}  block_table {s.block_table}")

# 4. Reference: what the engine computes today.
plan = cache.plan_decode(seqs, "cuda")
k_ref, v_ref, _ = cache.gather_decode(plan, 0)

# 5. Kernel inputs from the same sequences.
lengths = torch.tensor([s.length for s in seqs], dtype=torch.int32, device="cuda")
max_blocks = max(len(s.block_table) for s in seqs)
table = torch.tensor(
    [s.block_table + [-1] * (max_blocks - len(s.block_table)) for s in seqs],
    dtype=torch.int32, device="cuda",
)

# 6. Run the kernel on K and on V, compare exactly.
for name, pool, ref in [("K", cache.k[0], k_ref), ("V", cache.v[0], v_ref)]:
    out = torch.empty_like(ref)
    mod.gather(pool, table, lengths, out)
    torch.cuda.synchronize()
    print(f"{name}: shape {tuple(out.shape)}  exact match: {torch.equal(out, ref)}")
