import torch
from torch.utils.cpp_extension import load_inline

pool = torch.arange(1, 193, dtype=torch.bfloat16, device="cuda").view(8, 4, 2, 3)
table = torch.tensor([[5, 1], [2, -1]], dtype=torch.int32, device="cuda")
lengths = torch.tensor([6, 3], dtype=torch.int32, device="cuda")

out = torch.empty(2, 2, 6, 3, dtype=torch.bfloat16, device="cuda")

mod = load_inline(
    name="gather_ext",
    cpp_sources="void gather(torch::Tensor pool, torch::Tensor table, torch::Tensor lengths, torch::Tensor out);",
    cuda_sources=open("gather_kernel.cu").read(),
    functions=["gather"],
)

mod.gather(pool, table, lengths, out)
print(out)
