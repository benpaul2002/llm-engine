#include <torch/torch.h>

__global__ void fn(at::BFloat16* pool, int* table, int* lengths, at::BFloat16* out, int total, int block_size, int num_heads, int head_dim, int max_len, int max_blocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=total) {
        return;
    }
    int dim = i%head_dim;
    int rest = i/head_dim;
    int token = rest%max_len;
    rest /= max_len;
    int head = rest%num_heads;
    int seq = rest/num_heads;
    if(token>=lengths[seq]) {
        out[i] = 0;
        return;
    }
    int entry = token/block_size;
    int slot = token%block_size;
    int cache_block = table[seq*max_blocks + entry];
    int pool_pos = ((cache_block * block_size + slot) * num_heads + head) * head_dim + dim;
    out[i] = pool[pool_pos];
}

void gather(torch::Tensor pool, torch::Tensor table, torch::Tensor lengths, torch::Tensor out) {
    int block_size = pool.size(1);
    int num_heads = pool.size(2);
    int head_dim = pool.size(3);
    int max_blocks = table.size(1);
    int max_len = out.size(2);
    int total = out.numel();

    int threads = 256;
    int gpu_blocks = (total+255)/threads;
    fn<<<gpu_blocks, threads>>>(pool.data_ptr<at::BFloat16>(), table.data_ptr<int>(), lengths.data_ptr<int>(), out.data_ptr<at::BFloat16>(), total, block_size, num_heads, head_dim, max_len, max_blocks);
}
