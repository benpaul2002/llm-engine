#include <torch/torch.h>

__global__ void attn_kernel(at::BFloat16* q, at::BFloat16* kpool, at::BFloat16* vpool, float* scores, at::BFloat16* out, int* table, int* lengths, int max_blocks, int block_size, int max_len, int num_q_heads, int num_kv_heads, int head_dim) {
    int p = blockIdx.x;
    int i = threadIdx.x;
    if(i>=head_dim) {
        return;
    }

    int seq = p/num_q_heads;
    int q_head = p%num_q_heads;
    int kv_head = q_head/(num_q_heads/num_kv_heads);

    int l = lengths[seq];
    q += p*head_dim;
    scores += p*max_len;
    out += p*head_dim;

    __shared__ float partial[128];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int num_warps = head_dim/32;
    for(int t=warp; t<l; t+=num_warps) {
        float val = 0;
        int cache_block = table[seq*max_blocks + t/block_size];
        int slot = t % block_size;
        for(int d=lane; d<head_dim; d+=32) {
            val += (float)q[d] * (float)kpool[((cache_block*block_size + slot)*num_kv_heads + kv_head)*head_dim + d];
        }
        for(int offset=16; offset>=1; offset/=2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(lane==0) {
            scores[t] = val/sqrtf(head_dim);
        }
    }
    __syncthreads();

    float imax = -INFINITY;
    for (int t=i; t<l; t+=head_dim) {
        imax = fmaxf(imax, scores[t]);
    }
    partial[i] = imax;
    for(int j=head_dim/2; j>=1; j/=2) {
        __syncthreads();
        if(i<j) {
            partial[i] = fmaxf(partial[i], partial[i+j]);
        }
    }
    __syncthreads();

    imax = partial[0];
    __syncthreads();
    float sum = 0;
    for(int t=i; t<l; t+=head_dim) {
        sum += expf(scores[t]-imax);
    }
    partial[i] = sum;
    for(int j=head_dim/2; j>=1; j/=2) {
        __syncthreads();
        if(i<j) {
            partial[i] += partial[i+j];
        }
    }
    __syncthreads();

    sum = partial[0];
    __syncthreads();
    for(int t=i; t<l; t+=head_dim) {
        scores[t] = expf(scores[t]-imax)/sum;
    }
    
    __syncthreads();
    float acc = 0;
    for(int t=0; t<l; t++) {
        int cache_block = table[seq*max_blocks + t/block_size];
        int slot = t % block_size;
        acc += scores[t] * vpool[((cache_block*block_size + slot)*num_kv_heads + kv_head)*head_dim + i];
    }
    out[i] = acc;
}

void attention(torch::Tensor q, torch::Tensor kpool, torch::Tensor vpool, torch::Tensor table, torch::Tensor lengths, torch::Tensor out) {
    TORCH_CHECK(kpool.size(3)==64 || kpool.size(3)==128, "kernel assumes head_dim == 64 or 128");

    int block_size = kpool.size(1);
    int num_kv_heads = kpool.size(2);
    int head_dim = kpool.size(3);
    int num_q_heads = q.size(1);
    int num_seqs = lengths.size(0);
    int max_blocks = table.size(1);
    int max_len = max_blocks * block_size;

    auto scores = torch::empty({num_seqs * num_q_heads * max_len}, q.options().dtype(torch::kFloat32));

    attn_kernel<<<num_seqs * num_q_heads, head_dim>>>(q.data_ptr<at::BFloat16>(), kpool.data_ptr<at::BFloat16>(), vpool.data_ptr<at::BFloat16>(), scores.data_ptr<float>(), out.data_ptr<at::BFloat16>(), table.data_ptr<int>(), lengths.data_ptr<int>(), max_blocks, block_size, max_len, num_q_heads, num_kv_heads, head_dim);
}
