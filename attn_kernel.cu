#include <torch/torch.h>

__global__ void attn_kernel(at::BFloat16* q, at::BFloat16* kpool, at::BFloat16* vpool, float* scores, float* part_out, float* part_max, float* part_sum, int* table, int* lengths, int max_blocks, int block_size, int max_len, int num_q_heads, int num_kv_heads, int head_dim, int num_splits) {
    int p = blockIdx.x / num_splits;
    int split = blockIdx.x % num_splits;
    int slot = p*num_splits + split;
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

    int chunk = (l + num_splits - 1) / num_splits;
    int start = split * chunk;
    int end = min(start + chunk, l);

    __shared__ float partial[128];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int num_warps = head_dim/32;
    for(int t=start+warp; t<end; t+=num_warps) {
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
    for (int t=start+i; t<end; t+=head_dim) {
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
    for(int t=start+i; t<end; t+=head_dim) {
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
    for(int t=start+i; t<end; t+=head_dim) {
        scores[t] = expf(scores[t]-imax);
    }
    
    __syncthreads();
    float acc = 0;
    for(int t=start; t<end; t++) {
        int cache_block = table[seq*max_blocks + t/block_size];
        int slot = t % block_size;
        acc += scores[t] * vpool[((cache_block*block_size + slot)*num_kv_heads + kv_head)*head_dim + i];
    }
    part_out[slot*head_dim + i] = acc;

    if(i==0) {
        part_max[slot] = imax;
        part_sum[slot] = sum;
    }
}

__global__ void combine_kernel(float* part_out, float* part_max, float* part_sum, at::BFloat16* out, int head_dim, int num_splits) {
    int p = blockIdx.x;
    int i = threadIdx.x;
    float M = -INFINITY;
    for(int s=0; s<num_splits; s++) {
        M = fmaxf(M, part_max[p*num_splits + s]);
    }
    float total = 0;
    float o = 0;
    for(int s=0; s<num_splits; s++) {
        float scale = expf(part_max[p*num_splits + s] - M);
        total += part_sum[p*num_splits + s] * scale;
        o += part_out[(p*num_splits + s)*head_dim + i] * scale;
    }
    out[p*head_dim + i] = o/total;
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

    int num_splits = 4;

    auto scores = torch::empty({num_seqs * num_q_heads * max_len}, q.options().dtype(torch::kFloat32));
    auto part_out = torch::empty({num_seqs * num_q_heads * num_splits * head_dim}, q.options().dtype(torch::kFloat32));
    auto part_max = torch::empty({num_seqs * num_q_heads * num_splits}, q.options().dtype(torch::kFloat32));
    auto part_sum = torch::empty({num_seqs * num_q_heads * num_splits}, q.options().dtype(torch::kFloat32));

    attn_kernel<<<num_seqs * num_q_heads * num_splits, head_dim>>>(q.data_ptr<at::BFloat16>(), kpool.data_ptr<at::BFloat16>(), vpool.data_ptr<at::BFloat16>(), scores.data_ptr<float>(), part_out.data_ptr<float>(), part_max.data_ptr<float>(), part_sum.data_ptr<float>(), table.data_ptr<int>(), lengths.data_ptr<int>(), max_blocks, block_size, max_len, num_q_heads, num_kv_heads, head_dim, num_splits);

    combine_kernel<<<num_seqs * num_q_heads, head_dim>>> (part_out.data_ptr<float>(), part_max.data_ptr<float>(), part_sum.data_ptr<float>(), out.data_ptr<at::BFloat16>(), head_dim, num_splits);
}
