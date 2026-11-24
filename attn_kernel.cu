#include <torch/torch.h>

template <int HD>
__global__ void attn_kernel(at::BFloat16* q, at::BFloat16* kpool, at::BFloat16* vpool, float* part_out, float* part_max, float* part_sum, int* table, int* lengths, int max_blocks, int block_size, int num_q_heads, int num_kv_heads, int num_splits) {
    constexpr int TILE = 4096 / HD;
    constexpr int VEC = 8;
    constexpr int CHUNKS = HD / VEC;
    int group = num_q_heads / num_kv_heads;

    extern __shared__ float smem[];
    float* ks = smem;
    float* vs = ks + TILE * (HD + 1);
    float* qs = vs + TILE * HD;
    float* sc = qs + group * HD;
    int* base = (int*)(sc + group * TILE);

    int p = blockIdx.x / num_splits;
    int split = blockIdx.x % num_splits;
    int seq = p / num_kv_heads;
    int kv_head = p % num_kv_heads;
    int g = threadIdx.x / HD;
    int i = threadIdx.x % HD;
    int qp = seq * num_q_heads + kv_head * group + g;

    int l = lengths[seq];
    int chunk = (l + num_splits - 1) / num_splits;
    int start = split * chunk;
    int end = min(start + chunk, l);

    float* myq = qs + g * HD;
    float* mysc = sc + g * TILE;
    myq[i] = (float)q[qp * HD + i];
    float m = -INFINITY, lsum = 0, acc = 0;

    for (int t0 = start; t0 < end; t0 += TILE) {
        int n = min(TILE, end - t0);
        if (threadIdx.x < n) {
            int t = t0 + threadIdx.x;
            base[threadIdx.x] = ((table[seq * max_blocks + t / block_size] * block_size + t % block_size) * num_kv_heads + kv_head) * HD;
        }
        __syncthreads();

        for (int u = threadIdx.x; u < n * CHUNKS; u += blockDim.x) {
            int tt = u / CHUNKS;
            int c = (u % CHUNKS) * VEC;
            uint4 kraw = *(uint4*)(kpool + base[tt] + c);
            uint4 vraw = *(uint4*)(vpool + base[tt] + c);
            at::BFloat16* ke = (at::BFloat16*)&kraw;
            at::BFloat16* ve = (at::BFloat16*)&vraw;
            for (int j = 0; j < VEC; j++) {
                ks[tt * (HD + 1) + c + j] = (float)ke[j];
                vs[tt * HD + c + j] = (float)ve[j];
            }
        }
        __syncthreads();

        float s = 0;
        if (i < n) {
            for (int d = 0; d < HD; d++) s += myq[d] * ks[i * (HD + 1) + d];
            s = s / sqrtf(HD);
            mysc[i] = s;
        }
        __syncthreads();

        float mt = m;
        for (int t = 0; t < n; t++) mt = fmaxf(mt, mysc[t]);
        __syncthreads();

        if (i < n) mysc[i] = expf(s - mt);
        __syncthreads();

        float corr = expf(m - mt);
        lsum *= corr;
        acc *= corr;
        for (int t = 0; t < n; t++) {
            float pt = mysc[t];
            lsum += pt;
            acc += pt * vs[t * HD + i];
        }
        m = mt;
        __syncthreads();
    }

    int slot = qp * num_splits + split;
    part_out[slot * HD + i] = acc;
    if (i == 0) {
        part_max[slot] = m;
        part_sum[slot] = lsum;
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

template <int HD>
void launch(torch::Tensor q, torch::Tensor kpool, torch::Tensor vpool, torch::Tensor table, torch::Tensor lengths, torch::Tensor part_out, torch::Tensor part_max, torch::Tensor part_sum, int num_splits) {
    constexpr int TILE = 4096 / HD;
    int num_q_heads = q.size(1);
    int num_kv_heads = kpool.size(2);
    int group = num_q_heads / num_kv_heads;
    int num_seqs = lengths.size(0);
    int smem = (TILE * (HD + 1) + TILE * HD + group * HD + group * TILE + TILE) * 4;
    attn_kernel<HD><<<num_seqs * num_kv_heads * num_splits, group * HD, smem>>>(q.data_ptr<at::BFloat16>(), kpool.data_ptr<at::BFloat16>(), vpool.data_ptr<at::BFloat16>(), part_out.data_ptr<float>(), part_max.data_ptr<float>(), part_sum.data_ptr<float>(), table.data_ptr<int>(), lengths.data_ptr<int>(), table.size(1), kpool.size(1), num_q_heads, num_kv_heads, num_splits);
}

void attention(torch::Tensor q, torch::Tensor kpool, torch::Tensor vpool, torch::Tensor table, torch::Tensor lengths, torch::Tensor out) {
    TORCH_CHECK(kpool.size(3)==64 || kpool.size(3)==128, "kernel assumes head_dim == 64 or 128");

    int head_dim = kpool.size(3);
    int num_q_heads = q.size(1);
    int num_seqs = lengths.size(0);
    int num_splits = 4;

    auto part_out = torch::empty({num_seqs * num_q_heads * num_splits * head_dim}, q.options().dtype(torch::kFloat32));
    auto part_max = torch::empty({num_seqs * num_q_heads * num_splits}, q.options().dtype(torch::kFloat32));
    auto part_sum = torch::empty({num_seqs * num_q_heads * num_splits}, q.options().dtype(torch::kFloat32));

    if (head_dim == 64) launch<64>(q, kpool, vpool, table, lengths, part_out, part_max, part_sum, num_splits);
    else launch<128>(q, kpool, vpool, table, lengths, part_out, part_max, part_sum, num_splits);

    combine_kernel<<<num_seqs * num_q_heads, head_dim>>> (part_out.data_ptr<float>(), part_max.data_ptr<float>(), part_sum.data_ptr<float>(), out.data_ptr<at::BFloat16>(), head_dim, num_splits);
}
