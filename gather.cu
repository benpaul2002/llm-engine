#include <iostream>

using namespace std;

__global__ void fn(int* pool, int* table, int* lengths, int* out, int total, int block_size, int num_heads, int head_dim, int max_len, int max_blocks) {
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

int main() {
    int block_size = 4;
    int num_heads = 2;
    int head_dim = 3;
    int max_len = 6;
    int max_blocks = 2;
    int num_seqs = 2;
    int total = num_seqs * num_heads * max_len * head_dim;

    int pool[192];
    for(int i=0; i<192; i++) {
        pool[i] = i+1;
    }

    int table[4] = {5, 1, 2, -1};
    int lengths[2] = {6, 3};

    int* gpool;
    int* gtable;
    int* glengths;
    int* gout;
    cudaMalloc(&gpool, 192*sizeof(int));
    cudaMalloc(&gtable, 4*sizeof(int));
    cudaMalloc(&glengths, 2*sizeof(int));
    cudaMalloc(&gout, total*sizeof(int));

    cudaMemcpy(gpool, pool, 192*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gtable, table, 4*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(glengths, lengths, 2*sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int gpu_blocks = (total+255)/threads;
    fn<<<gpu_blocks, threads>>>(gpool, gtable, glengths, gout, total, block_size, num_heads, head_dim, max_len, max_blocks);

    int* res = new int[total];
    cudaMemcpy(res, gout, total*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<total; i++) {
        cout<<res[i]<<" ";
    }

    return 0;
}