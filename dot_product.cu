#include <iostream>

using namespace std;

__global__ void dot(float* q, float* kpool, float* vpool, float* scores, float* out, int* table, int* lengths, int max_blocks, int block_size, int max_len, int num_q_heads, int num_kv_heads) {
    int p = blockIdx.x;
    int i = threadIdx.x;
    if(i>=64) {
        return;
    }

    int seq = p/num_q_heads;
    int q_head = p%num_q_heads;
    int kv_head = q_head/(num_q_heads/num_kv_heads);

    int l = lengths[seq];
    q += p*64;
    scores += p*max_len;
    out += p*64;

    __shared__ float partial[64];

    for(int t=0; t<l; t++) {
        int cache_block = table[seq*max_blocks + t/block_size];
        int slot = t % block_size;
        partial[i] = q[i] * kpool[((cache_block*block_size + slot)*num_kv_heads + kv_head)*64 + i];
        for(int j=32; j>=1; j/=2) {
            __syncthreads();
            if(i<j) {
                partial[i] += partial[i+j];
            }
        }
        __syncthreads();
        if(i==0) {
            scores[t] = partial[0]/8;
        }
    }
    __syncthreads();

    float imax = -INFINITY;
    for (int t=i; t<l; t+=64) {
        imax = fmaxf(imax, scores[t]);
    }
    partial[i] = imax;
    for(int j=32; j>=1; j/=2) {
        __syncthreads();
        if(i<j) {
            partial[i] = fmaxf(partial[i], partial[i+j]);
        }
    }
    __syncthreads();

    imax = partial[0];
    __syncthreads();
    float sum = 0;
    for(int t=i; t<l; t+=64) {
        sum += expf(scores[t]-imax);
    }
    partial[i] = sum;
    for(int j=32; j>=1; j/=2) {
        __syncthreads();
        if(i<j) {
            partial[i] += partial[i+j];
        }
    }
    __syncthreads();

    sum = partial[0];
    __syncthreads();
    for(int t=i; t<l; t+=64) {
        scores[t] = expf(scores[t]-imax)/sum;
    }
    
    __syncthreads();
    float acc = 0;
    for(int t=0; t<l; t++) {
        int cache_block = table[seq*max_blocks + t/block_size];
        int slot = t % block_size;
        acc += scores[t] * vpool[((cache_block*block_size + slot)*num_kv_heads + kv_head)*64 + i];
    }
    out[i] = acc;
}

int main() {
    int P = 2;
    int block_size = 4;
    int num_cache_blocks = 8;
    int max_blocks = 2;
    int max_len = 6;

    int num_q_heads = 4;
    int num_kv_heads = 2;

    float q[P*num_q_heads*64];
    for(int i=0; i<P*num_q_heads*64; i++) {
        q[i] = 1;
    }
    int pool_size = num_cache_blocks*block_size*num_kv_heads*64;
    float kpool[pool_size], vpool[pool_size];
    for(int cb=0; cb<num_cache_blocks; cb++) {
        for(int s=0; s<block_size; s++) {
            for(int h=0; h<num_kv_heads; h++) {
                for(int d=0; d<64; d++) {
                    int idx = ((cb*block_size + s)*num_kv_heads + h)*64 + d;
                    kpool[idx] = 0;
                    vpool[idx] = 10*cb + s + d + 1000*h;
                }
            }
        }
    }

    for(int d=0; d<64; d++) {
        kpool[((1*block_size + 0)*num_kv_heads + 0)*64 + d] = 8;
        kpool[((2*block_size + 2)*num_kv_heads + 0)*64 + d] = 8;
        kpool[((2*block_size + 3)*num_kv_heads + 0)*64 + d] = 16;
        kpool[((5*block_size + 0)*num_kv_heads + 1)*64 + d] = 8;
        kpool[((2*block_size + 0)*num_kv_heads + 1)*64 + d] = 8;
        kpool[((2*block_size + 3)*num_kv_heads + 1)*64 + d] = 16;
    }

    int table[4] = {5, 1, 2, -1};
    int lengths[2] = {6, 3};
    
    float out[P*num_q_heads*64];

    float* gq;
    float* gkpool;
    float* gvpool;
    int* gtable;
    int* glengths;
    float* gscores;
    float* gout;
    cudaMalloc(&gq, P*num_q_heads*64*sizeof(float));
    cudaMalloc(&gkpool, pool_size*sizeof(float));
    cudaMalloc(&gvpool, pool_size*sizeof(float));
    cudaMalloc(&gtable, P*max_blocks*sizeof(int));
    cudaMalloc(&glengths, P*sizeof(int));
    cudaMalloc(&gscores, P*num_q_heads*max_len*sizeof(float));
    cudaMalloc(&gout, P*num_q_heads*64*sizeof(float));

    cudaMemcpy(gq, q, P*num_q_heads*64*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gkpool, kpool, pool_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gvpool, vpool, pool_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gtable, table, P*max_blocks*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(glengths, lengths, P*sizeof(int), cudaMemcpyHostToDevice);
    
    dot<<<P*num_q_heads, 64>>>(gq, gkpool, gvpool, gscores, gout, gtable, glengths, max_blocks, block_size, max_len, num_q_heads, num_kv_heads);
    
    cudaMemcpy(out, gout, P*num_q_heads*64*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<64; i++) {
        for(int p=0; p<P*num_q_heads; p++) {
            cout<<out[p*64 + i]<<" ";
        }
        cout<<endl;
    }
    return 0;
}