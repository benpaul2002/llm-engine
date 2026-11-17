#include <iostream>

using namespace std;

__global__ void fn(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=64) {
        return;
    }
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int val = threadIdx.x + 1;
    for(int offset=16; offset>=1; offset/=2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(lane==0) {
        out[warp] = val;
    }
}

int main() {
    int out[2];
    int* gout; 
    cudaMalloc(&gout, 2*sizeof(int));
    fn<<<1, 64>>>(gout);
    cudaMemcpy(out, gout, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cout<<out[0]<<" "<<out[1]<<endl;
    return 0;
}