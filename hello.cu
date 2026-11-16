#include <iostream>

using namespace std;

__global__ void fn(int* arr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<1000) {
        arr[i] = i;
    }
}

int main() {
    int* arr;
    cudaMalloc(&arr, 1000*sizeof(int));
    fn<<<4, 256>>>(arr);
    int check[1000];
    cudaMemcpy(check, arr, 1000*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<1000; i++) {
        if(check[i]!=i) {
            cout<<"Error!"<<endl;
            return;
        }
    }
    cout<<"Passed!"<<endl;
}