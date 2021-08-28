#include <iostream>
#include <stdio.h>
#define BLK_SIZE 128


__global__ void kernel(){
	printf("Hello World!");


}

__global__ void print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}


int main(){	
    print_kernel<<<10, 10>>>();
    cudaDeviceSynchronize();

return 0;
}
