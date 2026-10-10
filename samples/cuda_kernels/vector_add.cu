/*
 * vector_add.cu — Element-wise vector addition kernel
 *
 * Compile to PTX with:
 *   nvcc -ptx -arch=sm_89 vector_add.cu -o ../ptx_files/vector_add.ptx
 *
 * Then optimize:
 *   python -m src.optimizer samples/ptx_files/vector_add.ptx --stats -v
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// Host code to test the kernel
int main()
{
    const int N = 1024;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0]   = %.1f (expected %.1f)\n", h_C[0],   h_A[0]   + h_B[0]);
    printf("C[512] = %.1f (expected %.1f)\n", h_C[512], h_A[512] + h_B[512]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
