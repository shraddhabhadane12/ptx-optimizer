/*
 * reduction.cu — Parallel sum reduction kernel
 *
 * Compile to PTX with:
 *   nvcc -ptx -arch=sm_89 reduction.cu -o ../ptx_files/reduction.ptx
 *
 * Reduction kernels generate many warp-shuffle and shared-memory
 * instructions — excellent for showing copy elimination and
 * dead-code removal on realistic GPU patterns.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void reduce(const float* __restrict__ input,
                       float* __restrict__ output,
                       int N)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid    = threadIdx.x;
    int global = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global memory (0 for out-of-bounds threads)
    sdata[tid] = (global < N) ? input[global] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main()
{
    const int N = 1024;
    size_t size = N * sizeof(float);

    float *h_in  = (float*)malloc(size);
    float *h_out = (float*)malloc(N / BLOCK_SIZE * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  size);
    cudaMalloc(&d_out, N / BLOCK_SIZE * sizeof(float));
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce<<<blocks, BLOCK_SIZE>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; i++) total += h_out[i];
    printf("Sum = %.1f (expected %.1f)\n", total, (float)N);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
