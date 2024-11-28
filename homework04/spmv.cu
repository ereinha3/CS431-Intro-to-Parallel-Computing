#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"

#define BLOCKDIM 64

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz, 
                double* x, double* b)
{    
    T thread_sum = 0.0;

    for (unsigned int i = threadIdx.x; i<n; i += BLOCKDIM){
        if ((blockIdx.x * n + i) < (m * n)) { // Check to ensure within bounds
            unsigned int col = col_ind[blockIdx.x * n + i];
            thread_sum += (vals[blockIdx.x * n + i] * x[col]);
        }
    }

    __shared__ T shared_mem[BLOCKDIM];

    shared_mem[threadIdx.x] = thread_sum;

    __syncthreads();


    // parallel reduction
    for (unsigned int i = (BLOCKDIM >> 1); i > 0; i >>= 1)
    {
        if (threadIdx.x < i){
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        b[blockIdx.x] = shared_mem[0];
    }

}



void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int n_new, int nnz, 
                  double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    unsigned int blocks = m; 
    unsigned int threads = BLOCKDIM; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, n_new, nnz, x, b);
        printf("Kernel passed %d times\n", i+1);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            break;
        }
    } 
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}




void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, int n_new,
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // x -> n
    // b -> m
    CUDA_CHECK(cudaMalloc(dev_col_ind, m * n_new * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(dev_vals, m * n_new * sizeof(double)));
    CUDA_CHECK(cudaMalloc(dev_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(dev_b, m * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(*dev_col_ind, col_ind, m * n_new * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*dev_vals, vals, m * n_new * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(*dev_b, 0, m * sizeof(double)));

    // col_ind back to the host and print
    // unsigned int* host_col_ind = (unsigned int*)malloc(m * n_new * sizeof(unsigned int));
    // CUDA_CHECK(cudaMemcpy(host_col_ind, *dev_col_ind, m * n_new * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < (m * n_new); i++) {
    //     if (i % n_new == 0) {
    //         fprintf(stdout, "\n");
    //     }
    //     fprintf(stdout, "%d ", host_col_ind[i]);
    // }

    // free(host_col_ind);

    // unsigned int* host_vals = (unsigned int*)malloc(m * n_new * sizeof(unsigned int));
    // CUDA_CHECK(cudaMemcpy(host_vals, *dev_vals, m * n_new * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < (m * n_new); i++) {
    //     if (i % n_new == 0) {
    //         fprintf(stdout, "\n");
    //     }
    //     fprintf(stdout, "%d ", host_vals[i]);
    // }

    // free(host_vals);
}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{

    cudaMalloc(dev_row_ptr, (m+1) * sizeof(unsigned int));
    cudaMalloc(dev_col_ind, nnz * sizeof(unsigned int));
    cudaMalloc(dev_vals, nnz * sizeof(double));
    cudaMalloc(dev_x, n * sizeof(double));
    cudaMalloc(dev_b, m * sizeof(double));

    cudaMemcpy(*dev_row_ptr, row_ptr, (m+1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_col_ind, col_ind, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m, 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost( &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc( d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals, 
            int m, int n, int nnz, double* x, double* b)
{
    int row = blockIdx.x;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    int num_threads = blockDim.x;

    T temp = 0.0;

    for (int idx = row_start + threadIdx.x; idx < row_end; idx += num_threads) {
        unsigned int col = col_ind[idx]; 
        T value = vals[idx];             
        temp += (value * x[col]); 
    }

    __shared__ double shared_mem[BLOCKDIM];

    shared_mem[threadIdx.x] = temp;

    __syncthreads();

    // parallel reduction
    for (int i = (num_threads >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + i];
        }
    }
    
    __syncthreads();

    // result will all be in first thread
    if (threadIdx.x == 0) {
        b[row] = shared_mem[0];
    }
}



void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    // 1 thread block per row
    // 64 threads working on the non-zeros on the same row
    unsigned int blocks = m; 
    unsigned int threads = BLOCKDIM; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, 
                                                           vals, m, n, nnz, 
                                                           x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}
