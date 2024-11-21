#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"

template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz, 
                double* x, double* b)
{

    // COMPLETE THIS FUNCTION
}



void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
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
    unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, nnz, x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}




void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // x -> n
    // b -> m
    cudaError_t error = cudaMalloc((void**)dev_col_ind, nnz * sizeof(unsigned int));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_col_ind malloc' << cudaGetErrorString(err) << std::endl;
    }
    error = cudaMalloc((void**)dev_vals, nnz * sizeof(double));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_vals malloc' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_col_ind);
    }
    error = cudaMalloc((void**)dev_x, n * sizeof(double));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_x malloc' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
    }
    error = cudaMalloc((void**)dev_b, m * sizeof(double));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_b malloc' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
    }
    error = cudaMemcpy(*dev_col_ind, col_ind, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_col_ind memcpy' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
        cudaFree(*dev_b);
        return;
    }
    error = cudaMemcpy(*dev_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_vals memcpy' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
        cudaFree(*dev_b);
        return;
    }
    error = cudaMemcpy(*dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_x memcpy' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
        cudaFree(*dev_b);
        return;
    }

    // copy ELL data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION
}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{

    cudaError_t error = cudaMalloc((void**)dev_row_ptr, (m+1) * sizeof(unsigned int));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_row_ptr malloc' << cudaGetErrorString(err) << std::endl;
    }
    error = cudaMalloc((void**)dev_col_ind, nnz * sizeof(unsigned int));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_vals malloc' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_row_ptr);
    } 
    error = cudaMalloc((void**)dev_vals, nnz * sizeof(double));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_x malloc' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_row_ptr);
        cudaFree(*dev_col_ind);
    }
    error = cudaMalloc((void**)dev_x, n * sizeof(double));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_x malloc' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_row_ptr);
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
    }
    error = cudaMalloc((void**)dev_b, m * sizeof(double));
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_b malloc' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_row_ptr);
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
    }
    error = cudaMemcpy(*dev_col_ind, col_ind, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_col_ind memcpy' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_row_ptr);
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
        cudaFree(*dev_b);
        return;
    }
    error = cudaMemcpy(*dev_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_vals memcpy' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_row_ptr);
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
        cudaFree(*dev_b);
        return;
    }
    error = cudaMemcpy(*dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std:cerr << 'failed at dev_x memcpy' << cudaGetErrorString(err) << std::endl;
        cudaFree(*dev_row_ptr);
        cudaFree(*dev_col_ind);
        cudaFree(*dev_vals);
        cudaFree(*dev_x);
        cudaFree(*dev_b);
        return;
    }

    // copy CSR data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION
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
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
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
    extern __shared__ double shared_sum[];

    // Get the row index this block is responsible for
    int row = blockIdx.x;

    // Determine the start and end indices in the CSR format
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    
    // Compute the thread ID within the block
    int thread_id = threadIdx.x;

    // Initialize shared memory for the reduction
    shared_sum[thread_id] = 0.0;

    // Iterate over the elements in the row that this thread should handle
    for (int idx = row_start + thread_id; idx < row_end; idx += blockDim.x) {
        unsigned int col = col_ind[idx]; // Get the column index
        T value = vals[idx];             // Get the value in the matrix
        shared_sum[thread_id] += value * x[col]; // Compute partial dot product
    }

    // Synchronize to make sure all threads have completed the summation
    __syncthreads();

    // Perform reduction to get the final dot product result for the row
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared_sum[thread_id] += shared_sum[thread_id + stride];
        }
        __syncthreads();
    }

    // The result of the reduction is in shared_sum[0], so store it in b[row]
    if (thread_id == 0) {
        b[row] = shared_sum[0];
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
    unsigned int threads = 64; 
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
