/*
 * spmm_csc_driver.cu
 * Copyright (C) 2020
 *  Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#include "mm_helper.hpp"
#include "sparse_representation.hpp"
#include <iostream>
#define BLK_SIZE 32


void check_dmat(double* a, double *b, unsigned int n, unsigned int K, bool quit_on_err = true ) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            if(std::abs(a[i * K + k] - b[i * K + k]) > 1e-1) {
                std::cerr << "Possible error at " << i << std::endl;

                if(quit_on_err) {
                    exit(-1);
                }
            }
        }
    }
    if(quit_on_err)
        std::cout << "Verification succeeded\n";
    else
        std::cout << "Check error messages to see if verification succeeded. (No error msg == success)\n";
}

static unsigned int g_seed = 0X4B1D;
inline int fastrand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

void init_dmat(double *a, unsigned int n, unsigned int K, double offset) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            a[i * K + k]  = i * K + k + offset;
            //a[i * K + j]  = fastrand() + offset;
        }
    }
}

void print_dmat(double *a, unsigned int n, unsigned int K) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < K; ++j) {
            std::cout << a[i * K + j]   << ' ';
        }
        std::cout << '\n';
    }
}

void host_csc_spmm(CSC mat, double * dmat_in, double * dmat_out, unsigned int K) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        for (unsigned int k = 0; k < K; ++k) {
            dmat_out[r * K + k] = 0;
        }
    }
    for (unsigned int c = 0; c < mat.ncols; ++c) {
        unsigned int col_start = mat.col_indx[c];
        unsigned int col_end = mat.col_indx[c + 1];

        for (unsigned int r = col_start; r < col_end; ++r) {
            unsigned int row_id = mat.row_id[r];
            double val = mat.values[r];

            for (unsigned int k = 0; k < K; ++k) {
                dmat_out[row_id * K + k] += val * dmat_in[c * K + k];
            }
        }

    }
}

__global__ void dev_csc_spmm(double *values, int *row_id, int *col_indx, int nnz, int ncols, int nrows, int K, const double *D, double *O){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < K && col < nrows) {
       double dotProduct = 0;
       const int col_start = col_indx[col];
       const int col_end = col_indx[col+1];
       

       for (int element = col_start; element < col_end; ++element) {

	  dotProduct += values[element] * D[K * row_id[element] + row];
        
       }
    
       O[col * K + row] = dotProduct;
    
    }

}



int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    unsigned int K = std::atoi(argv[2]);
    CSC mat = read_matrix_market_to_CSC(argv[1]);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << K << '\n';

    double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));
    double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));

    init_dmat(dmat_in, mat.ncols, K, 1.0);
    //print_dmat(dmat_in, mat.ncols, K);

    host_csc_spmm(mat, dmat_in, dmat_out, K);
    //device array pointers
    double *d_values;
    int *d_row_id;
    int *d_col_indx;
    double *d_dmat_in;
    double *d_dmat_out;


    cudaMalloc(&d_values, sizeof(double)* mat.nnz);
    cudaMalloc(&d_row_id, sizeof(int) * (mat.nnz));
    cudaMalloc(&d_col_indx, sizeof(int)* mat.ncols+1);
    cudaMalloc(&d_dmat_in, sizeof(double)* K * mat.ncols);
    cudaMalloc(&d_dmat_out, sizeof(double)* K * (mat.nrows));

    //----------- Begin kernel call for SpMM_CSR ------------
    float time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //cudamemcopy functions
    cudaMemcpy(d_values, mat.values, sizeof(double) * mat.nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indx, mat.col_indx, sizeof(int) * mat.ncols+1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_id, mat.row_id, sizeof(int) * (mat.nnz), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dmat_in, dmat_in, sizeof(double)*K*mat.ncols, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_dmat_out, dmat_out, sizeof(double)*K*(mat.nrows), cudaMemcpyHostToDevice);

    //define blk and grid size
    dim3 threads(BLK_SIZE, BLK_SIZE);
    dim3 grid((int) ceil((float) K/BLK_SIZE), (int) ceil((float) mat.ncols/BLK_SIZE));


    //call gpu kernel
    dev_csc_spmm<<<grid, threads>>>(d_values, d_row_id, d_col_indx, mat.nnz, mat.ncols, mat.nrows, K, d_dmat_in, d_dmat_out);
    cudaDeviceSynchronize();

    //cudamemcopy gpu result from device to host
    double *gpu_result = (double*)malloc((mat.nrows) * K  * sizeof(double));

    cudaMemcpy(gpu_result, d_dmat_out, sizeof(double)*K*(mat.nrows), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    //compute GFLOPS
    double gflop = abs(((2 * K * mat.nnz)/1e9));
    double op_time_s = time_ms * 1e-3;
    double gflops = gflop/op_time_s;

    printf("Kernel time :  %f ms \n", time_ms);
    printf("GFLOPS :  %f \n", gflops);
   
    /*
    for(int i =0; i<mat.nrows*K; i++){
        printf("point: %d, gpu: %lf, cpu: %lf \n", i, gpu_result[i], dmat_out[i]);
    }
    */
   

    //std::cout << "replace one argument to the below function with the values from gpu " << std::endl;
    check_dmat(dmat_out, gpu_result, mat.nrows, K);

    //print_dmat(dmat_out, mat.nrows, K);


    free(mat.col_indx);
    free(mat.row_id);
    free(mat.values);
    free(gpu_result);

    cudaFree(d_values);
    cudaFree(d_col_indx);
    cudaFree(d_row_id);
    cudaFree(d_dmat_in);
    cudaFree(d_dmat_out);
    
    return 0;
}
