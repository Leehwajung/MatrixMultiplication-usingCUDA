#pragma once

// CUDA runtime
#include <cuda_runtime.h>

// For to access "MatrixMul_kernel.cu"
#define CU_MatrixMul_Kernel			"matrixMul_kernel.cu"
#define FN_matrixMulCUDA_block16	"matrixMulCUDA_block16"
#define FN_matrixMulCUDA_block32	"matrixMulCUDA_block32"

// Matrix data type
typedef float DATA;

// Prototypes

/**
 * Run a matrix multiplication using CUDA
 */
int matrixMultiplyUsingCUDA(const int block_size, const dim3 &dimsA, const dim3 &dimsB);

/**
 * Initialize DATA array
 */
void constantInit(DATA dataArr[], const int arrSize, const DATA value);