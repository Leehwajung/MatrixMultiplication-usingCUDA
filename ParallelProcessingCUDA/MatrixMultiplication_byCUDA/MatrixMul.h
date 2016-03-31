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
int matrixMultiplyUsingCUDA(const int blockSize, const dim3 &dimsA, const dim3 &dimsB);

/**
 * Run a matrix multiplication using CPU
 */
int matrixMultiplyUsingCPU(const dim3 &dimsA, const dim3 &dimsB);

/**
 * Matrix multiplication on the CPU: C = A * B
 * wC is C's width, hC is C's height and wA is A's width
 */
void matrixMulCPU(float* C, float* A, float* B, int wC, int hC, int wA);

/**
 * Matrix element multiplication on the CPU
 * wA is A's width and wB is B's width
 */
float matrixSubMulCPU(int xC, int yC, float* A, float* B, int wA, int wB);

/**
 * Check result of multiplication
 */
bool CheckResult(const dim3& dimsA, const dim3& dimsC, const DATA* h_C);

/**
 * Initialize DATA array
 */
void constantInit(DATA dataArr[], const int arrSize, const DATA value);