#pragma once

// CUDA runtime
#include <cuda_runtime.h>

// Matrix data type
typedef float DATA;

// Prototypes

/**
 * Run a matrix multiplication using CUDA
 */
int matrixMultiplyOnCUDA(const int blockSize, const dim3 &dimsA, const dim3 &dimsB);

/**
 * Run a matrix multiplication using CPU
 */
int matrixMultiplyOnCPU(const dim3 &dimsA, const dim3 &dimsB);

/**
 * Matrix multiplication on the CPU: C = A * B
 * wC is C's width, hC is C's height and wA is A's width
 */
void matrixMulCPU(float* C, float* A, float* B, int wC, int hC, int wA);

/**
 * Check result of multiplication
 */
bool CheckResult(const dim3& dimsA, const dim3& dimsC, const DATA* h_C);

/**
 * Initialize DATA array
 */
void constantInit(DATA dataArr[], const int arrSize, const DATA value);