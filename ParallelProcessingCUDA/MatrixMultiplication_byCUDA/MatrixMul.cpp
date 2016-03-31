#include "MatrixMul.h"

// System includes
#include <iostream>
#include <cassert>

// CUDA runtime
#include "nvrtc_helper.h"
#include <cudaProfiler.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <cuda.h>

// Initial values of matrices A and B
#define	ValA	1.0f
#define	ValB	0.01f

// Repeat number of matrix multiplication
#define NLTER	300

// Machine zero
#define EPS		1.e-6

// Used names
using std::cout;
using std::cerr;
using std::endl;


int matrixMultiplyUsingCUDA(const int blockSize, const dim3 &dimsA, const dim3 &dimsB) {

	// Allocate host memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(DATA);
	DATA *h_A = (DATA*)malloc(memSizeA);

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(DATA);
	DATA *h_B = (DATA*)malloc(memSizeB);

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(DATA);
	DATA *h_C = (DATA*)malloc(mem_size_C);

	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		cerr << "Failed to allocate host matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize host memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);

	// Initialize device and get moudule
	char *kernelFile = sdkFindFilePath(CU_MatrixMul_Kernel, NULL);
	char *ptx;
	size_t ptxSize;
	compileFileToPTX(kernelFile, 0, NULL, &ptx, &ptxSize);
	CUmodule mod_kernel = loadPTX(ptx, 0, NULL);

	// Allocate device memory
	CUdeviceptr d_A, d_B, d_C;
	checkCudaErrors(cuMemAlloc(&d_A, memSizeA));
	checkCudaErrors(cuMemAlloc(&d_B, memSizeB));
	checkCudaErrors(cuMemAlloc(&d_C, mem_size_C));

	// copy host memory to device
	checkCudaErrors(cuMemcpyHtoD(d_A, h_A, memSizeA));
	checkCudaErrors(cuMemcpyHtoD(d_B, h_B, memSizeB));

	// Get kernel function
	cout << "Computing result using CUDA Kernel..." << endl;
	
	CUfunction fn_matrixMulCUDA;
	if (blockSize == 16) {
		checkCudaErrors(cuModuleGetFunction(&fn_matrixMulCUDA, mod_kernel, FN_matrixMulCUDA_block16));
	}
	else {
		checkCudaErrors(cuModuleGetFunction(&fn_matrixMulCUDA, mod_kernel, FN_matrixMulCUDA_block32));
	}

	// Setup execution parameters
	dim3 threads(blockSize, blockSize);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	void *args[] = {
		(void*)&d_C,
		(void*)&d_A,
		(void*)&d_B,
		(void*)&dimsA.x,
		(void*)&dimsB.x
	};

	// Execute the kernel with timer
	float totalLeadTime = 0;
	for (int i = 0; i < NLTER; i++) {
		StopWatchWin eachTimer;
		eachTimer.start();

		checkCudaErrors(cuLaunchKernel(fn_matrixMulCUDA,
			grid.x, grid.y, grid.z,				/* grid dim */
			threads.x, threads.y, threads.z,	/* block dim */
			0, 0,								/* shared mem, stream */
			&args[0],							/* arguments */
			0));

		eachTimer.stop();
		checkCudaErrors(cuCtxSynchronize());
		
		totalLeadTime += eachTimer.getTime();
		cout << "Lead time (" << i << "): " << eachTimer.getTime() << endl;
	}
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;

	// Copy result from device to host
	checkCudaErrors(cuMemcpyDtoH(h_C, d_C, mem_size_C));
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);

	checkCudaErrors(cuMemFree(d_A));
	checkCudaErrors(cuMemFree(d_B));
	checkCudaErrors(cuMemFree(d_C));

	cuProfilerStop();

	if (correct) {
		return EXIT_SUCCESS;
	}
	else {
		return EXIT_FAILURE;
	}
}

int matrixMultiplyUsingCPU(const dim3 &dimsA, const dim3 &dimsB) {

	// Allocate memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(DATA);
	DATA *h_A = (DATA*)malloc(memSizeA);

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(DATA);
	DATA *h_B = (DATA*)malloc(memSizeB);

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(DATA);
	DATA *h_C = (DATA*)malloc(mem_size_C);

	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		cerr << "Failed to allocate matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);

	// Invoke the function with timer
	cout << "Computing result using CPU..." << endl;

	float totalLeadTime = 0;
	for (int i = 0; i < NLTER; i++) {
		StopWatchWin eachTimer;
		eachTimer.start();

		matrixMulCPU(h_C, h_A, h_B, dimsC.x, dimsC.y, dimsA.x);

		eachTimer.stop();
		checkCudaErrors(cuCtxSynchronize());

		totalLeadTime += eachTimer.getTime();
		cout << "Lead time (" << i << "): " << eachTimer.getTime() << endl;
	}
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;

	// Check the result
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);

	if (correct) {
		return EXIT_SUCCESS;
	}
	else {
		return EXIT_FAILURE;
	}
}

void matrixMulCPU(float* C, float* A, float* B, int wC, int hC, int wA) {

	for (int i = 0; i< hC; i++) {
		for (int j = 0; j < wC; j++) {
			C[i * wC + j] = matrixSubMulCPU(i, j, A, B, wA, wC);
		}
	}
}

float matrixSubMulCPU(int xC, int yC, float* A, float* B, int wA, int wB) {
	
	// Csub is used to store the element
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	for (int i = yC, j = xC; i <= yC + wA - 1; i++, j += wB) {
		// Multiply the two matrices together
		Csub += A[i] * B[j];
	}

	return Csub;
}

bool CheckResult(const dim3& dimsA, const dim3& dimsC, const DATA* h_C) {

	// Check the result
	cout << "Checking computed result for correctness: ";

	bool correct = true;

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps

	for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
		double abs_err = fabs(h_C[i] - (dimsA.x * ValB));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > EPS) {
			cerr << "Error! Matrix[" << i << " ]=" << h_C[i]
				<< " , ref=" << dimsA.x * ValB
				<< " error term is > " << EPS << endl;
			correct = false;
		}
	}

	if (correct) {
		cout << "Result = PASS" << endl;
	}
	else {
		cerr << "Result = FAIL" << endl;
	}

	return correct;
}

void constantInit(DATA dataArr[], const int arrSize, const DATA value) {

	for (int i = 0; i < arrSize; i++) {
		dataArr[i] = value;
	}
}