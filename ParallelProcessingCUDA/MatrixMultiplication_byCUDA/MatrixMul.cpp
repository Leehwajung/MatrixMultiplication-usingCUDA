#include "MatrixMul.h"

// System includes
#include <iostream>
#include <cassert>

// CUDA runtime
#include "nvrtc_helper.h"
#include <cudaProfiler.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "cublas_v2.h"

// Machine zero
#define EPS		1.e-6

// Initial values of matrices A and B
#define	ValA	1.0f
#define	ValB	0.01f

// Repeat number of matrix multiplication
#define NLTER	10

// For to access "MatrixMul_kernel.cu"
#define CU_MatrixMul_Kernel			"matrixMul_kernel.cu"
#define FN_matrixMulCUDA_block16	"matrixMulCUDA_block16"
#define FN_matrixMulCUDA_block32	"matrixMulCUDA_block32"

// Used names
using std::cout;
using std::cerr;
using std::endl;
#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif


int matrixMultiplyUsingCUDA(const dim3 &dimsA, const dim3 &dimsB, const int blockSize) {

	// Allocate host memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(DATA);
	DATA *h_A = new DATA[memSizeA];

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(DATA);
	DATA *h_B = new DATA[memSizeB];

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memSizeC = dimsC.x * dimsC.y * sizeof(DATA);
	DATA *h_C = new DATA[memSizeC];

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
	delete kernelFile;
	delete ptx;

	// Allocate device memory
	CUdeviceptr d_A, d_B, d_C;
	checkCudaErrors(cuMemAlloc(&d_A, memSizeA));
	checkCudaErrors(cuMemAlloc(&d_B, memSizeB));
	checkCudaErrors(cuMemAlloc(&d_C, memSizeC));

	// copy host memory to device
	checkCudaErrors(cuMemcpyHtoD(d_A, h_A, memSizeA));
	checkCudaErrors(cuMemcpyHtoD(d_B, h_B, memSizeB));

	// Get kernel function
	cout << "Computing result using CUDA Kernel..." << endl;
	
	CUfunction fn_matrixMulCUDA;
	if (blockSize == 16) {
		checkCudaErrors(cuModuleGetFunction(&fn_matrixMulCUDA,
			mod_kernel, FN_matrixMulCUDA_block16));
	}
	else {
		checkCudaErrors(cuModuleGetFunction(&fn_matrixMulCUDA,
			mod_kernel, FN_matrixMulCUDA_block32));
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
		StopWatchWin timer;
		timer.start();

		checkCudaErrors(cuLaunchKernel(fn_matrixMulCUDA,
			grid.x, grid.y, grid.z,				/* grid dim */
			threads.x, threads.y, threads.z,	/* block dim */
			0, 0,								/* shared mem, stream */
			&args[0],							/* arguments */
			NULL));

		timer.stop();
		checkCudaErrors(cuCtxSynchronize());
		
		totalLeadTime += timer.getTime();
		cout << "Lead time (" << i << "): " << timer.getTime() << endl;
	}
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;

	// Copy result from device to host
	checkCudaErrors(cuMemcpyDtoH(h_C, d_C, memSizeC));
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Clean up memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

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

int matrixMultiplyOnCPU(const dim3 &dimsA, const dim3 &dimsB) {

	// Allocate memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(DATA);
	DATA *h_A = new DATA[memSizeA];

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(DATA);
	DATA *h_B = new DATA[memSizeB];

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memSizeC = dimsC.x * dimsC.y * sizeof(DATA);
	DATA *h_C = new DATA[memSizeC];

	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		cerr << "Failed to allocate matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);

	// Invoke the multiply function with timer
	cout << "Computing result using CPU..." << endl;

	float totalLeadTime = 0;
	for (int i = 0; i < NLTER; i++) {
		StopWatchWin timer;
		timer.start();
		matrixMulCPU(h_C, h_A, h_B, dimsC.x, dimsC.y, dimsA.x);
		timer.stop();

		totalLeadTime += timer.getTime();
		cout << "Lead time (" << i << "): " << timer.getTime() << endl;
	}
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;

	// Check the result
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Clean up memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	if (correct) {
		return EXIT_SUCCESS;
	}
	else {
		return EXIT_FAILURE;
	}
}

void matrixMulCPU(DATA* C, DATA* A, DATA* B, int wC, int hC, int wA) {

	for (int i = 0; i< hC; i++) {
		for (int j = 0; j < wC; j++) {
			//C[i * wC + j] = matrixSubMulCPU(i, j, A, B, wA, wC);
			// Csub is used to store the element
			DATA Csub = 0;

			// Loop over all the sub-matrices of A and B
			for (int k = j, l = i; k <= j + wA - 1; k++, l += wC) {
				// Multiply the two matrices together
				Csub += A[k] * B[l];
			}
			C[i * wC + j] = Csub;
		}
	}
}

int matrixMultiplyUsingCUBLAS(const dim3 &dimsA, const dim3 &dimsB) {

	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	cudaError_t error;
	int devID = 0;
	int iSizeMultiple;

	//if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	//{
	//	devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		error = cudaSetDevice(devID);

	//	if (error != cudaSuccess)
	//	{
	//		printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
	//		exit(EXIT_FAILURE);
	//	}
	//}

	// get number of SMs on this GPU
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaDeviceProp deviceProp;

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

	// use a larger block size for Fermi and above
	//int blockSize = (deviceProp.major < 2) ? 16 : 32;


	// 여기까지 초기화
	//--------------------------------------------------------------------------------/




	// Allocate host memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(DATA);
	DATA *h_A = new DATA[memSizeA];

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(DATA);
	DATA *h_B = new DATA[memSizeB];

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memSizeC = dimsC.x * dimsC.y * sizeof(DATA);
	DATA *h_C = new DATA[memSizeC];

	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		cerr << "Failed to allocate host matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize host memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);
	


	// Initialize device and get moudule
	//char *kernelFile = sdkFindFilePath(CU_MatrixMul_Kernel, NULL);
	//char *ptx;
	//size_t ptxSize;
	//compileFileToPTX(kernelFile, 0, NULL, &ptx, &ptxSize);
	//CUmodule mod_kernel = loadPTX(ptx, 0, NULL);
	//delete kernelFile;
	//delete ptx;

	// Allocate device memory
	CUdeviceptr d_A, d_B, d_C;
	checkCudaErrors(cuMemAlloc(&d_A, memSizeA));
	checkCudaErrors(cuMemAlloc(&d_B, memSizeB));
	checkCudaErrors(cuMemAlloc(&d_C, memSizeC));
	//float *d_A, *d_B, *d_C;
	//checkCudaErrors(cudaMalloc((void **)&d_A, memSizeA));
	//checkCudaErrors(cudaMalloc((void **)&d_B, memSizeB));
	//checkCudaErrors(cudaMalloc((void **)&d_C, memSizeC));

	// copy host memory to device
	checkCudaErrors(cuMemcpyHtoD(d_A, h_A, memSizeA));
	checkCudaErrors(cuMemcpyHtoD(d_B, h_B, memSizeB));
	//checkCudaErrors(cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_B, h_B, memSizeB, cudaMemcpyHostToDevice));
	////////////////////////////////////////////////////////////

	// setup execution parameters
	dim3 threads(32, 32);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);



	// Get kernel function
	cout << "Computing result using CUBLAS..." << endl;
	/////////////////////////////////////////////////////
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasHandle_t handle;
	cudaEvent_t start, stop;

	checkCudaErrors(cublasCreate(&handle));

	// Perform warmup operation with cublas
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsB.x, dimsA.y, dimsA.x, &alpha, (DATA*)&d_B, dimsB.x, (DATA*)&d_A, dimsA.x, &beta, (DATA*)&d_C, dimsB.x));
	
	// Allocate CUDA events that we'll use for timing
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));





	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));

	float totalLeadTime = 0;
	for (int i = 0; i < NLTER; i++) {
		StopWatchWin timer;
		timer.start();

		// note cublas is column primary!
		// need to transpose the order
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsB.x, dimsA.y, dimsA.x, &alpha, (DATA*)&d_B, dimsB.x, (DATA*)&d_A, dimsA.x, &beta, (DATA*)&d_C, dimsB.x));

		timer.stop();
		totalLeadTime += timer.getTime();
		cout << "Lead time (" << i << "): " << timer.getTime() << endl;
	}
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


	//////////////////////////////////////////////////////
	//CUfunction fn_matrixMulCUDA;
	//if (blockSize == 16) {
	//	checkCudaErrors(cuModuleGetFunction(&fn_matrixMulCUDA,
	//		mod_kernel, FN_matrixMulCUDA_block16));
	//}
	//else {
	//	checkCudaErrors(cuModuleGetFunction(&fn_matrixMulCUDA,
	//		mod_kernel, FN_matrixMulCUDA_block32));
	//}

	//// Setup execution parameters
	//dim3 threads(blockSize, blockSize);
	//dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	//void *args[] = {
	//	(void*)&d_C,
	//	(void*)&d_A,
	//	(void*)&d_B,
	//	(void*)&dimsA.x,
	//	(void*)&dimsB.x
	//};

	//// Execute the kernel with timer
	//for (int i = 0; i < NLTER; i++) {

	//	checkCudaErrors(cuLaunchKernel(fn_matrixMulCUDA,
	//		grid.x, grid.y, grid.z,				/* grid dim */
	//		threads.x, threads.y, threads.z,	/* block dim */
	//		0, 0,								/* shared mem, stream */
	//		&args[0],							/* arguments */
	//		NULL));
	//}
	////////////////////////////////////////////////////////////

	// Copy result from device to host
	checkCudaErrors(cuMemcpyDtoH(h_C, d_C, memSizeC));
	//checkCudaErrors(cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost));
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Destroy the handle
	checkCudaErrors(cublasDestroy(handle));

	// Clean up memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	checkCudaErrors(cuMemFree(d_A));
	checkCudaErrors(cuMemFree(d_B));
	checkCudaErrors(cuMemFree(d_C));
	//checkCudaErrors(cudaFree(d_A));
	//checkCudaErrors(cudaFree(d_B));
	//checkCudaErrors(cudaFree(d_C));

	cudaDeviceReset();

	if (correct) {
		return EXIT_SUCCESS;
	}
	else {
		return EXIT_FAILURE;
	}
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










//////////////--------------------------------------------------------/////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(const dim3 &dimsA, const dim3 &dimsB, int blockSize)
{
	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	cudaError_t error = cudaSuccess;
	int devID = 0;

	// get number of SMs on this GPU

	//iSizeMultiple = min(iSizeMultiple, 10);
	//iSizeMultiple = max(iSizeMultiple, 1);

	cudaDeviceProp deviceProp;

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	//// use a larger block size for Fermi and above
	//blockSize = (deviceProp.major < 2) ? 16 : 32;

	// Allocate host memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(DATA);
	DATA *h_A = new DATA[memSizeA];

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(DATA);
	DATA *h_B = new DATA[memSizeB];

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memSizeC = dimsC.x * dimsC.y * sizeof(DATA);
	DATA *h_C = new DATA[memSizeC];

	printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
		dimsA.y, dimsA.x,
		dimsB.y, dimsB.x,
		dimsC.y, dimsC.x);
	if (dimsA.x != dimsB.y ||
		dimsA.y != dimsC.y ||
		dimsB.x != dimsC.x)
	{
		printf("ERROR: Matrix sizes do not match!\n");
		exit(-1);
	}

	// Initialize host memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);
	//////////////////////////////////////////////

	// allocate device memory
	float *d_A, *d_B, *d_C;
	unsigned int sizeC = dimsC.x * dimsC.y;

	// allocate host memory for the result
	//float *h_CUBLAS = (float *)malloc(mem_size_C);

	checkCudaErrors(cudaMalloc((void **)&d_A, memSizeA));
	checkCudaErrors(cudaMalloc((void **)&d_B, memSizeB));
	checkCudaErrors(cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, memSizeB, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&d_C, memSizeC));
	////////////////////////////////////////////////////////////



	// setup execution parameters
	dim3 threads(blockSize, blockSize);
	dim3 grid(dimsC.x / threads.x, dimsC.y / threads.y);

	// create and start timer
	printf("Computing result using CUBLAS...");

	// execute the kernel
	int nIter = 30;

	// CUBLAS version 2.0
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		cublasHandle_t handle;
		cudaEvent_t start, stop;

		checkCudaErrors(cublasCreate(&handle));

		//Perform warmup operation with cublas
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsB.x, dimsA.y, dimsA.x, &alpha, d_B, dimsB.x, d_A, dimsA.x, &beta, d_C, dimsB.x));

		// Allocate CUDA events that we'll use for timing
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// Record the start event
		checkCudaErrors(cudaEventRecord(start, NULL));

		for (int j = 0; j < nIter; j++)
		{
			//note cublas is column primary!
			//need to transpose the order
			checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsB.x, dimsA.y, dimsA.x, &alpha, d_B, dimsB.x, d_A, dimsA.x, &beta, d_C, dimsB.x));

		}

		printf("done.\n");

		// Record the stop event
		checkCudaErrors(cudaEventRecord(stop, NULL));

		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		// Compute and print the performance
		float msecPerMatrixMul = msecTotal / nIter;
		double flopsPerMatrixMul = 2.0 * (double)dimsC.y * (double)dimsC.x * (double)dimsB.y;
		double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
		printf(
			"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
			gigaFlops,
			msecPerMatrixMul,
			flopsPerMatrixMul);

		// copy result from device to host
		checkCudaErrors(cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost));

		// Destroy the handle
		checkCudaErrors(cublasDestroy(handle));
	}

	// compute reference solution
	//printf("Computing result using host CPU...");
	//float *reference = (float *)malloc(mem_size_C);
	//matrixMulCPU(reference, h_A, h_B, dimsA.y, dimsA.x, dimsB.x);
	//printf("done.\n");

	// check result (CUBLAS)
	//bool resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS, sizeC, 1.0e-6f);

	//if (resCUBLAS != true)
	//{
	//    printDiff(reference, h_CUBLAS, dimsC.x, dimsC.y, 100, 1.0e-5f);
	//}

	//printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");

	printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

	// clean up memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	//free(reference);
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	if (/*resCUBLAS*/true == true)
	{
		return EXIT_SUCCESS;    // return value = 1
	}
	else
	{
		return EXIT_FAILURE;     // return value = 0
	}
}