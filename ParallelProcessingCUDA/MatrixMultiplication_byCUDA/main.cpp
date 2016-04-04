#include <iostream>
#include "MatrixMul.h"
using namespace std;

int compareCUDAandCPUMultiplycation(const dim3& dimsA, const dim3& dimsB);


/**
 * Program main
 */
int main() {

	dim3 dimsL(10 * BLOCKSIZE, 10 * BLOCKSIZE, 1);
	dim3 dimsH(100 * BLOCKSIZE, 100 * BLOCKSIZE, 1);

	dim3 dimsA(10 * BLOCKSIZE, 10 * BLOCKSIZE, 1);
	dim3 dimsB(20 * BLOCKSIZE, 10 * BLOCKSIZE, 1);

	// matrix size 320 * 320
	//int matrixResult = compareCUDAandCPUMultiplycation(dimsH, dimsH);
	//if (matrixResult) {
	//	exit(matrixResult);
	//}

	//cout << endl << "-----" << endl;

	//// matrix size 3200 * 3200
	//matrixResult = compareCUDAandCPUMultiplycation(dimsH, dimsH);
	//exit(matrixResult);

	printf("[Matrix Multiply CUBLAS] - Starting...\n");

	//initializeCUDA(NULL, NULL, devID, sizeMult, matrix_size);

	int matrix_result = matrixMultiply(dimsA, dimsB);

	return matrix_result;
}


int compareCUDAandCPUMultiplycation(const dim3& dimsA, const dim3& dimsB) {

	if (dimsA.x != dimsB.y) {
		cerr << "Error: outer matrix dimensions must be equal. (" << dimsA.x << " != " << dimsB.y << ")" << endl;
		exit(EXIT_FAILURE);
	}
	cout << "MatrixA(" << dimsA.x << "," << dimsA.y << "), MatrixB(" << dimsB.x << "," << dimsB.y << ")" << endl;

	int matrixResult = matrixMultiplyUsingCUBLAS(dimsA, dimsB);
	if (matrixResult) {
		return(matrixResult);
	}

	cout << endl;

	matrixResult = matrixMultiplyUsingCUDA(dimsA, dimsB);
	if (matrixResult) {
		return(matrixResult);
	}

	return matrixResult;// matrixMultiplyOnCPU(dimsA, dimsB);
}